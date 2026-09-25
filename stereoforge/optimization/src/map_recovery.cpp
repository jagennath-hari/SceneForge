#include "stereoforge/optimization/map_builder.hpp"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <optional>
#include <random>
#include <tuple>

namespace stereoforge::optimization {
namespace {
constexpr double kPi = 3.14159265358979323846;
using Exclusions = std::set<std::pair<TrackId, FrameId>>;
struct Correspondence {
    TrackId track;
    Eigen::Vector3d position;
    Eigen::Vector2d pixel;
};
std::size_t Coverage(const std::vector<Correspondence>& pairs, const Camera& camera) {
    std::set<std::pair<int,int>> cells;
    for (const Correspondence& pair : pairs) {
        cells.emplace(std::clamp(static_cast<int>(4*pair.pixel.x()/camera.width),0,3),
                      std::clamp(static_cast<int>(4*pair.pixel.y()/camera.height),0,3));
    }
    return cells.size();
}
Eigen::Vector3d Ray(const Camera& camera, const Eigen::Vector2d& pixel) {
    return (camera.rotation*camera.intrinsics.inverse()*Eigen::Vector3d(pixel.x(),pixel.y(),1)).normalized();
}
double Angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::acos(std::clamp(a.dot(b),-1.0,1.0))*180/kPi;
}
std::optional<Eigen::Vector3d> FitPoint(const Observations& observations, const SparseMap& map) {
    Eigen::MatrixXd rows(2*observations.size(),4);
    Eigen::Index row = 0;
    // Translate the triangulation origin to avoid unnecessary large coordinates.
    const Eigen::Vector3d origin = map.cameras.at(observations.begin()->first).center;
    for (const std::pair<const FrameId,Eigen::Vector2d>& obs : observations) {
        const Camera& camera = map.cameras.at(obs.first);
        Eigen::Matrix<double,3,4> projection;
        projection.leftCols<3>() = camera.rotation.transpose();
        projection.rightCols<1>() = -camera.rotation.transpose()*(camera.center-origin);
        const Eigen::Vector3d ray = camera.intrinsics.inverse()*Eigen::Vector3d(obs.second.x(),obs.second.y(),1);
        rows.row(row++) = ray.x()*projection.row(2)-projection.row(0);
        rows.row(row++) = ray.y()*projection.row(2)-projection.row(1);
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(rows,Eigen::ComputeFullV);
    const Eigen::Vector4d h = svd.matrixV().col(3);
    if (!h.allFinite() || std::abs(h.w()) < 1e-10) { return std::nullopt; }
    const Eigen::Vector3d point = h.head<3>()/h.w()+origin;
    if (!point.allFinite()) { return std::nullopt; }
    return point;
}
std::optional<Landmark> Triangulate(const Observations& observations, const SparseMap& map) {
    if (observations.size() < 3) { return std::nullopt; }
    std::vector<FrameId> ids;
    for (const std::pair<const FrameId,Eigen::Vector2d>& obs : observations) { ids.push_back(obs.first); }
    std::vector<FrameId> sampled;
    const std::size_t count = std::min<std::size_t>(12,ids.size());
    for (std::size_t i = 0; i < count; ++i) { sampled.push_back(ids[i*(ids.size()-1)/(count-1)]); }
    std::vector<std::tuple<double,FrameId,FrameId>> hypotheses;
    for (std::size_t i = 0; i < sampled.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const FrameId a = sampled[i], b = sampled[j];
            const double angle = Angle(Ray(map.cameras.at(a),observations.at(a)),Ray(map.cameras.at(b),observations.at(b)));
            if (angle >= 1 && angle <= 90) { hypotheses.emplace_back(angle,a,b); }
        }
    }
    std::sort(hypotheses.rbegin(),hypotheses.rend());
    Observations best;
    double best_error = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < std::min<std::size_t>(16,hypotheses.size()); ++i) {
        const FrameId a = std::get<1>(hypotheses[i]), b = std::get<2>(hypotheses[i]);
        const std::optional<Eigen::Vector3d> xyz = FitPoint({{a,observations.at(a)},{b,observations.at(b)}},map);
        if (!xyz) { continue; }
        Observations valid;
        double error = 0;
        for (const std::pair<const FrameId,Eigen::Vector2d>& obs : observations) {
            const double residual = Reprojection(map.cameras.at(obs.first),*xyz,obs.second);
            if (residual <= 4) { valid.emplace(obs); error += residual; }
        }
        if (valid.size() > best.size() || (valid.size() == best.size() && error < best_error)) {
            best = std::move(valid); best_error = error;
        }
    }
    if (best.size() < 3) { return std::nullopt; }
    const std::optional<Eigen::Vector3d> xyz = FitPoint(best,map);
    if (!xyz) { return std::nullopt; }
    std::erase_if(best,[&](const std::pair<const FrameId,Eigen::Vector2d>& obs) {
        return Reprojection(map.cameras.at(obs.first),*xyz,obs.second) > 4;
    });
    if (best.size() < 3) { return std::nullopt; }
    std::vector<Eigen::Vector3d> rays;
    double maximum_angle = 0;
    for (const std::pair<const FrameId,Eigen::Vector2d>& obs : best) {
        const Eigen::Vector3d ray = Ray(map.cameras.at(obs.first),obs.second);
        for (const Eigen::Vector3d& previous : rays) { maximum_angle = std::max(maximum_angle,Angle(previous,ray)); }
        rays.push_back(ray);
    }
    if (maximum_angle < 1 || maximum_angle > 90) { return std::nullopt; }
    return Landmark{*xyz,{0,0,0},std::move(best)};
}
Camera RegisterCamera(const Camera& calibration, const std::vector<Correspondence>& train,
                      const std::vector<Correspondence>& held, FrameId frame) {
    if (train.size() < 30 || held.size() < 10 || Coverage(train,calibration) < 6 || Coverage(held,calibration) < 3) {
        throw std::runtime_error("PnP recovery frame " + std::to_string(frame) +
            ": insufficient map support/coverage (" + std::to_string(train.size()) + " training, " +
            std::to_string(held.size()) + " held out; need 30/10 and 6/3 grid cells)");
    }
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    for (const Correspondence& pair : train) { origin += pair.position; } origin /= train.size();
    double scale = 0;
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const Correspondence& pair : train) {
        const Eigen::Vector3d centered = pair.position-origin;
        scale += centered.squaredNorm(); covariance += centered*centered.transpose();
    }
    scale = std::sqrt(scale/train.size());
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance);
    if (!std::isfinite(scale) || scale <= 1e-10 || svd.singularValues()[1] < 1e-6*svd.singularValues()[0]) {
        throw std::runtime_error("PnP recovery has degenerate landmark layout at frame " + std::to_string(frame));
    }
    std::vector<cv::Point3d> objects;
    std::vector<cv::Point2d> pixels;
    for (const Correspondence& pair : train) {
        const Eigen::Vector3d xyz = (pair.position-origin)/scale;
        objects.emplace_back(xyz.x(),xyz.y(),xyz.z()); pixels.emplace_back(pair.pixel.x(),pair.pixel.y());
    }
    cv::Mat k(3,3,CV_64F);
    for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) { k.at<double>(r,c) = calibration.intrinsics(r,c); } }
    cv::Mat rotation, translation, inliers;
    if (!cv::solvePnPRansac(objects,pixels,k,cv::noArray(),rotation,translation,false,1000,4.0f,.999,inliers,cv::SOLVEPNP_EPNP) ||
        inliers.total() < 20 || inliers.total() < .5*train.size()) {
        throw std::runtime_error("PnP RANSAC failed at frame " + std::to_string(frame));
    }
    std::vector<cv::Point3d> inlier_objects;
    std::vector<cv::Point2d> inlier_pixels;
    for (int i = 0; i < inliers.rows; ++i) {
        const int index = inliers.at<int>(i,0);
        inlier_objects.push_back(objects.at(index)); inlier_pixels.push_back(pixels.at(index));
    }
    cv::solvePnPRefineLM(inlier_objects,inlier_pixels,k,cv::noArray(),rotation,translation,
                        cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,50,1e-9));
    cv::Mat matrix;
    cv::Rodrigues(rotation,matrix);
    Eigen::Matrix3d r;
    Eigen::Vector3d t;
    for (int row = 0; row < 3; ++row) {
        t[row] = translation.at<double>(row,0);
        for (int col = 0; col < 3; ++col) { r(row,col) = matrix.at<double>(row,col); }
    }
    Camera camera = calibration;
    camera.rotation = r.transpose(); camera.center = origin-scale*r.transpose()*t;
    if (!camera.rotation.allFinite() || !camera.center.allFinite() || std::abs(camera.rotation.determinant()-1) > 1e-3) {
        throw std::runtime_error("PnP returned an invalid pose");
    }
    std::vector<Correspondence> valid;
    for (const Correspondence& pair : train) { if (Reprojection(camera,pair.position,pair.pixel) <= 4) { valid.push_back(pair); } }
    if (valid.size() < 20 || valid.size() < .5*train.size() || Coverage(valid,camera) < 6) {
        throw std::runtime_error("PnP refined training support failed at frame " + std::to_string(frame));
    }
    std::size_t passed = 0;
    for (const Correspondence& pair : held) { passed += Reprojection(camera,pair.position,pair.pixel) <= 5; }
    if (passed < .8*held.size()) { throw std::runtime_error("PnP held-out support below 80% at frame " + std::to_string(frame)); }
    return camera;
}
}
SparseMap MapBuilder::Recover(const std::vector<DepthFrame>& frames, const SparseMap& local,
                              Boundary& recovery_checks, const std::function<void(const std::string&)>& progress) const {
    SparseMap candidate = this->map_;
    Exclusions excluded;
    for (const Boundary& boundary : this->boundaries_) {
        for (const Holdout& held : boundary.observations) { excluded.emplace(held.track,held.frame); }
    }
    std::map<FrameId,const DepthFrame*> images;
    for (const DepthFrame& frame : frames) { images.emplace(frame.id,&frame); }
    // Triangulate only absent tracks, in the existing gauge. Existing depths are
    // never replaced by the independently reconstructed incoming window.
    const std::function<void(FrameId)> extend = [&](FrameId frame) {
        if (!this->frame_tracks_.contains(frame)) { return; }
        for (const TrackId id : this->frame_tracks_.at(frame)) {
            const Observations& track = this->tracks_.at(id);
            if (candidate.landmarks.contains(id)) {
                Landmark& point = candidate.landmarks.at(id);
                if (!excluded.contains({id,frame}) && Reprojection(candidate.cameras.at(frame),point.position,track.at(frame)) <= 4) {
                    point.observations.emplace(frame,track.at(frame));
                }
                continue;
            }
            Observations available;
            for (const std::pair<const FrameId,Eigen::Vector2d>& obs : track) {
                if (candidate.cameras.contains(obs.first) && !excluded.contains({id,obs.first})) { available.emplace(obs); }
            }
            std::optional<Landmark> point = Triangulate(available,candidate);
            if (!point) { continue; }
            bool colored = false;
            for (const std::pair<const FrameId,Eigen::Vector2d>& obs : point->observations) {
                if (!images.contains(obs.first)) { continue; }
                const DepthFrame& image = *images.at(obs.first);
                const int x = std::clamp(static_cast<int>(std::nearbyint(obs.second.x())),0,image.camera.width-1);
                const int y = std::clamp(static_cast<int>(std::nearbyint(obs.second.y())),0,image.camera.height-1);
                for (int c = 0; c < 3; ++c) { point->color[c] = image.rgb[3*(y*image.camera.width+x)+c]; }
                colored = true; break;
            }
            if (colored) { candidate.landmarks.emplace(id,std::move(*point)); }
        }
    };
    for (const std::pair<const FrameId,const DepthFrame*>& entry : images) {
        if (candidate.cameras.contains(entry.first)) { extend(entry.first); }
    }
    for (const std::pair<const FrameId,const DepthFrame*>& entry : images) {
        const FrameId frame = entry.first;
        if (candidate.cameras.contains(frame)) { continue; }
        progress("PnP/RANSAC recovery: frame " + std::to_string(frame));
        std::vector<Correspondence> pairs;
        if (this->frame_tracks_.contains(frame)) {
            for (const TrackId id : this->frame_tracks_.at(frame)) {
                if (!candidate.landmarks.contains(id) || excluded.contains({id,frame})) { continue; }
                const Landmark& point = candidate.landmarks.at(id);
                if (point.observations.size() < 3 || !point.position.allFinite()) { continue; }
                const Eigen::Vector2d uv = this->tracks_.at(id).at(frame);
                const Camera& calibration = local.cameras.at(frame);
                if (uv.x() < 0 || uv.y() < 0 || uv.x() >= calibration.width || uv.y() >= calibration.height) { continue; }
                pairs.push_back({id,point.position,uv});
            }
        }
        std::mt19937 generator(static_cast<std::uint32_t>(frame));
        std::shuffle(pairs.begin(),pairs.end(),generator);
        std::vector<Correspondence> train, held;
        for (std::size_t i = 0; i < pairs.size(); ++i) { (i%4 ? train : held).push_back(pairs[i]); }
        const Camera camera = RegisterCamera(local.cameras.at(frame),train,held,frame);
        candidate.cameras.emplace(frame,camera);
        recovery_checks.shared.insert(frame);
        for (const Correspondence& pair : held) {
            excluded.emplace(pair.track,frame);
            recovery_checks.observations.push_back({pair.track,frame,pair.pixel,candidate.landmarks.at(pair.track).observations});
        }
        // Only training inliers enter BA; held-out pixels remain excluded even
        // during subsequent triangulation or extension rounds.
        for (const Correspondence& pair : train) {
            if (Reprojection(camera,pair.position,pair.pixel) <= 4) { candidate.landmarks.at(pair.track).observations.emplace(frame,pair.pixel); }
        }
        progress("triangulating recovered frame " + std::to_string(frame));
        extend(frame);
    }
    for (const std::pair<const FrameId,Camera>& camera : candidate.cameras) { recovery_checks.cameras.insert(camera.first); }
    return candidate;
}
} // namespace stereoforge::optimization
