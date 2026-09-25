#include "stereoforge/optimization/map_builder.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace stereoforge::optimization {
namespace {
constexpr double kPi = 3.14159265358979323846;
using Parameters = Eigen::Matrix<double, 7, 1>;
double Median(std::vector<double> v) {
    if (v.empty()) { throw std::runtime_error("Empty alignment statistic"); }
    std::sort(v.begin(), v.end());
    return v.size()%2 ? v[v.size()/2] : (v[v.size()/2-1]+v[v.size()/2])/2;
}
bool Reliable(const SparseMap& model, const Landmark& point) {
    if (point.observations.size() < 3) { return false; }
    std::vector<Eigen::Vector3d> rays;
    std::vector<double> errors;
    for (const std::pair<const FrameId, Eigen::Vector2d>& obs : point.observations) {
        const Camera& camera = model.cameras.at(obs.first);
        const Eigen::Vector3d ray = (camera.rotation*camera.intrinsics.inverse()*Eigen::Vector3d(obs.second.x(),obs.second.y(),1)).normalized();
        const double error = Reprojection(camera, point.position, obs.second);
        if (!ray.allFinite() || !std::isfinite(error) || error > 3) { return false; }
        rays.push_back(ray); errors.push_back(error);
    }
    if (Median(errors) > 1.5) { return false; }
    double cosine = 1;
    for (std::size_t i = 0; i < rays.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) { cosine = std::min(cosine, rays[i].dot(rays[j])); }
    }
    const double angle = std::acos(std::clamp(cosine,-1.0,1.0))*180/kPi;
    return angle >= 5 && angle <= 90;
}
Similarity Fit(const std::vector<Eigen::Vector3d>& source, const std::vector<Eigen::Vector3d>& target,
               const std::vector<std::size_t>& indices) {
    Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
    for (const std::size_t i : indices) { a += source[i]; b += target[i]; }
    a /= indices.size(); b /= indices.size();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double variance = 0;
    for (const std::size_t i : indices) { covariance += (target[i]-b)*(source[i]-a).transpose(); variance += (source[i]-a).squaredNorm(); }
    covariance /= indices.size(); variance /= indices.size();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU|Eigen::ComputeFullV);
    const Eigen::Vector3d singular = svd.singularValues();
    if (singular[0] <= 0 || singular[1] < singular[0]*1e-6 || variance <= 0) { throw std::runtime_error("Degenerate Sim(3) support"); }
    Eigen::Vector3d signs(1,1,(svd.matrixU()*svd.matrixV().transpose()).determinant() < 0 ? -1 : 1);
    Similarity result;
    result.rotation = svd.matrixU()*signs.asDiagonal()*svd.matrixV().transpose();
    result.scale = singular.dot(signs)/variance;
    result.translation = b-result.scale*result.rotation*a;
    if (!std::isfinite(result.scale) || result.scale <= 0 || !result.translation.allFinite()) { throw std::runtime_error("Invalid Sim(3)"); }
    return result;
}
Eigen::Matrix3d Exp(const Eigen::Vector3d& v) {
    const double angle = v.norm();
    if (angle < 1e-16) { return Eigen::Matrix3d::Identity(); }
    return Eigen::AngleAxisd(angle, v/angle).toRotationMatrix();
}
void CheckCoverage(const SparseMap& model, const std::vector<TrackId>& ids, const std::vector<std::size_t>& indices,
                   const std::vector<FrameId>& shared) {
    std::size_t supported = 0;
    for (const FrameId frame : shared) {
        const Camera& camera = model.cameras.at(frame);
        std::set<std::pair<int,int>> cells;
        std::size_t count = 0;
        for (const std::size_t i : indices) {
            const Observations& obs = model.landmarks.at(ids[i]).observations;
            if (!obs.contains(frame)) { continue; }
            const Eigen::Vector2d uv = obs.at(frame);
            cells.emplace(std::clamp(static_cast<int>(4*uv.x()/camera.width),0,3), std::clamp(static_cast<int>(4*uv.y()/camera.height),0,3));
            ++count;
        }
        supported += count >= 3 && cells.size() >= 3;
    }
    if (supported < 6) { throw std::runtime_error("Alignment landmarks lack six-camera image coverage"); }
}
// Seven-dimensional CPU solve. The objective matches the previous camera-aware
// soft-L1 group means; Eigen handles the tiny dense normal system. BA remains CUDA.
Similarity Refine(const Similarity& initial, const std::vector<Eigen::Vector3d>& source,
                  const std::vector<Eigen::Vector3d>& target, const std::vector<std::size_t>& train,
                  const SparseMap& reference, const SparseMap& local, const std::vector<FrameId>& shared, double depth) {
    std::vector<FrameId> cameras;
    for (std::size_t i = 0; i < shared.size(); ++i) { if (i%3) { cameras.push_back(shared[i]); } }
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const std::size_t i : train) { center += source[i]; } center /= train.size();
    const Eigen::Vector3d anchor = initial.scale*initial.rotation*center+initial.translation;
    const std::function<Similarity(const Parameters&)> decode = [&](const Parameters& p) {
        Similarity s;
        s.scale = initial.scale*std::exp(p[0]);
        s.rotation = Exp(p.segment<3>(1))*initial.rotation;
        s.translation = anchor+depth*p.tail<3>()-s.scale*s.rotation*center;
        return s;
    };
    const std::function<Eigen::VectorXd(const Parameters&)> residual = [&](const Parameters& p) {
        const Similarity s = decode(p);
        Eigen::VectorXd r(3*(train.size()+2*cameras.size()));
        Eigen::Index offset = 0;
        const std::function<void(const Eigen::Vector3d&, std::size_t)> append = [&](const Eigen::Vector3d& value, std::size_t count) {
            const double weight = std::sqrt(2/(std::sqrt(1+value.squaredNorm())+1))/std::sqrt(static_cast<double>(count));
            r.segment<3>(offset) = weight*value; offset += 3;
        };
        for (const std::size_t i : train) { append((s.scale*s.rotation*source[i]+s.translation-target[i])/(.05*depth),train.size()); }
        for (const FrameId f : cameras) { append((s.scale*s.rotation*local.cameras.at(f).center+s.translation-reference.cameras.at(f).center)/(.05*depth),cameras.size()); }
        for (const FrameId f : cameras) {
            const Eigen::AngleAxisd aa(reference.cameras.at(f).rotation.transpose()*s.rotation*local.cameras.at(f).rotation);
            append(aa.axis()*aa.angle()/(10*kPi/180),cameras.size());
        }
        return r;
    };
    Parameters p = Parameters::Zero();
    Eigen::VectorXd r = residual(p);
    double damping = 1e-3;
    bool converged = false;
    for (int iteration = 0; iteration < 200; ++iteration) {
        Eigen::MatrixXd jacobian(r.size(),7);
        for (int column = 0; column < 7; ++column) {
            const double h = 1e-6*std::max(1.0,std::abs(p[column]));
            Parameters plus = p, minus = p; plus[column] += h; minus[column] -= h;
            jacobian.col(column) = (residual(plus)-residual(minus))/(2*h);
        }
        const Eigen::Matrix<double,7,7> normal = jacobian.transpose()*jacobian;
        const Parameters gradient = jacobian.transpose()*r;
        if (gradient.lpNorm<Eigen::Infinity>() < 1e-9) { converged = true; break; }
        Eigen::Matrix<double,7,7> damped = normal;
        for (int i = 0; i < 7; ++i) { damped(i,i) += damping*std::max(1.0,normal(i,i)); }
        const Parameters step = damped.ldlt().solve(-gradient);
        if (!step.allFinite()) { throw std::runtime_error("Nonfinite camera-aware Sim(3) step"); }
        const Parameters candidate = p+step;
        if (std::abs(candidate[0]) >= std::log(4.0)) { damping *= 10; continue; }
        const Eigen::VectorXd next = residual(candidate);
        if (next.allFinite() && next.squaredNorm() < r.squaredNorm()) {
            const double improvement = r.squaredNorm()-next.squaredNorm();
            const double old_cost = r.squaredNorm();
            p = candidate; r = next; damping = std::max(1e-12,damping/3);
            if (improvement <= 1e-9*old_cost || step.norm() < 1e-9*(1e-9+p.norm())) { converged = true; break; }
        } else {
            // Stationary finite objective, not a large damping-induced small step.
            if (step.norm() < 1e-9*(1e-9+p.norm()) && damping < 1e-3) { converged = true; break; }
            damping *= 10;
        }
    }
    if (!converged || !r.allFinite() || std::abs(p[0]) >= std::log(4.0)-1e-8) { throw std::runtime_error("Camera-aware Sim(3) did not converge to an interior solution"); }
    return decode(p);
}
}
Similarity Align(const SparseMap& reference, const SparseMap& local) {
    std::vector<FrameId> shared;
    for (const std::pair<const FrameId, Camera>& c : local.cameras) { if (reference.cameras.contains(c.first)) { shared.push_back(c.first); } }
    std::vector<TrackId> ids;
    std::vector<Eigen::Vector3d> source, target;
    std::vector<double> distances;
    std::vector<std::size_t> eligible;
    for (const std::pair<const TrackId, Landmark>& entry : local.landmarks) {
        if (!reference.landmarks.contains(entry.first)) { continue; }
        const Landmark& a = reference.landmarks.at(entry.first);
        const Landmark& b = entry.second;
        bool common = false;
        for (const FrameId f : shared) {
            if (a.observations.contains(f) && b.observations.contains(f)) {
                const Eigen::Vector2d x = a.observations.at(f), y = b.observations.at(f);
                if (std::nearbyint(x.x()*10000) == std::nearbyint(y.x()*10000) && std::nearbyint(x.y()*10000) == std::nearbyint(y.y()*10000)) { common = true; }
            }
        }
        if (!common) { continue; }
        if (Reliable(reference,a) && Reliable(local,b)) { eligible.push_back(ids.size()); }
        ids.push_back(entry.first); source.push_back(b.position); target.push_back(a.position);
        for (const FrameId f : shared) { if (a.observations.contains(f)) { distances.push_back((a.position-reference.cameras.at(f).center).norm()); } }
    }
    if (shared.size() < 6) { throw std::runtime_error("Alignment requires six shared cameras"); }
    if (ids.size() < 60 || eligible.size() < 60) {
        throw InsufficientAlignmentSupport("Insufficient shared alignment support: " + std::to_string(shared.size()) + " cameras, " + std::to_string(eligible.size()) + "/60 depth-observable landmarks");
    }
    const double depth = Median(distances);
    if (!std::isfinite(depth) || depth <= 1e-8) { throw std::runtime_error("Degenerate overlap scale"); }
    std::mt19937 generator(0);
    std::shuffle(eligible.begin(),eligible.end(),generator);
    std::vector<std::size_t> train, held;
    for (std::size_t i = 0; i < eligible.size(); ++i) { (i%3 ? train : held).push_back(eligible[i]); }
    for (const std::vector<std::size_t>* subset : {&train,&held}) { CheckCoverage(reference,ids,*subset,shared); CheckCoverage(local,ids,*subset,shared); }
    std::vector<std::size_t> best;
    std::uniform_int_distribution<std::size_t> choose(0,train.size()-1);
    for (int trial = 0; trial < 512; ++trial) {
        std::set<std::size_t> selected;
        while (selected.size() < 3) { selected.insert(train[choose(generator)]); }
        Similarity candidate;
        try { candidate = Fit(source,target,std::vector<std::size_t>(selected.begin(),selected.end())); }
        catch (const std::runtime_error&) { continue; }
        std::vector<std::size_t> inliers;
        for (const std::size_t i : train) { if ((candidate.scale*candidate.rotation*source[i]+candidate.translation-target[i]).norm() <= .05*depth) { inliers.push_back(i); } }
        if (inliers.size() > best.size()) { best = std::move(inliers); }
    }
    if (best.size() < 20) { throw std::runtime_error("No robust Sim(3) initialization"); }
    const Similarity result = Refine(Fit(source,target,best),source,target,train,reference,local,shared,depth);
    for (const std::vector<std::size_t>* subset : {&train,&held}) {
        std::size_t passed = 0;
        for (const std::size_t i : *subset) { passed += (result.scale*result.rotation*source[i]+result.translation-target[i]).norm() <= .05*depth; }
        if (passed < .8*subset->size()) { throw std::runtime_error("Sim(3) landmark agreement below 80%"); }
    }
    for (const FrameId f : shared) {
        const Camera& a = reference.cameras.at(f); const Camera& b = local.cameras.at(f);
        const double angle = std::acos(std::clamp(((a.rotation.transpose()*result.rotation*b.rotation).trace()-1)/2,-1.0,1.0))*180/kPi;
        if (angle > 10 || (result.scale*result.rotation*b.center+result.translation-a.center).norm() > .05*depth) {
            throw std::runtime_error("Sim(3) shared camera disagreement at frame " + std::to_string(f));
        }
    }
    return result;
}
void Transform(SparseMap& model, const Similarity& transform) {
    for (std::pair<const FrameId, Camera>& entry : model.cameras) {
        entry.second.rotation = transform.rotation*entry.second.rotation;
        entry.second.center = transform.scale*transform.rotation*entry.second.center+transform.translation;
    }
    for (std::pair<const TrackId, Landmark>& entry : model.landmarks) { entry.second.position = transform.scale*transform.rotation*entry.second.position+transform.translation; }
}
} // namespace stereoforge::optimization
