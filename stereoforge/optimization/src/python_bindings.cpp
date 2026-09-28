#include "stereoforge/optimization/map_builder.hpp"
#include "stereoforge/optimization/dense_fusion.hpp"
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <Eigen/LU>
#include <stdexcept>
#include <cstring>
#include <cunls/common/log.h>

namespace py = pybind11;
namespace so = stereoforge::optimization;
namespace {
template <typename T>
py::array_t<T> VectorArray(const std::vector<T>& values) {
    py::array_t<T> result(values.size());
    if (!values.empty()) { std::memcpy(result.mutable_data(), values.data(), values.size()*sizeof(T)); }
    return result;
}
void StateArrays(py::dict& result, const std::vector<so::BACamera>& cameras,
                 const std::vector<std::array<float, 3>>& points) {
    const py::ssize_t n = cameras.size(), m = points.size();
    py::array_t<float> poses(std::vector<py::ssize_t>{n,4,4}), intrinsics(std::vector<py::ssize_t>{n,4});
    py::array_t<float> xyz(std::vector<py::ssize_t>{m,3});
    for (py::ssize_t i = 0; i < n; ++i) {
        std::copy(cameras[i].world_to_camera.begin(), cameras[i].world_to_camera.end(), poses.mutable_data()+16*i);
        std::copy(cameras[i].intrinsics.begin(), cameras[i].intrinsics.end(), intrinsics.mutable_data()+4*i);
    }
    for (py::ssize_t i = 0; i < m; ++i) { std::copy(points[i].begin(), points[i].end(), xyz.mutable_data()+3*i); }
    result["world_to_camera"] = poses; result["intrinsics"] = intrinsics; result["points"] = xyz;
}
py::dict Objective(const so::BAObjective& value) {
    py::dict d;
    d["robust_pixels"] = value.robust_pixels; d["cheirality"] = value.cheirality;
    d["pose_prior_unweighted"] = value.pose_prior_unweighted; d["pose_prior_weighted"] = value.pose_prior_weighted;
    d["focal_prior"] = value.focal_prior; d["principal_prior"] = value.principal_prior;
    d["scale_anchor"] = value.scale_anchor;
    return d;
}
py::dict CheckpointDocument(const std::string& name, const so::BAInput& input, const so::BAResult& output) {
    py::dict d, meta, options;
    meta["format_version"] = 1; meta["stage"] = name;
    meta["normalization_origin"] = input.origin; meta["normalization_scale"] = input.normalization_scale;
    meta["cost_convention"] = "unhalved squared residual sums";
    meta["fixed_camera_index"] = 0;
    meta["fixed_camera_id"] = input.camera_ids.front();
    meta["scale_camera_id"] = input.camera_ids.at(input.options.scale_camera);
    const so::BAOptions& o = input.options;
    options["pose_prior_weight"] = o.pose_prior_weight; options["scale_camera"] = o.scale_camera;
    options["scale_target"] = o.scale_target; options["scale_log_sigma"] = o.scale_log_sigma;
    options["huber_delta_pixels"] = o.huber_delta_pixels; options["rotation_sigma_radians"] = o.rotation_sigma_radians;
    options["translation_sigma"] = o.translation_sigma; options["focal_sigma_pixels"] = o.focal_sigma_pixels;
    options["principal_sigma_pixels"] = o.principal_sigma_pixels; options["lm_iterations"] = o.lm_iterations;
    options["shared_intrinsics"] = o.shared_intrinsics; options["optimize_principal"] = o.optimize_principal;
    options["use_gnc"] = o.use_gnc; options["check_jacobians"] = o.check_jacobians;
    options["threshold_pixels"] = o.threshold_pixels;
    meta["options"] = options;
    if (name == "input") {
        StateArrays(d, input.cameras, input.points);
        py::dict prior; StateArrays(prior, input.prior_cameras, {});
        d["prior_world_to_camera"] = prior["world_to_camera"]; d["prior_intrinsics"] = prior["intrinsics"];
        d["camera_ids"] = VectorArray(input.camera_ids); d["track_ids"] = VectorArray(input.track_ids);
        const py::ssize_t n = input.observations.size();
        py::array_t<std::int64_t> cameras(n), points(n);
        py::array_t<float> pixels(std::vector<py::ssize_t>{n,2});
        for (py::ssize_t i = 0; i < n; ++i) {
            cameras.mutable_data()[i] = input.observations[i].camera;
            points.mutable_data()[i] = input.observations[i].point;
            pixels.mutable_data()[2*i] = input.observations[i].pixel[0];
            pixels.mutable_data()[2*i+1] = input.observations[i].pixel[1];
        }
        d["observation_camera"] = cameras; d["observation_point"] = points; d["observation_pixel"] = pixels;
    } else {
        StateArrays(d, output.cameras, output.points);
        d["squared_errors_before"] = VectorArray(output.squared_errors_before);
        d["squared_errors_after"] = VectorArray(output.squared_errors_after);
        const so::BAReport& r = output.report;
        meta["objective_before"] = Objective(r.objective_before); meta["objective_after"] = Objective(r.objective_after);
        meta["scale_distance_before"] = r.scale_distance_before; meta["scale_distance_after"] = r.scale_distance_after;
        meta["termination_reason"] = "not exposed by cuNLS summary; inspect solver log";
        meta["objective_decreased"] = !r.rounds.empty() && r.rounds.back().weighted_cost_after < r.rounds.front().weighted_cost_before;
        meta["optimization_complete"] = r.optimization_complete; meta["lm_budget_exhausted"] = r.lm_budget_exhausted;
        meta["jacobian_maximum_tolerance_ratio"] = r.maximum_jacobian_tolerance_ratio;
        py::list rounds;
        for (const so::BARound& round : r.rounds) {
            py::dict item; item["iterations"] = round.lm_iterations;
            item["iteration_costs"] = round.iteration_costs;
            item["solver_cost_before"] = round.weighted_cost_before; item["solver_cost_after"] = round.weighted_cost_after;
            rounds.append(item);
        }
        meta["rounds"] = rounds;
    }
    d["metadata"] = meta;
    return d;
}
// Replay uses the same native solver and binary checkpoint arrays as production.
// Called in a dedicated diagnostic process because cuNLS logging is process-global.
template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> ReplayArray(
    const py::dict& data, const char* key, const std::vector<py::ssize_t>& shape) {
    py::array_t<T, py::array::c_style | py::array::forcecast> value =
        py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(data[key]);
    if (!value || value.ndim() != static_cast<py::ssize_t>(shape.size())) {
        throw std::invalid_argument(std::string("Invalid checkpoint array: ")+key);
    }
    for (py::ssize_t i = 0; i < value.ndim(); ++i) {
        if (shape[i] >= 0 && value.shape(i) != shape[i]) {
            throw std::invalid_argument(std::string("Invalid checkpoint shape: ")+key);
        }
    }
    return value;
}
py::dict ReplayBA(const py::dict& data, const py::dict& metadata,
                  int device, const std::string& log_path) {
    if (py::cast<int>(metadata["format_version"]) != 1) {
        throw std::invalid_argument("Unsupported BA checkpoint version");
    }
    so::BAInput input;
    input.origin = py::cast<std::array<double,3>>(metadata["normalization_origin"]);
    input.normalization_scale = py::cast<double>(metadata["normalization_scale"]);
    const py::dict options = py::cast<py::dict>(metadata["options"]);
    input.options.pose_prior_weight = py::cast<float>(options["pose_prior_weight"]);
    input.options.scale_camera = py::cast<int>(options["scale_camera"]);
    input.options.scale_target = py::cast<float>(options["scale_target"]);
    input.options.scale_log_sigma = py::cast<float>(options["scale_log_sigma"]);
    input.options.huber_delta_pixels = py::cast<float>(options["huber_delta_pixels"]);
    input.options.rotation_sigma_radians = py::cast<float>(options["rotation_sigma_radians"]);
    input.options.translation_sigma = py::cast<float>(options["translation_sigma"]);
    input.options.focal_sigma_pixels = py::cast<float>(options["focal_sigma_pixels"]);
    input.options.principal_sigma_pixels = py::cast<float>(options["principal_sigma_pixels"]);
    input.options.lm_iterations = py::cast<int>(options["lm_iterations"]);
    input.options.shared_intrinsics = py::cast<bool>(options["shared_intrinsics"]);
    input.options.optimize_principal = py::cast<bool>(options["optimize_principal"]);
    input.options.use_gnc = py::cast<bool>(options["use_gnc"]);
    input.options.check_jacobians = py::cast<bool>(options["check_jacobians"]);
    input.options.threshold_pixels = py::cast<float>(options["threshold_pixels"]);
    const py::array_t<float> poses = ReplayArray<float>(data,"world_to_camera",{-1,4,4});
    const py::ssize_t n = poses.shape(0);
    if (n < 2) { throw std::invalid_argument("Replay requires at least two cameras"); }
    const py::array_t<float> k = ReplayArray<float>(data,"intrinsics",{n,4});
    const py::array_t<float> prior = ReplayArray<float>(data,"prior_world_to_camera",{n,4,4});
    const py::array_t<float> prior_k = ReplayArray<float>(data,"prior_intrinsics",{n,4});
    const py::array_t<std::int64_t> ids = ReplayArray<std::int64_t>(data,"camera_ids",{n});
    input.camera_ids.assign(ids.data(),ids.data()+n);
    for (py::ssize_t i = 0; i < n; ++i) {
        so::BACamera camera, target;
        std::copy_n(poses.data()+16*i,16,camera.world_to_camera.begin());
        std::copy_n(k.data()+4*i,4,camera.intrinsics.begin());
        std::copy_n(prior.data()+16*i,16,target.world_to_camera.begin());
        std::copy_n(prior_k.data()+4*i,4,target.intrinsics.begin());
        input.cameras.push_back(camera); input.prior_cameras.push_back(target);
    }
    const py::array_t<float> points = ReplayArray<float>(data,"points",{-1,3});
    const py::ssize_t m = points.shape(0);
    const py::array_t<std::int64_t> tracks = ReplayArray<std::int64_t>(data,"track_ids",{m});
    input.track_ids.assign(tracks.data(),tracks.data()+m);
    for (py::ssize_t i = 0; i < m; ++i) {
        input.points.push_back({points.data()[3*i],points.data()[3*i+1],points.data()[3*i+2]});
    }
    const py::array_t<float> pixels = ReplayArray<float>(data,"observation_pixel",{-1,2});
    const py::ssize_t count = pixels.shape(0);
    const py::array_t<std::int64_t> cameras = ReplayArray<std::int64_t>(data,"observation_camera",{count});
    const py::array_t<std::int64_t> landmarks = ReplayArray<std::int64_t>(data,"observation_point",{count});
    for (py::ssize_t i = 0; i < count; ++i) {
        const std::int64_t camera = cameras.data()[i], point = landmarks.data()[i];
        if (camera < 0 || camera >= n || point < 0 || point >= m) {
            throw std::invalid_argument("Checkpoint observation index outside state arrays");
        }
        input.observations.push_back({static_cast<std::size_t>(camera),static_cast<std::size_t>(point),
                                     {pixels.data()[2*i],pixels.data()[2*i+1]}});
    }
    so::BAResult result;
    {
        py::gil_scoped_release release;
        cunls::SetLoggerOptions(cunls::Verbosity::Message,cunls::Sink::File,log_path);
        result = so::BundleAdjuster(device).Solve(input);
    }
    return CheckpointDocument("replay",input,result);
}
so::DepthFrame ReadFrame(so::FrameId id, const Eigen::Matrix4d& pose, const Eigen::Matrix3d& k,
                        const py::array_t<float, py::array::c_style | py::array::forcecast>& depth,
                        const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& rgb) {
    if (depth.ndim() != 2 || rgb.ndim() != 3 || rgb.shape(0) != depth.shape(0) || rgb.shape(1) != depth.shape(1) || rgb.shape(2) != 3) {
        throw std::invalid_argument("Expected depth HxW and RGB HxWx3 on the same grid");
    }
    if (!pose.allFinite() || (pose.row(3)-Eigen::RowVector4d(0,0,0,1)).norm() > 1e-6 ||
        (pose.topLeftCorner<3,3>().transpose()*pose.topLeftCorner<3,3>()-Eigen::Matrix3d::Identity()).norm() > 1e-3 ||
        std::abs(pose.topLeftCorner<3,3>().determinant()-1) > 1e-3) { throw std::invalid_argument("Invalid camera-to-world pose"); }
    so::DepthFrame frame;
    frame.id = id; frame.camera.rotation = pose.topLeftCorner<3,3>(); frame.camera.center = pose.topRightCorner<3,1>();
    frame.camera.intrinsics = k; frame.camera.height = static_cast<int>(depth.shape(0)); frame.camera.width = static_cast<int>(depth.shape(1));
    frame.depth.assign(depth.data(), depth.data()+depth.size()); frame.rgb.assign(rgb.data(), rgb.data()+rgb.size());
    return frame;
}
}
PYBIND11_MODULE(_stereoforge_map, module) {
    module.attr("api_version") = 19;
    module.def("replay_ba", &ReplayBA, py::arg("arrays"), py::arg("metadata"), py::arg("device"), py::arg("log_path"));
    py::class_<so::DenseFusion>(module,"DenseFusion")
        .def(py::init<double>())
        .def("add",[](so::DenseFusion& fusion,
            const py::array_t<float,py::array::c_style|py::array::forcecast>& xyz,
            const py::array_t<std::uint8_t,py::array::c_style|py::array::forcecast>& rgb, std::int64_t frame) {
            if (xyz.ndim()!=2 || xyz.shape(1)!=3 || rgb.ndim()!=2 || rgb.shape(1)!=3 || rgb.shape(0)!=xyz.shape(0)) {
                throw std::invalid_argument("Dense fusion expects Nx3 points and colors");
            }
            const float* points=xyz.data(); const std::uint8_t* colors=rgb.data();
            const std::size_t count=xyz.shape(0);
            py::gil_scoped_release release;
            fusion.Add(points,colors,count,frame);
        })
        .def("write",[](const so::DenseFusion& fusion,const std::string& path) {
            so::DensePreview preview;
            { py::gil_scoped_release release; preview=fusion.Write(path,240000); }
            const py::ssize_t n=preview.points.size();
            py::array_t<float> xyz(std::vector<py::ssize_t>{n,3});
            py::array_t<std::uint8_t> rgb(std::vector<py::ssize_t>{n,3});
            py::array_t<std::int64_t> frames(n);
            for (py::ssize_t i=0;i<n;++i) {
                for (int j=0;j<3;++j) { xyz.mutable_data()[3*i+j]=preview.points[i][j]; rgb.mutable_data()[3*i+j]=preview.colors[i][j]; }
                frames.mutable_data()[i]=preview.frames[i];
            }
            py::dict result; result["xyz"]=xyz; result["rgb"]=rgb; result["frames"]=frames; result["count"]=preview.total;
            return result;
        });
    py::class_<so::Camera>(module,"Camera")
        .def_readonly("rotation",&so::Camera::rotation).def_readonly("center",&so::Camera::center)
        .def_readonly("intrinsics",&so::Camera::intrinsics).def_readonly("height",&so::Camera::height).def_readonly("width",&so::Camera::width);
    py::class_<so::Landmark>(module,"Landmark")
        .def_readonly("position",&so::Landmark::position).def_readonly("color",&so::Landmark::color).def_readonly("observations",&so::Landmark::observations);
    py::class_<so::SparseMap>(module,"SparseMap")
        .def_readonly("cameras",&so::SparseMap::cameras).def_readonly("landmarks",&so::SparseMap::landmarks);
    py::class_<so::DepthFrame>(module,"DepthFrame").def(py::init(&ReadFrame));
    py::class_<so::MapBuilder>(module,"MapBuilder")
        .def(py::init<int,int,bool>())
        .def("rerun_dense",[](so::MapBuilder& builder,
            const py::array_t<float,py::array::c_style|py::array::forcecast>& xyz,
            const py::array_t<std::uint8_t,py::array::c_style|py::array::forcecast>& rgb) {
            if (xyz.ndim()!=2 || rgb.ndim()!=2 || xyz.shape(1)!=3 || rgb.shape(1)!=3 ||
                xyz.shape(0)!=rgb.shape(0) || xyz.shape(0)>240000) {
                throw std::invalid_argument("Expected matching Nx3 dense XYZ/RGB arrays, at most 240000 points");
            }
            py::gil_scoped_release release;
            builder.RerunDense(xyz.data(),rgb.data(),static_cast<std::size_t>(xyz.shape(0)));
        })
        .def("rerun_event",&so::MapBuilder::RerunEvent,py::call_guard<py::gil_scoped_release>())
        .def("rerun_keyframe",[](so::MapBuilder& builder, std::int64_t id,
            const py::array_t<std::uint8_t,py::array::c_style|py::array::forcecast>& rgb) {
            if (id < 0 || rgb.ndim()!=3 || rgb.shape(2)!=3 || rgb.shape(0)<=0 || rgb.shape(1)<=0 ||
                rgb.shape(0)>256 || rgb.shape(1)>256) { throw std::invalid_argument("Expected a keyframe thumbnail no larger than 256x256 RGB"); }
            so::DepthFrame frame;
            frame.id=id; frame.camera.width=static_cast<int>(rgb.shape(1)); frame.camera.height=static_cast<int>(rgb.shape(0));
            frame.rgb.assign(rgb.data(),rgb.data()+rgb.size());
            py::gil_scoped_release release;
            builder.RerunKeyframe(frame);
        })
        .def("preview_window",&so::MapBuilder::PreviewWindow,py::call_guard<py::gil_scoped_release>())
        .def("enable_rerun",&so::MapBuilder::EnableRerun,py::call_guard<py::gil_scoped_release>())
        .def("set_tracks",&so::MapBuilder::SetTracks,py::call_guard<py::gil_scoped_release>())
        .def("add_window",&so::MapBuilder::AddWindow,py::call_guard<py::gil_scoped_release>())
        .def("rank_windows",&so::MapBuilder::RankWindows,py::call_guard<py::gil_scoped_release>())
        .def("set_ba_checkpoint", [](so::MapBuilder& builder, py::object callback) {
            if (callback.is_none()) { builder.SetBACheckpoint({}); return; }
            const py::function function = callback.cast<py::function>();
            builder.SetBACheckpoint([function](const std::string& name, const so::BAInput& input, const so::BAResult& output) {
                py::gil_scoped_acquire acquire;
                function(name, CheckpointDocument(name, input, output));
            });
        })
        .def("finalize",&so::MapBuilder::Finalize,py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("shared_calibration_complete",&so::MapBuilder::SharedCalibrationComplete)
        .def_property_readonly("supported_initialization_frames",&so::MapBuilder::SupportedInitializationFrames)
        .def_property_readonly("unanchored_frames",&so::MapBuilder::UnanchoredFrames)
        .def_property_readonly("accepted_windows",&so::MapBuilder::AcceptedWindows)
        .def_property_readonly("map",&so::MapBuilder::Map,py::return_value_policy::reference_internal);
}
