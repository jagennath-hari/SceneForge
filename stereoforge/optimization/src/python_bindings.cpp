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

namespace py = pybind11;
namespace so = stereoforge::optimization;
namespace {
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
    module.attr("api_version") = 7;
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
        .def("set_tracks",&so::MapBuilder::SetTracks,py::call_guard<py::gil_scoped_release>())
        .def("add_window",&so::MapBuilder::AddWindow,py::call_guard<py::gil_scoped_release>())
        .def("rank_windows",&so::MapBuilder::RankWindows,py::call_guard<py::gil_scoped_release>())
        .def("finalize",&so::MapBuilder::Finalize,py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("calibration_stage",&so::MapBuilder::CalibrationStage)
        .def_property_readonly("accepted_windows",&so::MapBuilder::AcceptedWindows)
        .def_property_readonly("map",&so::MapBuilder::Map,py::return_value_policy::reference_internal);
}
