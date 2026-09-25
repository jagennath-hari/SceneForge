#include "stereoforge/optimization/factors.cuh"

#include <algorithm>
#include <cmath>
#include <cunls/common/helper.h>

namespace stereoforge::optimization {
// cuNLS error macros expand an unqualified LogError at the call site.
using cunls::LogError;

namespace {
constexpr unsigned int kThreads = 128; // Four full warps; per-observation registers.
constexpr float kNear = 1e-4f; // Scene normalized to median nonzero camera step = 1.
constexpr float kBarrier = 1000.0f;

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>(std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
}

__global__ void ProjectKernel(const float* pixels, const float* principal,
    const float* weights, float const* const* states, float* residuals,
    float* jacobians, float* errors, std::size_t count) {
    // No inter-thread dependency or shared reusable tile: barriers/shared memory
    // would add overhead here. Grid-stride loop also bounds launch dimensions.
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float* pose = states[3*i];
        const float* point = states[3*i+1];
        const float* focal = states[3*i+2];
        const float fx = expf(focal[0]);
        const float fy = expf(focal[1]);
        const float x = pose[0]*point[0] + pose[1]*point[1] + pose[2]*point[2] + pose[3];
        const float y = pose[4]*point[0] + pose[5]*point[1] + pose[6]*point[2] + pose[7];
        const float z = pose[8]*point[0] + pose[9]*point[1] + pose[10]*point[2] + pose[11];
        const float inverse = 1.0f / fmaxf(z, kNear);
        const float u = fx*x*inverse;
        const float v = fy*y*inverse;
        const float du = u + principal[2*i] - pixels[2*i];
        const float dv = v + principal[2*i+1] - pixels[2*i+1];
        if (errors != nullptr) {
            errors[i] = z > kNear && isfinite(du) && isfinite(dv) ? du*du + dv*dv : INFINITY;
        }
        const float weight = weights == nullptr ? 1.0f : weights[i];
        if (residuals != nullptr) {
            residuals[3*i] = weight*du;
            residuals[3*i+1] = weight*dv;
            residuals[3*i+2] = kBarrier*fminf(z-kNear, 0.0f);
        }
        if (jacobians == nullptr) { continue; }
        float* jac = jacobians + 33*i; // 3 rows x (6 pose + 3 point + 2 focal).
        float projection[3][3] = {
            {weight*fx*inverse, 0, z > kNear ? -weight*u*inverse : 0},
            {0, weight*fy*inverse, z > kNear ? -weight*v*inverse : 0},
            {0, 0, z < kNear ? kBarrier : 0}};
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            float world[3];
#pragma unroll
            for (int column = 0; column < 3; ++column) {
                world[column] = projection[row][0]*pose[column] +
                    projection[row][1]*pose[4+column] + projection[row][2]*pose[8+column];
                jac[row*11+3+column] = world[column];
                jac[row*11+6+column] = world[column];
            }
            // Right SE3 update: d(R*P+t)/d(delta) = [-R*[P]x, R].
            jac[row*11] = world[2]*point[1] - world[1]*point[2];
            jac[row*11+1] = world[0]*point[2] - world[2]*point[0];
            jac[row*11+2] = world[1]*point[0] - world[0]*point[1];
            jac[row*11+9] = row == 0 ? weight*u : 0;
            jac[row*11+10] = row == 1 ? weight*v : 0;
        }
    }
}

__global__ void FocalPriorKernel(const float* targets, float const* const* states,
    float* residuals, float* jacobians, std::size_t count, float sigma) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float fx = expf(states[i][0]);
        const float fy = expf(states[i][1]);
        if (residuals != nullptr) {
            residuals[2*i] = (fx-targets[2*i])/sigma;
            residuals[2*i+1] = (fy-targets[2*i+1])/sigma;
        }
        if (jacobians != nullptr) {
            jacobians[4*i] = fx/sigma; jacobians[4*i+1] = 0;
            jacobians[4*i+2] = 0; jacobians[4*i+3] = fy/sigma;
        }
    }
}

__global__ void ScalePosePriorKernel(float* residuals, float* jacobians,
    std::size_t count, float rotation_sigma, float translation_sigma) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count*6; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float sigma = i%6 < 3 ? rotation_sigma : translation_sigma;
        if (residuals != nullptr) { residuals[i] /= sigma; }
        if (jacobians != nullptr) {
#pragma unroll
            for (int j = 0; j < 6; ++j) { jacobians[6*i+j] /= sigma; }
        }
    }
}

__global__ void TlsKernel(const float* errors, float* weights, std::size_t count,
                          float threshold_squared, float mu) {
    const float lower = mu/(mu+1)*threshold_squared;
    const float upper = (mu+1)/mu*threshold_squared;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float error = errors[i];
        float weight = 0;
        if (isfinite(error) && error <= lower) { weight = 1; }
        else if (isfinite(error) && error < upper) {
            weight = sqrtf(threshold_squared*mu*(mu+1)/error)-mu;
        }
        // cuNLS receives sqrt(w)*r and sqrt(w)*J, not w*r.
        weights[i] = sqrtf(fminf(1.0f, fmaxf(0.0f, weight)));
    }
}
} // namespace

PixelReprojectionFactors::PixelReprojectionFactors(const float* pixels, const float* principal,
    const float* weights, std::size_t count)
    : pixels_(pixels), principal_points_(principal), sqrt_weights_(weights), count_(count) {}
std::size_t PixelReprojectionFactors::NumFactors() const { return this->count_; }
bool PixelReprojectionFactors::Evaluate(float* r, float* j, float const* const* states, cudaStream_t stream) const {
    if (this->count_ == 0) { return true; }
    ProjectKernel<<<Blocks(this->count_), kThreads, 0, stream>>>(this->pixels_, this->principal_points_,
        this->sqrt_weights_, states, r, j, nullptr, this->count_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return true;
}
FocalPriorFactors::FocalPriorFactors(const float* targets, std::size_t count, float sigma)
    : focal_pixels_(targets), count_(count), sigma_pixels_(sigma) {}
std::size_t FocalPriorFactors::NumFactors() const { return this->count_; }
bool FocalPriorFactors::Evaluate(float* r, float* j, float const* const* states, cudaStream_t stream) const {
    if (this->count_ == 0) { return true; }
    FocalPriorKernel<<<Blocks(this->count_), kThreads, 0, stream>>>(this->focal_pixels_, states,
        r, j, this->count_, this->sigma_pixels_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return true;
}
PosePriorFactors::PosePriorFactors(const cunls::SE3Transform* poses, std::size_t count, float rs, float ts)
    : factors_(poses, count), count_(count), rotation_sigma_(rs), translation_sigma_(ts) {}
std::size_t PosePriorFactors::NumFactors() const { return this->count_; }
bool PosePriorFactors::Evaluate(float* r, float* j, float const* const* states, cudaStream_t stream) const {
    if (this->count_ == 0) { return true; }
    if (!this->factors_.Evaluate(r, j, states, stream)) { return false; }
    ScalePosePriorKernel<<<Blocks(this->count_*6), kThreads, 0, stream>>>(r, j, this->count_,
        this->rotation_sigma_, this->translation_sigma_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return true;
}
void EvaluatePixelErrors(const float* pixels, const float* principal, float const* const* states,
    float* errors, std::size_t count, cudaStream_t stream) {
    if (count == 0) { return; }
    ProjectKernel<<<Blocks(count), kThreads, 0, stream>>>(pixels, principal, nullptr, states,
        nullptr, nullptr, errors, count);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
}
void UpdateTlsWeights(const float* errors, float* weights, std::size_t count, float c2,
    float mu, cudaStream_t stream) {
    if (count == 0) { return; }
    TlsKernel<<<Blocks(count), kThreads, 0, stream>>>(errors, weights, count, c2, mu);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
}
} // namespace stereoforge::optimization
