#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cunls/factor/sized_factor_batch.h>
#include <cunls/factor/prior/se3_prior_factor_batch.h>

namespace sceneforge::optimization {

// Four state blocks: world-to-camera SE3 (right perturbation), world point,
// [log(fx), log(fy)] and [cx, cy] in processed-image pixels.
// Residuals: two weighted pixel errors and an unweighted cheirality barrier.
class PixelReprojectionFactors final : public cunls::SizedFactorBatch<3, 6, 3, 2, 2> {
public:
    PixelReprojectionFactors(const float* pixels,
                             const float* sqrt_weights, std::size_t count, float huber_delta = 0.0f);
    [[nodiscard]] bool Evaluate(float* residuals, float* jacobians,
                               float const* const* states, cudaStream_t stream) const override;
    [[nodiscard]] std::size_t NumFactors() const override;
private:
    const float* pixels_;
    const float* sqrt_weights_;
    float huber_delta_;
    std::size_t count_;
};

class CalibrationPriorFactors final : public cunls::SizedFactorBatch<2, 2> {
public:
    CalibrationPriorFactors(const float* focal_pixels, std::size_t count, float sigma_pixels, bool logarithmic = true);
    [[nodiscard]] bool Evaluate(float* residuals, float* jacobians,
                               float const* const* states, cudaStream_t stream) const override;
    [[nodiscard]] std::size_t NumFactors() const override;
private:
    const float* focal_pixels_;
    std::size_t count_;
    float sigma_pixels_;
    bool logarithmic_;
};

class PosePriorFactors final : public cunls::SizedFactorBatch<6, 6> {
public:
    PosePriorFactors(const cunls::SE3Transform* poses, std::size_t count,
                     float rotation_sigma, float translation_sigma);
    [[nodiscard]] bool Evaluate(float* residuals, float* jacobians,
                               float const* const* states, cudaStream_t stream) const override;
    [[nodiscard]] std::size_t NumFactors() const override;
private:
    cunls::SE3PriorFactorBatch factors_;
    std::size_t count_;
    float rotation_sigma_;
    float translation_sigma_;
};

// Unweighted errors are separate from the LM residual buffer. GNC weights stay
// fixed during each LM solve, and update only between solves on the same stream.
void EvaluatePixelErrors(const float* pixels,
                         float const* const* states, float* squared_errors,
                         std::size_t count, cudaStream_t stream);
void UpdateTlsWeights(const float* squared_errors, float* sqrt_weights,
                      std::size_t count, float threshold_squared, float mu,
                      cudaStream_t stream);
} // namespace sceneforge::optimization
