#include "sceneforge/video/cuda_ransac.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cmath>
#include <stdexcept>
#include <string>

namespace sceneforge::video {
namespace {
constexpr int threads = 128;
constexpr unsigned warp_mask = 0xffffffffu;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("CUDA RANSAC: ") + cudaGetErrorString(error));
}
template <typename T> __device__ float value(T input) { return static_cast<float>(input); }
template <> __device__ float value(__half input) { return __half2float(input); }

// Ordered block scan compacts valid matches without atomics or a host count.
// Every thread participates, including padded lanes in the final tile.
template <typename T>
__global__ void compact(const T* a, const T* sa, const T* b, const T* sb,
    const std::int32_t* indices, const T* confidence, int count,
    int w0, int h0, int w1, int h1, float detector, float threshold,
    DeviceMatch* output, int* output_count, RansacSummary* summary) {
    using Scan = cub::BlockScan<int, threads>;
    __shared__ Scan::TempStorage storage;
    __shared__ int total;
    if (threadIdx.x == 0) { total = 0; *summary = {}; }
    __syncthreads();
    for (int base = 0; base < count; base += threads) {
        const int i = base + threadIdx.x;
        DeviceMatch match{};
        int usable = 0;
        if (i < count) {
            const int j = indices[i];
            const float score = value(confidence[i]);
            if (!isfinite(score) || j < -1 || j >= count) atomicExch(&summary->invalid, 1);
            if (j >= 0 && j < count && isfinite(score) && score >= threshold) {
                match = {value(a[2*i]), value(a[2*i+1]), value(b[2*j]), value(b[2*j+1]), i, j, score};
                const float score0 = value(sa[i]), score1 = value(sb[j]);
                usable = isfinite(match.ax) && isfinite(match.ay) && isfinite(match.bx) && isfinite(match.by) &&
                    isfinite(score0) && isfinite(score1) && score0 >= detector && score1 >= detector &&
                    match.ax >= 4 && match.ay >= 4 && match.ax < w0-4 && match.ay < h0-4 &&
                    match.bx >= 4 && match.by >= 4 && match.bx < w1-4 && match.by < h1-4;
            }
        }
        int offset = 0, tile_count = 0;
        Scan(storage).ExclusiveSum(usable, offset, tile_count);
        if (usable) output[total + offset] = match;
        __syncthreads();
        if (threadIdx.x == 0) total += tile_count;
        __syncthreads();
    }
    if (threadIdx.x == 0) { *output_count = total; summary->matches = total; }
}

// Cyclic Jacobi for the small symmetric DLT normal matrix. Double precision
// and Hartley normalization avoid solving raw pixel-coordinate normal equations.
// The fit runs on one lane; the expensive consensus scoring is block parallel.
template <int N>
__device__ bool eigenvectors(double* matrix, double* vectors) {
    for (int i = 0; i < N*N; ++i) vectors[i] = (i/N == i%N) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 40; ++sweep) {
        double off = 0, diagonal = 0;
        for (int p = 0; p < N; ++p) {
            diagonal += fabs(matrix[p*N+p]);
            for (int q = p+1; q < N; ++q) off += fabs(matrix[p*N+q]);
        }
        if (!isfinite(off) || !isfinite(diagonal)) return false;
        if (off <= 1e-12 * fmax(diagonal, 1e-20)) return true;
        for (int p = 0; p < N-1; ++p) for (int q = p+1; q < N; ++q) {
            const double apq = matrix[p*N+q];
            if (fabs(apq) <= 1e-15 * fmax(diagonal, 1e-20)) continue;
            const double tau = (matrix[q*N+q] - matrix[p*N+p]) / (2*apq);
            const double t = copysign(1.0, tau) / (fabs(tau) + hypot(1.0, tau));
            const double c = 1 / sqrt(1 + t*t), s = t*c;
            matrix[p*N+p] -= t*apq;
            matrix[q*N+q] += t*apq;
            matrix[p*N+q] = matrix[q*N+p] = 0;
            for (int k = 0; k < N; ++k) {
                if (k != p && k != q) {
                    const double kp = matrix[k*N+p], kq = matrix[k*N+q];
                    matrix[k*N+p] = matrix[p*N+k] = c*kp-s*kq;
                    matrix[k*N+q] = matrix[q*N+k] = s*kp+c*kq;
                }
                const double vp = vectors[k*N+p], vq = vectors[k*N+q];
                vectors[k*N+p] = c*vp-s*vq;
                vectors[k*N+q] = s*vp+c*vq;
            }
        }
    }
    return false;
}
__device__ float errorSquared(const RansacModel& model, const DeviceMatch& p) {
    if (!model.valid) return CUDART_INF_F;
    const double* m = model.matrix;
    const double x = m[0]*p.ax + m[1]*p.ay + m[2];
    const double y = m[3]*p.ax + m[4]*p.ay + m[5];
    const double z = m[6]*p.ax + m[7]*p.ay + m[8];
    if (model.homography) {
        if (fabs(z) < 1e-12) return CUDART_INF_F;
        const double dx = x/z-p.bx, dy = y/z-p.by;
        return static_cast<float>(dx*dx+dy*dy);
    }
    const double tx = m[0]*p.bx + m[3]*p.by + m[6];
    const double ty = m[1]*p.bx + m[4]*p.by + m[7];
    const double denominator = fmin(x*x+y*y, tx*tx+ty*ty);
    if (denominator < 1e-24) return CUDART_INF_F;
    const double residual = p.bx*x+p.by*y+z;
    // Maximum of the two point-to-epipolar-line distances, in pixel units.
    return static_cast<float>(residual*residual/denominator);
}
__device__ unsigned randomNext(unsigned& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
__device__ void row(const DeviceMatch& p, const double* norm, bool homography, int equation, double* a) {
    const double x = (p.ax-norm[0])*norm[2], y = (p.ay-norm[1])*norm[2];
    const double u = (p.bx-norm[3])*norm[5], v = (p.by-norm[4])*norm[5];
    if (!homography) {
        a[0]=u*x; a[1]=u*y; a[2]=u; a[3]=v*x; a[4]=v*y; a[5]=v; a[6]=x; a[7]=y; a[8]=1;
    } else if (equation == 0) {
        a[0]=x; a[1]=y; a[2]=1; a[3]=0; a[4]=0; a[5]=0; a[6]=-u*x; a[7]=-u*y; a[8]=-u;
    } else {
        a[0]=0; a[1]=0; a[2]=0; a[3]=x; a[4]=y; a[5]=1; a[6]=-v*x; a[7]=-v*y; a[8]=-v;
    }
}
__device__ bool solve(double* normal, const double* norm, RansacModel& model) {
    double vectors[81];
    if (!eigenvectors<9>(normal, vectors)) return false;
    int minimum = 0;
    double largest = 0;
    for (int i = 0; i < 9; ++i) {
        if (normal[i*9+i] < normal[minimum*9+minimum]) minimum = i;
        largest = fmax(largest, normal[i*9+i]);
    }
    double second = CUDART_INF;
    for (int i = 0; i < 9; ++i) if (i != minimum) second = fmin(second, normal[i*9+i]);
    if (largest <= 0 || second <= largest*1e-9) return false; // degenerate / collinear sample
    double m[9];
    for (int i = 0; i < 9; ++i) m[i] = vectors[i*9+minimum];
    if (!model.homography) {
        double gram[9]{}, basis[9];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) gram[i*3+j] += m[k*3+i]*m[k*3+j];
        if (!eigenvectors<3>(gram, basis)) return false;
        int smallest = 0;
        for (int i = 1; i < 3; ++i) if (gram[i*3+i] < gram[smallest*3+smallest]) smallest = i;
        double second_singular = CUDART_INF, largest_singular = 0;
        for (int i = 0; i < 3; ++i) {
            largest_singular = fmax(largest_singular, gram[i*3+i]);
            if (i != smallest) second_singular = fmin(second_singular, gram[i*3+i]);
        }
        if (second_singular <= largest_singular*1e-12) return false;
        // Project away the smallest singular component to enforce rank two.
        for (int i = 0; i < 3; ++i) {
            double projection = 0;
            for (int j = 0; j < 3; ++j) projection += m[i*3+j]*basis[j*3+smallest];
            for (int j = 0; j < 3; ++j) m[i*3+j] -= projection*basis[j*3+smallest];
        }
    } else {
        const double determinant = m[0]*(m[4]*m[8]-m[5]*m[7]) -
            m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
        if (fabs(determinant) < 1e-12) return false;
    }
    const double ta[9]{norm[2],0,-norm[0]*norm[2],0,norm[2],-norm[1]*norm[2],0,0,1};
    const double left_f[9]{norm[5],0,0,0,norm[5],0,-norm[3]*norm[5],-norm[4]*norm[5],1};
    const double left_h[9]{1/norm[5],0,norm[3],0,1/norm[5],norm[4],0,0,1};
    const double* left = model.homography ? left_h : left_f;
    double temp[9]{};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) temp[i*3+j] += m[i*3+k]*ta[k*3+j];
    double length = 0;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        double entry = 0;
        for (int k = 0; k < 3; ++k) entry += left[i*3+k]*temp[k*3+j];
        model.matrix[i*3+j] = entry;
        length += entry*entry;
    }
    if (!isfinite(length) || length < 1e-30) return false;
    for (int i = 0; i < 9; ++i) model.matrix[i] /= sqrt(length);
    return true;
}

// One block per hypothesis. Normal-matrix entries are formed cooperatively;
// the fitted model is shared by all lanes for coalesced consensus evaluation.
// A second launch uses all inliers of each winning model for a DLT refit.
template <bool Refit>
__global__ void hypotheses(const DeviceMatch* matches, const int* count, RansacModel* models,
    const int* winners, float threshold2, unsigned seed) {
    using IntReduce = cub::BlockReduce<int, threads>;
    using FloatReduce = cub::BlockReduce<float, threads>;
    __shared__ IntReduce::TempStorage integers;
    __shared__ FloatReduce::TempStorage floats;
    __shared__ int chosen[Refit ? maximum_keypoints : 8];
    __shared__ int n;
    __shared__ double normalization[6];
    __shared__ double normal[81];
    __shared__ RansacModel model;
    const int id = Refit ? 2*ransac_hypotheses + blockIdx.x : blockIdx.x;
    const int family = Refit ? blockIdx.x : blockIdx.x / ransac_hypotheses;
    if (threadIdx.x == 0) {
        model = {}; model.homography = family; n = 0;
        if (Refit) {
            const RansacModel& best = models[winners[family]];
            for (int i = 0; i < *count; ++i)
                if (errorSquared(best, matches[i]) <= threshold2) chosen[n++] = i;
        } else if (*count >= 8) {
            n = family ? 4 : 8;
            unsigned state = seed ^ (0x9e3779b9u * (id+1));
            if (!state) state = 1;
            for (int i = 0; i < n; ++i) {
                int candidate; bool duplicate;
                do {
                    candidate = randomNext(state) % *count; duplicate = false;
                    for (int j = 0; j < i; ++j) duplicate |= chosen[j] == candidate;
                } while (duplicate);
                chosen[i] = candidate;
            }
        }
        for (int j = 0; j < 6; ++j) normalization[j] = 0;
        if (n >= (family ? 4 : 8)) {
            for (int i = 0; i < n; ++i) {
                const DeviceMatch& p = matches[chosen[i]];
                normalization[0]+=static_cast<double>(p.ax)/n; normalization[1]+=static_cast<double>(p.ay)/n;
                normalization[3]+=static_cast<double>(p.bx)/n; normalization[4]+=static_cast<double>(p.by)/n;
            }
            double radius0 = 0, radius1 = 0;
            for (int i = 0; i < n; ++i) {
                const DeviceMatch& p = matches[chosen[i]];
                radius0 += hypot(p.ax-normalization[0],p.ay-normalization[1])/n;
                radius1 += hypot(p.bx-normalization[3],p.by-normalization[4])/n;
            }
            if (radius0 > 1e-6 && radius1 > 1e-6) {
                normalization[2] = sqrt(2.0)/radius0; normalization[5] = sqrt(2.0)/radius1;
                model.valid = 1;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 81) {
        double sum = 0;
        if (model.valid) for (int i = 0; i < n; ++i) for (int e = 0; e < (family ? 2 : 1); ++e) {
            double a[9]; row(matches[chosen[i]], normalization, family, e, a);
            sum += a[threadIdx.x/9]*a[threadIdx.x%9];
        }
        normal[threadIdx.x] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0 && model.valid) model.valid = solve(normal, normalization, model);
    __syncthreads();
    int inliers = 0; float cost = 0;
    for (int i = threadIdx.x; i < *count; i += blockDim.x) {
        const float residual = errorSquared(model, matches[i]);
        inliers += residual <= threshold2;
        cost += fminf(residual, threshold2);
    }
    const int sum_inliers = IntReduce(integers).Sum(inliers);
    const float sum_cost = FloatReduce(floats).Sum(cost);
    if (threadIdx.x == 0) {
        model.inliers = sum_inliers; model.cost = sum_cost; models[id] = model;
    }
}
__device__ bool better(const RansacModel* models, int a, int b) {
    return models[a].inliers > models[b].inliers ||
        (models[a].inliers == models[b].inliers && (models[a].cost < models[b].cost ||
        (models[a].cost == models[b].cost && a < b)));
}
__global__ void choose(const RansacModel* models, int* winners) {
    __shared__ int best[threads];
    const int start = blockIdx.x * ransac_hypotheses;
    int local = start;
    for (int i = start + threadIdx.x; i < start + ransac_hypotheses; i += blockDim.x)
        if (better(models, i, local)) local = i;
    best[threadIdx.x] = local;
    __syncthreads();
    for (int stride = threads/2; stride; stride /= 2) {
        if (threadIdx.x < stride && better(models, best[threadIdx.x+stride], best[threadIdx.x]))
            best[threadIdx.x] = best[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) winners[blockIdx.x] = best[0];
}
__global__ void summarize(const DeviceMatch* matches, const int* count, const RansacModel* models,
    const int* winners, unsigned char* mask, RansacSummary* summary, float threshold2,
    int w0, int h0, int w1, int h1) {
    using Sort = cub::BlockRadixSort<float, 256, 16>;
    __shared__ Sort::TempStorage sort_storage;
    __shared__ int winner;
    __shared__ unsigned coverage0, coverage1;
    if (threadIdx.x == 0) {
        int f = winners[0], h = winners[1];
        if (better(models, 2*ransac_hypotheses, f)) f = 2*ransac_hypotheses;
        if (better(models, 2*ransac_hypotheses+1, h)) h = 2*ransac_hypotheses+1;
        winner = models[h].inliers > models[f].inliers ? h : f;
        summary->inliers = models[winner].inliers;
        summary->homography = models[winner].homography;
        coverage0 = coverage1 = 0;
    }
    __syncthreads();
    float distances[16];
    unsigned bits0 = 0, bits1 = 0;
    for (int j = 0; j < 16; ++j) {
        const int i = threadIdx.x + j*blockDim.x;
        distances[j] = CUDART_INF_F;
        if (i < *count) {
            const DeviceMatch p = matches[i];
            const bool inlier = errorSquared(models[winner], p) <= threshold2;
            mask[i] = inlier;
            if (inlier) {
                bits0 |= 1u << (min(3,static_cast<int>(p.ay*4/h0))*4 + min(3,static_cast<int>(p.ax*4/w0)));
                bits1 |= 1u << (min(3,static_cast<int>(p.by*4/h1))*4 + min(3,static_cast<int>(p.bx*4/w1)));
                distances[j] = hypotf(p.ax-p.bx, p.ay-p.by);
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        bits0 |= __shfl_down_sync(warp_mask, bits0, offset);
        bits1 |= __shfl_down_sync(warp_mask, bits1, offset);
    }
    if (threadIdx.x % 32 == 0) { atomicOr(&coverage0, bits0); atomicOr(&coverage1, bits1); }
    // Sort accepts blocked input; sorting a full fixed 4096 slots puts all
    // finite inlier movements before infinity padding, regardless of layout.
    Sort(sort_storage).Sort(distances);
    const int median = summary->inliers / 2;
    if (summary->inliers > 0 && threadIdx.x == median/16)
        summary->displacement = distances[median%16] / hypotf(static_cast<float>(w0), static_cast<float>(h0));
    __syncthreads();
    if (threadIdx.x == 0) summary->coverage = min(__popc(coverage0),__popc(coverage1))/16.0f;
}
}
void verifyDeviceMatches(const void* p0, const void* s0, const void* p1, const void* s1,
    const std::int32_t* indices, const void* confidence, bool half, int count,
    int w0, int h0, int w1, int h1, float detector, float match_threshold, float threshold,
    unsigned seed, DeviceMatch* matches, int* match_count, RansacModel* models,
    int* winners, unsigned char* mask, RansacSummary* summary, cudaStream_t stream) {
    if (count < 8 || count > maximum_keypoints || w0 < 1 || h0 < 1 || w1 < 1 || h1 < 1 ||
        !std::isfinite(threshold) || threshold <= 0)
        throw std::invalid_argument("Invalid CUDA RANSAC dimensions or threshold");
    if (half) compact<<<1,threads,0,stream>>>(static_cast<const __half*>(p0),static_cast<const __half*>(s0),
        static_cast<const __half*>(p1),static_cast<const __half*>(s1),indices,static_cast<const __half*>(confidence),
        count,w0,h0,w1,h1,detector,match_threshold,matches,match_count,summary);
    else compact<<<1,threads,0,stream>>>(static_cast<const float*>(p0),static_cast<const float*>(s0),
        static_cast<const float*>(p1),static_cast<const float*>(s1),indices,static_cast<const float*>(confidence),
        count,w0,h0,w1,h1,detector,match_threshold,matches,match_count,summary);
    check(cudaGetLastError());
    const float threshold2 = threshold*threshold;
    hypotheses<false><<<2*ransac_hypotheses,threads,0,stream>>>(matches,match_count,models,winners,threshold2,seed);
    check(cudaGetLastError());
    choose<<<2,threads,0,stream>>>(models,winners);
    check(cudaGetLastError());
    hypotheses<true><<<2,threads,0,stream>>>(matches,match_count,models,winners,threshold2,seed);
    check(cudaGetLastError());
    summarize<<<1,256,0,stream>>>(matches,match_count,models,winners,mask,summary,threshold2,w0,h0,w1,h1);
    check(cudaGetLastError());
}
}  // namespace sceneforge::video
