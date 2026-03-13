// cuda_triangle.cu
// batched CUDA rasterizer. triangles are queued host-side and flushed
// in one kernel launch instead of one launch per triangle.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>

// forward declare rasterizer types
class TGAImage;
class TGAColor;
template<size_t DIM, typename T> struct vec;
typedef vec<3, float> Vec3f;
typedef vec<4, float> Vec4f;

struct CudaVec3 {
    float x, y, z;
    __device__ __host__ CudaVec3() : x(0), y(0), z(0) {}
    __device__ __host__ CudaVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct CudaVec4 {
    float x, y, z, w;
    __device__ __host__ CudaVec4() : x(0), y(0), z(0), w(1) {}
    __device__ __host__ CudaVec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct CudaColor {
    unsigned char r, g, b;
    __device__ __host__ CudaColor() : r(0), g(0), b(0) {}
    __device__ __host__ CudaColor(unsigned char r_, unsigned char g_, unsigned char b_)
        : r(r_), g(g_), b(b_) {}
};

// per-triangle data uploaded to GPU in one shot
struct CudaTriangle {
    CudaVec4 v[3];
    CudaColor color;
    // screen-space bounding box, precomputed on CPU so the kernel can skip early
    int bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y;
};

__device__
CudaVec3 cuda_barycentric(float ax, float ay, float bx, float by,
                          float cx, float cy, float px, float py) {
    float s0x = cx - ax;
    float s0y = cy - ay;
    float s1x = bx - ax;
    float s1y = by - ay;
    float s2x = ax - px;
    float s2y = ay - py;

    float cross_z = s0x * s1y - s0y * s1x;

    if (fabsf(cross_z) < 1e-6f) {
        return CudaVec3(-1, 1, 1);
    }

    float u = (s1y * s2x - s1x * s2y) / cross_z;
    float v = (s0x * s2y - s0y * s2x) / cross_z;

    return CudaVec3(1.0f - u - v, v, u);
}

// fill zbuffer without the old CPU malloc+loop+memcpy nonsense
__global__
void zbuffer_fill_kernel(int* zbuffer, int fill_val, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        zbuffer[idx] = fill_val;
}

// one thread per pixel, loops over all triangles.
// bbox check gives early-out so threads away from any geometry skip fast.
// zbuffer is int-typed so we can use atomicMin — works for positive floats
// because IEEE754 preserves ordering when reinterpreted as int (same sign bit)
__global__
void batched_raster_kernel(CudaTriangle* triangles, int num_triangles,
                           unsigned char* framebuffer, int* zbuffer,
                           int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;

    for (int t = 0; t < num_triangles; t++) {
        CudaTriangle& tri = triangles[t];

        // bbox early-out
        if (x < tri.bbox_min_x || x > tri.bbox_max_x ||
            y < tri.bbox_min_y || y > tri.bbox_max_y)
            continue;

        float v0x = tri.v[0].x / tri.v[0].w;
        float v0y = tri.v[0].y / tri.v[0].w;
        float v1x = tri.v[1].x / tri.v[1].w;
        float v1y = tri.v[1].y / tri.v[1].w;
        float v2x = tri.v[2].x / tri.v[2].w;
        float v2y = tri.v[2].y / tri.v[2].w;

        CudaVec3 bary = cuda_barycentric(v0x, v0y, v1x, v1y, v2x, v2y,
                                          (float)x, (float)y);

        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float z = tri.v[0].z * bary.x + tri.v[1].z * bary.y + tri.v[2].z * bary.z;
        float w = tri.v[0].w * bary.x + tri.v[1].w * bary.y + tri.v[2].w * bary.z;
        float depth = z / w;

        if (depth < 0.0f) continue;

        // atomic depth test via int reinterpretation.
        // the color write below isn't atomic with the depth — two tris at the
        // exact same depth could race on color. in practice this is a one-pixel
        // one-frame glitch that you'd never notice. the depth buffer itself
        // stays correct either way.
        int depth_int = __float_as_int(depth);
        int old = atomicMin(&zbuffer[pixel_idx], depth_int);

        if (depth_int <= old) {
            framebuffer[pixel_idx * 3 + 0] = tri.color.r;
            framebuffer[pixel_idx * 3 + 1] = tri.color.g;
            framebuffer[pixel_idx * 3 + 2] = tri.color.b;
        }
    }
}

// 64k triangles per batch (~2.5 MB staging buffer), auto-flushes if exceeded
static const int MAX_BATCH = 65536;

class CudaTriangleRasterizer {
private:
    CudaTriangle* d_triangles;
    unsigned char* d_framebuffer;
    int* d_zbuffer;
    int width, height;
    bool initialized;

    // host-side staging buffer — triangles accumulate here until flush
    std::vector<CudaTriangle> h_batch;

public:
    CudaTriangleRasterizer(int w, int h) : width(w), height(h), initialized(false) {
        cudaError_t err;

        err = cudaMalloc(&d_triangles, MAX_BATCH * sizeof(CudaTriangle));
        if (err != cudaSuccess) {
            printf("CUDA malloc triangles failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaMalloc(&d_framebuffer, width * height * 3);
        if (err != cudaSuccess) {
            printf("CUDA malloc framebuffer failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles);
            return;
        }

        err = cudaMalloc(&d_zbuffer, width * height * sizeof(int));
        if (err != cudaSuccess) {
            printf("CUDA malloc zbuffer failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles);
            cudaFree(d_framebuffer);
            return;
        }

        h_batch.reserve(4096);
        initialized = true;
        printf("CUDA rasterizer initialized: %dx%d (batched)\n", width, height);
    }

    ~CudaTriangleRasterizer() {
        if (initialized) {
            cudaFree(d_triangles);
            cudaFree(d_framebuffer);
            cudaFree(d_zbuffer);
        }
    }

    bool isInitialized() const { return initialized; }

    void clear() {
        if (!initialized) return;

        cudaMemset(d_framebuffer, 0, width * height * 3);

        // zbuffer stores ints (float-as-int), fill with int rep of 1000.0f
        union { float f; int i; } far_val;
        far_val.f = 1000.0f;

        int pixel_count = width * height;
        int block = 256;
        int grid = (pixel_count + block - 1) / block;
        zbuffer_fill_kernel<<<grid, block>>>(d_zbuffer, far_val.i, pixel_count);

        h_batch.clear();
    }

    // just queue on CPU, no GPU work yet
    void submitTriangle(float v0x, float v0y, float v0z, float v0w,
                        float v1x, float v1y, float v1z, float v1w,
                        float v2x, float v2y, float v2z, float v2w,
                        unsigned char r, unsigned char g, unsigned char b) {
        if (!initialized) return;

        CudaTriangle tri;
        tri.v[0] = CudaVec4(v0x, v0y, v0z, v0w);
        tri.v[1] = CudaVec4(v1x, v1y, v1z, v1w);
        tri.v[2] = CudaVec4(v2x, v2y, v2z, v2w);
        tri.color = CudaColor(r, g, b);

        // compute screen-space bbox from perspective-divided coords
        float sx0 = v0x / v0w, sy0 = v0y / v0w;
        float sx1 = v1x / v1w, sy1 = v1y / v1w;
        float sx2 = v2x / v2w, sy2 = v2y / v2w;

        float fmin_x = fminf(fminf(sx0, sx1), sx2);
        float fmax_x = fmaxf(fmaxf(sx0, sx1), sx2);
        float fmin_y = fminf(fminf(sy0, sy1), sy2);
        float fmax_y = fmaxf(fmaxf(sy0, sy1), sy2);

        tri.bbox_min_x = std::max(0, (int)floorf(fmin_x));
        tri.bbox_max_x = std::min(width - 1, (int)ceilf(fmax_x));
        tri.bbox_min_y = std::max(0, (int)floorf(fmin_y));
        tri.bbox_max_y = std::min(height - 1, (int)ceilf(fmax_y));

        // completely offscreen, don't even queue it
        if (tri.bbox_min_x > tri.bbox_max_x || tri.bbox_min_y > tri.bbox_max_y)
            return;

        h_batch.push_back(tri);

        if ((int)h_batch.size() >= MAX_BATCH)
            flush();
    }

    // upload queued triangles and rasterize in one kernel launch
    void flush() {
        if (!initialized || h_batch.empty()) return;

        int num_tris = (int)h_batch.size();

        cudaMemcpy(d_triangles, h_batch.data(), num_tris * sizeof(CudaTriangle),
                   cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

        batched_raster_kernel<<<gridSize, blockSize>>>(
            d_triangles, num_tris, d_framebuffer, d_zbuffer, width, height
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        }

        h_batch.clear();
    }

    void copyToCPU(unsigned char* host_framebuffer, float* host_zbuffer) {
        if (!initialized) return;

        flush();
        cudaDeviceSynchronize();

        cudaMemcpy(host_framebuffer, d_framebuffer, width * height * 3,
                   cudaMemcpyDeviceToHost);

        // zbuffer is int on GPU, convert back to float
        int* h_zbuf_int = new int[width * height];
        cudaMemcpy(h_zbuf_int, d_zbuffer, width * height * sizeof(int),
                   cudaMemcpyDeviceToHost);

        union { int i; float f; } conv;
        for (int idx = 0; idx < width * height; idx++) {
            conv.i = h_zbuf_int[idx];
            host_zbuffer[idx] = conv.f;
        }
        delete[] h_zbuf_int;
    }

    void synchronize() {
        if (initialized) {
            flush();
            cudaDeviceSynchronize();
        }
    }
};

static CudaTriangleRasterizer* g_cuda_rasterizer = nullptr;

#include "geometry.h"
#include "tgaimage.h"

// C interface — same signatures Engine.cpp already calls
extern "C" {
    bool initCudaRasterizer(int width, int height) {
        if (g_cuda_rasterizer) delete g_cuda_rasterizer;

        g_cuda_rasterizer = new CudaTriangleRasterizer(width, height);
        return g_cuda_rasterizer->isInitialized();
    }

    void cleanupCudaRasterizer() {
        if (g_cuda_rasterizer) {
            delete g_cuda_rasterizer;
            g_cuda_rasterizer = nullptr;
        }
    }

    void cudaClearBuffers() {
        if (g_cuda_rasterizer) {
            g_cuda_rasterizer->clear();
        }
    }

    // called per-triangle from Engine.cpp — just queues, no GPU work yet
    void cudaRenderTriangle(const Vec4f& v0, const Vec4f& v1, const Vec4f& v2,
                            const TGAColor& tga_color) {
        if (g_cuda_rasterizer) {
            TGAColor& color = const_cast<TGAColor&>(tga_color);

            g_cuda_rasterizer->submitTriangle(
                v0[0], v0[1], v0[2], v0[3],
                v1[0], v1[1], v1[2], v1[3],
                v2[0], v2[1], v2[2], v2[3],
                color.bgra[2], color.bgra[1], color.bgra[0]
            );
        }
    }

    void cudaCopyResults(TGAImage& framebuffer, TGAImage& zbuffer) {
        if (!g_cuda_rasterizer) return;

        int width = framebuffer.get_width();
        int height = framebuffer.get_height();

        unsigned char* h_framebuffer = new unsigned char[width * height * 3];
        float* h_zbuffer = new float[width * height];

        g_cuda_rasterizer->copyToCPU(h_framebuffer, h_zbuffer);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                TGAColor color(h_framebuffer[idx * 3 + 0],
                              h_framebuffer[idx * 3 + 1],
                              h_framebuffer[idx * 3 + 2]);
                framebuffer.set(x, y, color);

                unsigned char depth = (unsigned char)(h_zbuffer[idx] / 10.0f);
                zbuffer.set(x, y, TGAColor(depth));
            }
        }

        delete[] h_framebuffer;
        delete[] h_zbuffer;
    }
}
