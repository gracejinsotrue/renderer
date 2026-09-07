// Diffuse image-based lighting: turning an environment map into the irradiance
// arriving at a surface from every direction.
//
//     E(n) = integral over the hemisphere of L(w) * max(0, n . w) dw
//
// What is stored is E/pi, so a Lambertian surface is just albedo * E/pi and
// the shader carries no constants of its own. A constant environment L then
// stores exactly L, which is what test_environment checks: the integral of
// cosine over a hemisphere is pi, and getting that factor wrong is the whole
// bug surface of this file.
//
// Runs once when an environment is loaded, not per frame.
#include <cmath>

#include "common.cuh"
#include "stages.cuh"

// The cosine lobe is about as low-frequency as a filter gets, so the source
// is reduced to this before being convolved. 64x32 x 32x16 is two million
// multiply-adds, against four billion straight off a 4K map for an answer
// that would not differ visibly.
#define IBL_CONV_W 64
#define IBL_CONV_H 32

// Box average rather than point sampling. A sun can be four texels wide in a
// 4K map and would simply be missed by 2048 point samples, taking its energy
// with it. Averaging preserves the integral over each cell, which is the only
// property the convolution below actually needs.
__global__
void env_reduce_kernel(cudaTextureObject_t env, float4* dst,
                       int dst_w, int dst_h, int src_w, int src_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int x0 = (int)((long long)x * src_w / dst_w);
    int x1 = (int)((long long)(x + 1) * src_w / dst_w);
    int y0 = (int)((long long)y * src_h / dst_h);
    int y1 = (int)((long long)(y + 1) * src_h / dst_h);
    if (x1 <= x0) x1 = x0 + 1;
    if (y1 <= y0) y1 = y0 + 1;

    float r = 0.f, g = 0.f, b = 0.f;
    int n = 0;
    for (int j = y0; j < y1; j++) {
        for (int i = x0; i < x1; i++) {
            float4 c = tex2D<float4>(env, (i + 0.5f) / src_w, (j + 0.5f) / src_h);
            r += c.x; g += c.y; b += c.z;
            n++;
        }
    }
    float inv = 1.0f / (float)n;
    dst[(size_t)y * dst_w + x] = make_float4(r * inv, g * inv, b * inv, 1.f);
}

__global__
void irradiance_kernel(const float4* src, int src_w, int src_h,
                       float4* dst, int dst_w, int dst_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float nx, ny, nz;
    equirect_dir((x + 0.5f) / dst_w, (y + 0.5f) / dst_h, &nx, &ny, &nz);

    const float PI = 3.141592653589793f;
    // solid angle of one source texel, less the sin(theta) that varies by row
    float dtheta = PI / (float)src_h;
    float dphi = 2.0f * PI / (float)src_w;
    float cell = dtheta * dphi;

    float r = 0.f, g = 0.f, b = 0.f;
    for (int j = 0; j < src_h; j++) {
        float theta = (j + 0.5f) / src_h * PI;
        float st = sinf(theta);
        float wy = cosf(theta);
        // rows near the poles subtend almost nothing; without this the map
        // comes out with the poles weighted as heavily as the equator
        float row = st * cell;
        for (int i = 0; i < src_w; i++) {
            float phi = ((i + 0.5f) / src_w - 0.5f) * 2.0f * PI;
            float wx = cosf(phi) * st;
            float wz = sinf(phi) * st;

            float ndotw = nx * wx + ny * wy + nz * wz;
            if (ndotw <= 0.f) continue;

            float4 L = src[(size_t)j * src_w + i];
            float w = ndotw * row;
            r += L.x * w; g += L.y * w; b += L.z * w;
        }
    }

    float inv_pi = 1.0f / PI;
    dst[(size_t)y * dst_w + x] = make_float4(r * inv_pi, g * inv_pi, b * inv_pi, 1.f);
}

int cudaIrradianceConvWidth()  { return IBL_CONV_W; }
int cudaIrradianceConvHeight() { return IBL_CONV_H; }

void cudaLaunchEnvReduce(cudaTextureObject_t env, float4* dst,
                         int dst_w, int dst_h, int src_w, int src_h)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    env_reduce_kernel<<<grid, block>>>(env, dst, dst_w, dst_h, src_w, src_h);
}

void cudaLaunchIrradiance(const float4* src, int src_w, int src_h,
                          float4* dst, int dst_w, int dst_h)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    irradiance_kernel<<<grid, block>>>(src, src_w, src_h, dst, dst_w, dst_h);
}
