#include <stdio.h>
#include <cmath>
#include <cfloat>

#include "common.cuh"
#include "stages.cuh"

// The rasterizer's only light is one directional lamp plus a flat ambient
// term, so cavities read exactly as bright as flat surfaces and the image
// looks pasted together. SSAO recovers the contact shading a path tracer gets
// for free: for each pixel, sample a hemisphere around its normal and count
// how many of those sample points are buried behind geometry the camera can
// already see.
//
// Positions are reconstructed rather than stored. screen = Viewport *
// Projection * eye, and Viewport * Projection is the same for every mesh in a
// frame (only ModelView differs), so inverting it turns any (x, y, depth)
// straight back into eye space without a position buffer. The normals do have
// to be written out, which is what normalbuf is for.
//
// Convention note: this z-buffer keeps the LARGEST depth as nearest, and eye
// space has the camera at +z, so "closer to the camera" is "greater z" in both
// spaces. An occluder is therefore one whose eye z is greater than the sample
// point's.
__device__ __forceinline__
void mat4_mul_point(const float* m, float x, float y, float z,
                    float& ox, float& oy, float& oz)
{
    float w = m[12] * x + m[13] * y + m[14] * z + m[15];
    if (fabsf(w) < 1e-12f) w = 1e-12f;
    ox = (m[0] * x + m[1] * y + m[2]  * z + m[3])  / w;
    oy = (m[4] * x + m[5] * y + m[6]  * z + m[7])  / w;
    oz = (m[8] * x + m[9] * y + m[10] * z + m[11]) / w;
}

// integer hash, for the per-pixel rotation that breaks up banding
__device__ __forceinline__ unsigned int ssao_hash(unsigned int v)
{
    v ^= v >> 16; v *= 0x7feb352dU;
    v ^= v >> 15; v *= 0x846ca68bU;
    v ^= v >> 16;
    return v;
}

__global__
void ssao_kernel(const int* zbuffer, const float* normalbuf, float* ao,
                 const float* inv_vp, const float* vp,
                 const float* kernel_samples,
                 float radius, float bias, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    float depth = __int_as_float(zbuffer[idx]);
    if (depth <= 0.0f) { ao[idx] = 1.0f; return; }   // background

    float nx = normalbuf[idx * 3 + 0];
    float ny = normalbuf[idx * 3 + 1];
    float nz = normalbuf[idx * 3 + 2];
    float nlen = sqrtf(nx * nx + ny * ny + nz * nz);
    if (nlen < 1e-6f) { ao[idx] = 1.0f; return; }
    nx /= nlen; ny /= nlen; nz /= nlen;

    float px, py, pz;
    mat4_mul_point(inv_vp, (float)x, (float)y, depth, px, py, pz);

    // random in-plane rotation so neighbouring pixels use different sample
    // directions; the blur pass then averages the noise away
    unsigned int h = ssao_hash((unsigned int)(y * width + x) * 2654435761u);
    float ang = (h & 0xFFFF) * (6.2831853f / 65536.0f);
    float rc = cosf(ang), rs = sinf(ang);

    // Gram-Schmidt a tangent frame off an arbitrary axis
    float ax = (fabsf(nz) < 0.9f) ? 0.0f : 1.0f;
    float ay = (fabsf(nz) < 0.9f) ? 0.0f : 0.0f;
    float az = (fabsf(nz) < 0.9f) ? 1.0f : 0.0f;
    float tx = ay * nz - az * ny;
    float ty = az * nx - ax * nz;
    float tz = ax * ny - ay * nx;
    float tl = sqrtf(tx * tx + ty * ty + tz * tz);
    if (tl < 1e-6f) { ao[idx] = 1.0f; return; }
    tx /= tl; ty /= tl; tz /= tl;
    float bx = ny * tz - nz * ty;
    float by = nz * tx - nx * tz;
    float bz = nx * ty - ny * tx;

    // How fast the surface itself recedes across the sampling footprint. A
    // face-on surface (|nz| = 1) barely changes depth from one sample to the
    // next; a surface seen edge-on changes a lot, and a fixed bias then reads
    // that slope as occlusion and covers everything in acne. Scaling the bias
    // by the slope is what separates "the surface tilts away here" from
    // "something is actually in front of this point".
    float slope = sqrtf(fmaxf(0.0f, 1.0f - nz * nz)) / fmaxf(fabsf(nz), 0.15f);

    float occlusion = 0.0f;
    for (int i = 0; i < SSAO_SAMPLES; i++) {
        float sx = kernel_samples[i * 3 + 0];
        float sy = kernel_samples[i * 3 + 1];
        float sz = kernel_samples[i * 3 + 2];

        // rotate the sample about the normal
        float rx = sx * rc - sy * rs;
        float ry = sx * rs + sy * rc;

        // tangent space -> eye space
        float ex = tx * rx + bx * ry + nx * sz;
        float ey = ty * rx + by * ry + ny * sz;
        float ez = tz * rx + bz * ry + nz * sz;

        float qx = px + ex * radius;
        float qy = py + ey * radius;
        float qz = pz + ez * radius;

        // project the sample point back to the screen
        float ux, uy, uz;
        mat4_mul_point(vp, qx, qy, qz, ux, uy, uz);
        int ix = (int)(ux + 0.5f), iy = (int)(uy + 0.5f);
        if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;

        float sdepth = __int_as_float(zbuffer[iy * width + ix]);
        if (sdepth <= 0.0f) continue;            // nothing drawn there

        // what the camera actually sees along that ray, in eye space
        float ox2, oy2, oz2;
        mat4_mul_point(inv_vp, (float)ix, (float)iy, sdepth, ox2, oy2, oz2);

        // the footprint of this particular sample, and what the surface alone
        // would do over that distance
        float ox = qx - px, oy = qy - py, oz = qz - pz;
        float off = sqrtf(ox * ox + oy * oy + oz * oz);
        float bias_eff = bias + off * slope;

        // greater eye z means nearer the camera, so this occludes the sample
        if (oz2 >= qz + bias_eff) {
            // fade out occluders that sit far outside the sampling radius,
            // otherwise a silhouette throws a dark halo onto whatever is
            // behind it. smoothstep rather than a hard cutoff so the
            // transition does not show up as an edge.
            float dz = fabsf(pz - oz2);
            float t = fminf(1.0f, radius / fmaxf(dz, 1e-6f));
            occlusion += t * t * (3.0f - 2.0f * t);
        }
    }

    ao[idx] = fmaxf(0.0f, 1.0f - occlusion / (float)SSAO_SAMPLES);
}

// separable would be cheaper, but the noise here is a per-pixel rotation
// rather than a tiled pattern, so a small box blur is enough to clean it up
__global__
void ssao_blur_kernel(const float* ao_in, float* ao_out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int R = 2;
    float sum = 0.0f;
    int n = 0;
    for (int dy = -R; dy <= R; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (int dx = -R; dx <= R; dx++) {
            int xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            sum += ao_in[yy * width + xx];
            n++;
        }
    }
    ao_out[y * width + x] = n ? sum / n : 1.0f;
}

// writes the occlusion term straight to the frame as greyscale, so the term
// can be inspected on its own instead of inferred from the shaded result
__global__
void ssao_debug_kernel(float4* framebuffer, const float* ao,
                       const float* normalbuf, const int* zbuffer,
                       const float* inv_vp, const float* vp,
                       int mode, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    int ci = (height - 1 - y) * width + x;

    // These write 0..1 like every other producer, but they are measurements
    // rather than light: the tone map is told to pass them through.
    if (mode == 2) {
        // eye-space normal as rgb, 0.5 grey is zero
        float nx = normalbuf[idx * 3 + 0];
        float ny = normalbuf[idx * 3 + 1];
        float nz = normalbuf[idx * 3 + 2];
        framebuffer[ci] = make_float4(fminf(1.f, fmaxf(0.f, nx * 0.5f + 0.5f)),
                                      fminf(1.f, fmaxf(0.f, ny * 0.5f + 0.5f)),
                                      fminf(1.f, fmaxf(0.f, nz * 0.5f + 0.5f)), 1.f);
        return;
    }
    if (mode == 4) {
        // round-trip test: screen -> eye -> screen must land where it started.
        // red is x error, green y, blue depth error, all scaled by 16.
        float d = __int_as_float(zbuffer[idx]);
        if (d <= 0.0f) {
            framebuffer[ci] = make_float4(0.f, 0.f, 0.f, 1.f); return;
        }
        float ex, ey, ez;
        mat4_mul_point(inv_vp, (float)x, (float)y, d, ex, ey, ez);
        float rx, ry, rz;
        mat4_mul_point(vp, ex, ey, ez, rx, ry, rz);
        float e0 = fabsf(rx - (float)x) * (16.0f / 255.f);
        float e1 = fabsf(ry - (float)y) * (16.0f / 255.f);
        float e2 = fabsf(rz - d) * (16.0f / 255.f);
        framebuffer[ci] = make_float4(fminf(1.f, e0), fminf(1.f, e1),
                                      fminf(1.f, e2), 1.f);
        return;
    }
    if (mode == 3) {
        // raw screen depth, which spans 0..255
        float d = __int_as_float(zbuffer[idx]) * (1.f / 255.f);
        float v = fminf(1.f, fmaxf(0.f, d));
        framebuffer[ci] = make_float4(v, v, v, 1.f);
        return;
    }
    float a = fminf(1.0f, fmaxf(0.0f, ao[idx]));
    framebuffer[ci] = make_float4(a, a, a, 1.f);
}

// multiplies the occlusion into the finished frame. the framebuffer is
// top-down and everything above is bottom-up, hence the flip on the index.
__global__
void ssao_apply_kernel(float4* framebuffer, const float* ao,
                       float intensity, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float a = ao[y * width + x];
    a = 1.0f - intensity * (1.0f - a);      // intensity 0 disables, 1 is full
    a = fminf(1.0f, fmaxf(0.0f, a));

    int ci = (height - 1 - y) * width + x;
    float4 c = framebuffer[ci];
    framebuffer[ci] = make_float4(c.x * a, c.y * a, c.z * a, c.w);
}

// The whole occlusion pass. Debug mode replaces the apply step rather than
// following it, so the term can be looked at instead of its effect.
void cudaLaunchSSAO(const int* zbuffer, const float* normalbuf,
                    float* ao, float* ao_blur, float4* framebuffer,
                    const float* inv_vp, const float* vp,
                    const float* kernel_samples,
                    float radius, float bias, float intensity,
                    int ssao_debug, int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    ssao_kernel<<<grid, block>>>(zbuffer, normalbuf, ao,
                                 inv_vp, vp, kernel_samples,
                                 radius, bias, width, height);
    ssao_blur_kernel<<<grid, block>>>(ao, ao_blur, width, height);
    if (ssao_debug)
        ssao_debug_kernel<<<grid, block>>>(framebuffer, ao_blur, normalbuf,
                                           zbuffer, inv_vp, vp,
                                           ssao_debug, width, height);
    else
        ssao_apply_kernel<<<grid, block>>>(framebuffer, ao_blur,
                                           intensity, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("SSAO kernel error: %s\n", cudaGetErrorString(err));
}
