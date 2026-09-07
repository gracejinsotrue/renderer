// The stages that touch the finished frame rather than geometry: the backdrop
// the frame is cleared to, the supersample resolve, and the tone map.
//
// The colour target is linear float, unbounded above. Nothing in this file
// clamps until tone_map_kernel, which is the one place that has to produce a
// number a display can show. That ordering is what lets a sun in the
// environment stay a sun through the resolve instead of being 255 like every
// other bright pixel.
#include <cmath>

#include "common.cuh"
#include "stages.cuh"

// Supersampling resolve: average each ss x ss block of the oversampled frame
// down to one output pixel. Both buffers are stored top-down in the same R,G,B
// order, so this is a straight block mean with no flip.
//
// This is what anti-aliases the image. Coverage was decided at ss*ss points
// inside every output pixel instead of one, so a silhouette that crosses a
// pixel comes out proportionally blended instead of all-or-nothing.
__global__
void downsample_kernel(const float4* src, float4* dst,
                       int out_w, int out_h, int ss)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    int src_w = out_w * ss;
    float inv_n = 1.0f / (float)(ss * ss);
    float ar = 0.f, ag = 0.f, ab = 0.f;

    for (int j = 0; j < ss; j++) {
        const float4* row = src + (size_t)(y * ss + j) * src_w + x * ss;
        for (int i = 0; i < ss; i++) {
            ar += row[i].x;
            ag += row[i].y;
            ab += row[i].z;
        }
    }

    // Averaged in linear radiance, before the curve. Averaging display values
    // instead would make a half-covered edge against a bright sky land at the
    // wrong brightness, because the curve is not linear and the mean of two
    // curved values is not the curve of their mean.
    dst[(size_t)y * out_w + x] = make_float4(ar * inv_n, ag * inv_n, ab * inv_n, 1.f);
}

__global__
void clear_colour_kernel(float4* framebuffer, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) framebuffer[i] = make_float4(0.f, 0.f, 0.f, 1.f);
}

// Stretches the background image over the whole render target, under the
// geometry: clear() runs this in place of the framebuffer memset.
__global__
void background_kernel(float4* framebuffer, cudaTextureObject_t bg,
                       int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / (float)width;
    float v = (y + 0.5f) / (float)height;
    float4 c = tex2D<float4>(bg, u, v);

    // Taken as linear, though a TGA off disk is almost certainly sRGB. Model
    // textures are read the same way, so decoding here alone would put the
    // backdrop on a different footing from the geometry in front of it. Both
    // want doing together, with the material model.
    framebuffer[(size_t)y * width + x] = make_float4(c.z, c.y, c.x, 1.f);
}

__device__ inline
CudaVec3 unproject(const Mat4& inv, float x, float y, float z)
{
    float ox = inv.m[0]*x  + inv.m[1]*y  + inv.m[2]*z  + inv.m[3];
    float oy = inv.m[4]*x  + inv.m[5]*y  + inv.m[6]*z  + inv.m[7];
    float oz = inv.m[8]*x  + inv.m[9]*y  + inv.m[10]*z + inv.m[11];
    float ow = inv.m[12]*x + inv.m[13]*y + inv.m[14]*z + inv.m[15];
    float s = (fabsf(ow) > 1e-12f) ? 1.0f / ow : 0.0f;
    return CudaVec3(ox * s, oy * s, oz * s);
}

// The environment as the frame's backdrop. Unlike background_kernel this
// tracks the camera: each pixel is turned back into a world-space ray and the
// map is sampled along it, so the backdrop moves when the view does.
//
// Two unprojections rather than a stored camera position: the weak-perspective
// Projection here is not a standard frustum, so recovering the eye point from
// it is fiddlier than just taking two depths on the same pixel and subtracting.
__global__
void environment_kernel(float4* framebuffer, cudaTextureObject_t env,
                        Mat4 inv_vp, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // the frame is stored top-down; the screen space inv_vp inverts is not
    float sx = (float)x + 0.5f;
    float sy = (float)(height - 1 - y) + 0.5f;

    // Viewport maps z into 0..255, and the depth test resolves by atomicMax,
    // so 255 is the NEAR end. Subtracting the other way gives a ray pointing
    // back at the camera, which lands on the opposite side of the map in both
    // axes at once.
    CudaVec3 near_p = unproject(inv_vp, sx, sy, 255.f);
    CudaVec3 far_p  = unproject(inv_vp, sx, sy, 0.f);
    float dx = far_p.x - near_p.x, dy = far_p.y - near_p.y, dz = far_p.z - near_p.z;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    if (len < 1e-12f) return;
    dx /= len; dy /= len; dz /= len;

    float u, v;
    equirect_uv(dx, dy, dz, &u, &v);

    // .x=R here, where the LDR background texture is BGRA and reads .z=R:
    // this one is uploaded straight from float RGBA rather than from a TGA.
    // Written as the radiance it is, with no exposure and no curve; the tone
    // map applies both, once, to the whole frame.
    float4 c = tex2D<float4>(env, u, v);
    framebuffer[(size_t)y * width + x] = make_float4(c.x, c.y, c.z, 1.f);
}

// Narkowicz's fit to the ACES filmic curve (2015). Cheap enough to be free
// here and the shape most renderers are judged against, which matters more
// than the exact coefficients: it rolls highlights off instead of clipping
// them, so the sun in an environment map reads as bright rather than as a
// flat white disc.
__device__ inline float aces(float x)
{
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    float y = (x * (a * x + b)) / (x * (c * x + d) + e);
    return fminf(fmaxf(y, 0.f), 1.f);
}

__device__ inline unsigned char encode_srgb(float v)
{
    v = fminf(fmaxf(v, 0.f), 1.f);
    float s = (v <= 0.0031308f) ? (12.92f * v)
                                : (1.055f * __powf(v, 1.f / 2.4f) - 0.055f);
    return (unsigned char)(s * 255.f + 0.5f);
}

// The only stage that produces a number a display can show.
//
// passthrough exists because two callers put things in this buffer that are
// not light: the SSAO debug views write normals and occlusion, and the tests
// that compare against the CPU rasterizer need the shading path's own values
// rather than a display transform of them. Both want the raw 0..1 scaled to
// 0..255 and nothing else.
__global__
void tone_map_kernel(const float4* src, unsigned char* dst,
                     int width, int height, float exposure, int passthrough)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 c = src[(size_t)y * width + x];
    unsigned char* o = dst + ((size_t)y * width + x) * 3;

    if (passthrough) {
        o[0] = (unsigned char)fminf(fmaxf(c.x, 0.f) * 255.f, 255.f);
        o[1] = (unsigned char)fminf(fmaxf(c.y, 0.f) * 255.f, 255.f);
        o[2] = (unsigned char)fminf(fmaxf(c.z, 0.f) * 255.f, 255.f);
        return;
    }

    o[0] = encode_srgb(aces(c.x * exposure));
    o[1] = encode_srgb(aces(c.y * exposure));
    o[2] = encode_srgb(aces(c.z * exposure));
}

void cudaLaunchClearColour(float4* framebuffer, int count)
{
    int block = 256;
    clear_colour_kernel<<<(count + block - 1) / block, block>>>(framebuffer, count);
}

void cudaLaunchDownsample(const float4* src, float4* dst,
                          int out_w, int out_h, int ss)
{
    dim3 block(16, 16);
    dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
    downsample_kernel<<<grid, block>>>(src, dst, out_w, out_h, ss);
}

void cudaLaunchBackground(float4* framebuffer, cudaTextureObject_t bg,
                          int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    background_kernel<<<grid, block>>>(framebuffer, bg, width, height);
}

void cudaLaunchEnvironment(float4* framebuffer, cudaTextureObject_t env,
                           Mat4 inv_vp, int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    environment_kernel<<<grid, block>>>(framebuffer, env, inv_vp, width, height);
}

void cudaLaunchToneMap(const float4* src, unsigned char* dst,
                       int width, int height, float exposure, int passthrough)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    tone_map_kernel<<<grid, block>>>(src, dst, width, height, exposure, passthrough);
}
