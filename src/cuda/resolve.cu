// The stages that touch the finished frame rather than geometry: the
// supersample resolve, and the background the frame is cleared to -- either a
// flat image or an equirectangular environment map.
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
void downsample_kernel(const unsigned char* src, unsigned char* dst,
                       int out_w, int out_h, int ss)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    int src_w = out_w * ss;
    int n = ss * ss;
    int acc[3] = {0, 0, 0};

    for (int j = 0; j < ss; j++) {
        const unsigned char* row = src + ((size_t)(y * ss + j) * src_w + x * ss) * 3;
        for (int i = 0; i < ss; i++) {
            acc[0] += row[i * 3 + 0];
            acc[1] += row[i * 3 + 1];
            acc[2] += row[i * 3 + 2];
        }
    }

    unsigned char* o = dst + ((size_t)y * out_w + x) * 3;
    // + n/2 rounds to nearest rather than truncating, which otherwise biases
    // the whole frame a fraction of a level darker.
    o[0] = (unsigned char)((acc[0] + n / 2) / n);
    o[1] = (unsigned char)((acc[1] + n / 2) / n);
    o[2] = (unsigned char)((acc[2] + n / 2) / n);
}

// Stretches the background image over the whole render target, under the
// geometry: clear() runs this in place of the framebuffer memset.
__global__
void background_kernel(unsigned char* framebuffer, cudaTextureObject_t bg,
                       int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / (float)width;
    float v = (y + 0.5f) / (float)height;
    float4 c = tex2D<float4>(bg, u, v);

    unsigned char* o = framebuffer + ((size_t)y * width + x) * 3;
    o[0] = (unsigned char)(c.z * 255.0f);
    o[1] = (unsigned char)(c.y * 255.0f);
    o[2] = (unsigned char)(c.x * 255.0f);
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
void environment_kernel(unsigned char* framebuffer, cudaTextureObject_t env,
                        Mat4 inv_vp, float exposure, int width, int height)
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

    // Equirectangular: longitude around Y, latitude from +Y down. v is not
    // flipped because row 0 of the decoded .hdr is the top of the sphere.
    const float INV_TWO_PI = 0.15915494309189535f;
    const float INV_PI     = 0.31830988618379067f;
    float u = atan2f(dz, dx) * INV_TWO_PI + 0.5f;
    float v = acosf(fminf(fmaxf(dy, -1.f), 1.f)) * INV_PI;

    // .x=R here, where the LDR background texture is BGRA and reads .z=R:
    // this one is uploaded straight from float RGBA rather than from a TGA.
    float4 c = tex2D<float4>(env, u, v);
    float r = c.x * exposure, g = c.y * exposure, bl = c.z * exposure;

    // Reinhard, and no gamma: the shading path writes its values into this
    // same 8-bit frame with no encode of its own, so an sRGB step here alone
    // would put the backdrop and the geometry in different spaces. Both move
    // together when the framebuffer becomes float.
    r = r / (1.f + r);
    g = g / (1.f + g);
    bl = bl / (1.f + bl);

    unsigned char* o = framebuffer + ((size_t)y * width + x) * 3;
    o[0] = (unsigned char)(fminf(r,  1.f) * 255.f + 0.5f);
    o[1] = (unsigned char)(fminf(g,  1.f) * 255.f + 0.5f);
    o[2] = (unsigned char)(fminf(bl, 1.f) * 255.f + 0.5f);
}

void cudaLaunchDownsample(const unsigned char* src, unsigned char* dst,
                          int out_w, int out_h, int ss)
{
    dim3 block(16, 16);
    dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);
    downsample_kernel<<<grid, block>>>(src, dst, out_w, out_h, ss);
}

void cudaLaunchBackground(unsigned char* framebuffer, cudaTextureObject_t bg,
                          int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    background_kernel<<<grid, block>>>(framebuffer, bg, width, height);
}

void cudaLaunchEnvironment(unsigned char* framebuffer, cudaTextureObject_t env,
                           Mat4 inv_vp, float exposure, int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    environment_kernel<<<grid, block>>>(framebuffer, env, inv_vp, exposure,
                                        width, height);
}
