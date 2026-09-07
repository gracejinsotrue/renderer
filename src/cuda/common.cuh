// Data the whole pipeline agrees on: the triangle and material layouts the
// kernels read, and the sizes the host allocates against. Anything here is
// shared by at least two stages; a type used by one stage lives with it.
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

// The equirectangular convention, in one place because three stages now use
// it: the backdrop samples the environment, the convolution walks it, and the
// shader samples the irradiance map built from it. Longitude runs around Y,
// latitude from +Y down, which puts row 0 of a decoded .hdr at the top.
__device__ inline
void equirect_uv(float dx, float dy, float dz, float* u, float* v)
{
    const float INV_TWO_PI = 0.15915494309189535f;
    const float INV_PI     = 0.31830988618379067f;
    *u = atan2f(dz, dx) * INV_TWO_PI + 0.5f;
    *v = acosf(fminf(fmaxf(dy, -1.f), 1.f)) * INV_PI;
}

__device__ inline
void equirect_dir(float u, float v, float* dx, float* dy, float* dz)
{
    const float TWO_PI = 6.283185307179586f;
    const float PI     = 3.141592653589793f;
    float phi = (u - 0.5f) * TWO_PI;
    float theta = v * PI;
    float st = sinf(theta);
    *dx = cosf(phi) * st;
    *dy = cosf(theta);
    *dz = sinf(phi) * st;
}

// per-triangle data uploaded to GPU in one shot
struct CudaTriangle {
    CudaVec4 v[3];
    // eye-space normals and uvs per corner, interpolated per fragment
    float nx[3], ny[3], nz[3];
    float u[3], vt[3];
    CudaColor color;
    int mat;              // index into the per-frame material table
    // screen-space bounding box, precomputed on CPU so the kernel can skip early
    int bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y;
};

// Prefiltered specular levels. Level 0 is roughness 0 and level
// IBL_SPEC_LEVELS-1 is roughness 1.
#define IBL_SPEC_LEVELS 5

// Level 0 is a straight reduction rather than a convolution -- a mirror's lobe
// is a delta function -- so it is cheap and kept large: it is the only level
// where the environment's own detail survives, and a chrome surface reflecting
// a 128x64 map looks like frosted glass however good the source was.
//
// The convolved levels are small on purpose. A lobe that wide cannot carry the
// detail a larger map would hold, and the convolution is O(output x source), so
// each doubling costs four times as much for something nobody can see.
__host__ __device__ inline
void ibl_spec_size(int level, int* w, int* h)
{
    switch (level) {
        case 0:  *w = 512; *h = 256; break;
        case 1:  *w = 128; *h = 64;  break;
        case 2:  *w = 64;  *h = 32;  break;
        case 3:  *w = 32;  *h = 16;  break;
        default: *w = 16;  *h = 8;   break;
    }
}

// The environment BRDF table, over (n.v, roughness). Small because it is
// smooth: the function has no features a larger table would resolve.
#define IBL_BRDF_LUT_SIZE 64
// Monte Carlo budget per texel. The table stops changing visibly here.
#define IBL_BRDF_LUT_SAMPLES 1024

// one per mesh per frame. the raster kernel rasterizes every mesh in a single
// launch, so per-mesh state has to be reachable from the triangle itself.
struct CudaMaterial {
    cudaTextureObject_t diffuse, nm, spec;
    int has_diffuse, has_nm, has_spec;
    int unlit;            // slot 0: emit the triangle colour verbatim
    int has_shadow;
    float shadow_bias;
    int debug_shadow;     // 1: emit (sz, stored, 0) instead of shading
    // maps a fragment's screen-space position into shadow-map space:
    // light_viewport*proj*modelview * inverse(camera_viewport*proj*modelview)
    float mshadow[16];
    float lx, ly, lz;     // light direction in eye space for this mesh
    float lcr, lcg, lcb;  // light colour in RGB, 0..1
    float lintensity;     // scales the direct term, ambient stays fixed
    // 3x3 of the inverse-transpose. the normal map is object-space, so the
    // sampled normal must be carried to eye space like the vertex normals.
    // using it raw collapses the diffuse term to near zero.
    float mit[9];
    // Diffuse image-based lighting. The map holds E/pi for each direction, so
    // the shader multiplies it by albedo and nothing else. Frame-global, and
    // copied per mesh for the same reason the light above is: the material
    // table is the only thing a triangle can reach from inside the kernel.
    cudaTextureObject_t irradiance;
    int has_irradiance;
    float ibl_intensity;
    // eye -> world rotation, so an eye-space normal can index a map that is
    // built in world space. The camera rotation is orthonormal, so this is
    // just its transpose.
    float e2w[9];
    // Metallic-roughness, scalar for the whole mesh. Metals have no diffuse
    // response and take their Fresnel colour from the albedo; dielectrics
    // reflect a colourless 4%. Roughness is perceptual, squared to GGX alpha
    // inside the BRDF.
    float metallic;
    float roughness;
    // Specular image-based lighting, split-sum. prefiltered[i] is the
    // environment blurred by the GGX lobe at roughness i/(levels-1); brdf_lut
    // is the scale and bias applied to F0. The shader lerps between two levels
    // rather than relying on a mip chain, because each level is its own
    // equirectangular map at its own size.
    cudaTextureObject_t prefiltered[IBL_SPEC_LEVELS];
    cudaTextureObject_t brdf_lut;
    int has_prefiltered;
    int has_brdf_lut;
};
// starting capacity only. the table is uploaded fresh each flush and grown on
// demand, so there is no cap on how many meshes a frame may draw.
static const int MATERIALS_INITIAL = 64;

// Screen tiling. Each triangle is binned into the tiles its bbox touches,
// then each tile rasterizes only its own list, instead of every pixel walking
// every triangle.
#define TILE_W 16
#define TILE_H 16

// per-frame triangle capacity. rumi is ~190k faces across its meshes, and
// whole scenes are transformed in one go. ~19 MB at 72 bytes a triangle.
static const int MAX_BATCH = 262144;

// row-major 4x4, passed to the setup kernel by value
struct Mat4 { float m[16]; };

// One queued mesh. Geometry pointers are device-resident and owned by the
// DeviceMesh; only the matrices change from frame to frame.
struct MeshDraw {
    const float* verts;
    const int*   faces;
    const float* cnorms;
    const float* cuvs;
    int nfaces;
    int face_begin;        // running total over the queue, so one launch can
                           // cover every mesh and each thread find its own
    Mat4 mvp, clip, mit;
    int mat_id;
    CudaColor color;
};

// The irradiance map's size. A cosine lobe removes essentially all angular
// detail, so this is as much as the result can carry; the whole map is 512
// texels.
#define IRRADIANCE_W 32
#define IRRADIANCE_H 16

// The hemisphere sample count. It sizes a host-side staging array as well as
// the kernel's loop, so both sides have to see the same number.
#define SSAO_SAMPLES 24
