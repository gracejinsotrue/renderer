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

// The hemisphere sample count. It sizes a host-side staging array as well as
// the kernel's loop, so both sides have to see the same number.
#define SSAO_SAMPLES 24
