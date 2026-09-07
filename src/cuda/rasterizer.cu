// Owns every device allocation and sequences the pipeline stages. The kernels
// themselves live in the stage files; nothing here launches one directly.
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>

#include "common.cuh"
#include "stages.cuh"

// forward declare rasterizer types
class TGAImage;
class TGAColor;
template<size_t DIM, typename T> struct vec;
typedef vec<3, float> Vec3f;
typedef vec<4, float> Vec4f;

static bool g_use_linear_filter = true;

// Which of the two colour paths runs. Deferred shades once per pixel; forward
// shades inside the depth loop and so pays per fragment that ever won. Kept
// switchable at run time so the two can be compared inside one process, the
// same reason the present paths are.
static bool g_deferred_shading = true;

// geometry that lives on the device across frames
struct DeviceMesh {
    float* d_verts;
    int*   d_faces;
    float* d_norms;    // per-corner, nfaces*3*3, may be NULL
    float* d_uvs;      // per-corner, nfaces*3*2, may be NULL
    cudaArray_t  tex_arr[3];      // diffuse, normal map, specular
    cudaTextureObject_t tex[3];
    bool has_tex[3];
    int nverts, nfaces;
    bool alive;
};

// uploads one TGAImage-style buffer (BGR/BGRA/greyscale, bottom-up already
// flipped by Model::load_texture) as an RGBA8 texture.
static bool uploadTexture(const unsigned char* px, int w, int h, int bpp,
                          cudaArray_t* arr_out, cudaTextureObject_t* tex_out)
{
    if (!px || w <= 0 || h <= 0) return false;
    std::vector<unsigned char> rgba((size_t)w * h * 4);
    for (int i = 0; i < w * h; i++) {
        unsigned char b = px[(size_t)i * bpp + 0];
        unsigned char g = (bpp > 1) ? px[(size_t)i * bpp + 1] : b;
        unsigned char r = (bpp > 2) ? px[(size_t)i * bpp + 2] : b;
        rgba[(size_t)i * 4 + 0] = b;   // keep BGRA order; the shader indexes
        rgba[(size_t)i * 4 + 1] = g;   // .x=B .y=G .z=R to match TGAColor
        rgba[(size_t)i * 4 + 2] = r;
        rgba[(size_t)i * 4 + 3] = 255;
    }
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    if (cudaMallocArray(arr_out, &ch, w, h) != cudaSuccess) return false;
    cudaMemcpy2DToArray(*arr_out, 0, 0, rgba.data(), (size_t)w * 4,
                        (size_t)w * 4, h, cudaMemcpyHostToDevice);

    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = *arr_out;

    cudaTextureDesc td; memset(&td, 0, sizeof(td));
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    // matches the CPU path's bilinear sampler, so the two rasterizers agree.
    td.filterMode = g_use_linear_filter ? cudaFilterModeLinear : cudaFilterModePoint;
    td.readMode = cudaReadModeNormalizedFloat;
    td.normalizedCoords = 1;
    return cudaCreateTextureObject(tex_out, &rd, &td, NULL) == cudaSuccess;
}

// Linear RGBA float, uploaded as-is. Separate from uploadTexture because that
// one exists to match the CPU sampler on TGA data: it swizzles to BGRA,
// normalizes to 0..1 and clamps in both axes, none of which an HDR
// environment wants.
static bool uploadFloatTexture(const float* rgba, int w, int h,
                               cudaArray_t* arr_out, cudaTextureObject_t* tex_out)
{
    if (!rgba || w <= 0 || h <= 0) return false;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();
    if (cudaMallocArray(arr_out, &ch, w, h) != cudaSuccess) return false;
    size_t pitch = (size_t)w * 4 * sizeof(float);
    cudaMemcpy2DToArray(*arr_out, 0, 0, rgba, pitch, pitch, h,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = *arr_out;

    cudaTextureDesc td; memset(&td, 0, sizeof(td));
    // u wraps because 0 and 1 are the same meridian; v clamps because the
    // poles are ends, not a seam.
    td.addressMode[0] = cudaAddressModeWrap;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    // not NormalizedFloat: these values are radiance and routinely exceed 1
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 1;
    return cudaCreateTextureObject(tex_out, &rd, &td, NULL) == cudaSuccess;
}

class CudaTriangleRasterizer {
private:
    CudaTriangle* d_triangles;
    // Linear radiance, unbounded above, at render resolution. Every colour
    // producer writes here; only the tone map reads it and turns it into
    // something a display can show. float4 rather than three floats so a
    // pixel is one 16-byte transaction.
    float4* d_hdr;
    int* d_zbuffer;
    int* d_tile_counts;    // one per tile, reset each flush
    int* d_tile_offsets;   // exclusive prefix sum of the counts
    int* d_tile_cursor;    // write cursor per tile during scatter
    int* d_tri_indices;    // flat, exactly sum(counts) long. grown on demand.
    size_t idx_capacity;   // entries currently allocated in d_tri_indices
    void*  d_scan_temp;    // cub scratch
    size_t scan_temp_bytes;
    int    stat_bin_entries;  // total (triangle, tile) pairs last flush
    int*   d_tri_count;       // triangles appended by mesh_setup_kernel
    int*   d_stats;           // [culled_back, culled_off, dropped, submitted,
                          //  shade_invocations]
    CudaMaterial* d_materials;   // one slot per mesh drawn this frame
    size_t mat_capacity;         // slots currently allocated in d_materials
    MeshDraw* d_draws;           // the frame's queued meshes
    size_t draw_capacity;
    int*   d_shadowbuf;          // depth from the light's point of view
    // Deferred path: one 64-bit word per pixel, depth in the high half and the
    // winning triangle index in the low half. Only allocated once; the forward
    // path simply never reads it.
    unsigned long long* d_visbuffer;

    // SSAO. normals are bottom-up like the zbuffer; ao/ao_blur are one float
    // per pixel. d_ssao_kernel holds the fixed hemisphere sample set.
    float* d_normalbuf;
    float* d_ao;
    float* d_ao_blur;
    float* d_ssao_kernel;
    float* d_ssao_inv_vp;
    float* d_ssao_vp;
    // staged host-side and uploaded in one memcpy per flush. slot 0 is always
    // the neutral pass-through material the host staging path points at.
    std::vector<CudaMaterial> h_materials;
    // meshes queued since the last flush, with their running face total.
    // drawMesh only appends; flush() launches setup once for all of them.
    std::vector<MeshDraw> h_draws;
    int h_draw_faces;

    // width/height are the RENDER dimensions, which are out_width/out_height
    // scaled by ss. Every buffer and kernel above works at render resolution;
    // only the resolve and the two readback paths use the output size.
    int width, height;
    int out_width, out_height;
    int ss;
    float4* d_hdr_resolved;     // NULL when ss == 1: the frame is already 1:1
    // The tone map's output, and the only 8-bit colour in the pipeline.
    // Always out_width x out_height, top-down R,G,B.
    unsigned char* d_ldr;
    float tone_exposure;
    // Off for the differential tests, which compare the shading path against
    // the CPU rasterizer and want its numbers rather than a display transform
    // of them.
    bool tone_enabled;
    // set by applySSAO when a debug view is up, cleared by clear(). those
    // views carry normals and occlusion, which a tone curve would misreport.
    bool tone_passthrough;
    int tiles_x, tiles_y, num_tiles;

    // optional HDR environment, drawn by clear() as the frame's backdrop. It
    // does not survive a resize; the engine re-uploads it. Sampling it needs
    // the camera, so env_inv_vp is refreshed by the engine every frame; it is
    // stale by exactly one frame if the engine forgets, which shows up as a
    // backdrop that lags the view rather than as anything harder to see.
    cudaArray_t d_env_arr;
    cudaTextureObject_t d_env_tex;
    bool has_env;
    Mat4 env_inv_vp;

    // Diffuse IBL, built from the environment once at load. Small on purpose:
    // a cosine lobe cannot carry detail finer than this.
    cudaArray_t d_irr_arr;
    cudaTextureObject_t d_irr_tex;
    bool has_irr;
    float ibl_intensity;
    float ibl_e2w[9];

    // per-stage GPU timing. nsys can't get a GPU timeline through WSL2 and
    // ncu needs a driver permission change, so the kernels time themselves.
    cudaEvent_t ev_start, ev_upload, ev_bin, ev_raster;
    bool timing_ready;
    bool initialized;

    // host-side staging buffer: triangles accumulate here until flush
    std::vector<CudaTriangle> h_batch;

    // per-frame counters, reset in clear(). the kernel walks every queued
    // triangle for every pixel, so "kept" is what actually drives frame cost.
    int stat_submitted, stat_culled_back, stat_culled_offscreen;

public:
    CudaTriangleRasterizer(int out_w, int out_h, int ss_factor)
        : width(out_w * ss_factor), height(out_h * ss_factor),
          out_width(out_w), out_height(out_h), ss(ss_factor),
          d_hdr_resolved(NULL), d_ldr(NULL), tone_exposure(1.0f),
          tone_enabled(true), tone_passthrough(false),
          d_env_arr(NULL), d_env_tex(0), has_env(false),
          d_irr_arr(NULL), d_irr_tex(0), has_irr(false), ibl_intensity(1.0f),
          d_draws(NULL), draw_capacity(0), h_draw_faces(0),
          initialized(false),
          stat_submitted(0), stat_culled_back(0), stat_culled_offscreen(0) {
        // identity until the engine sets a real one, so an environment drawn
        // before the first setEnvironmentView is wrong rather than undefined
        for (int i = 0; i < 16; i++) env_inv_vp.m[i] = (i % 5 == 0) ? 1.f : 0.f;
        for (int i = 0; i < 9; i++) ibl_e2w[i] = (i % 4 == 0) ? 1.f : 0.f;

        int w = width, h = height;
        tiles_x = (w + TILE_W - 1) / TILE_W;
        tiles_y = (h + TILE_H - 1) / TILE_H;
        num_tiles = tiles_x * tiles_y;

        cudaError_t err;

        err = cudaMalloc(&d_triangles, MAX_BATCH * sizeof(CudaTriangle));
        if (err != cudaSuccess) {
            printf("CUDA malloc triangles failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaMalloc(&d_hdr, (size_t)width * height * sizeof(float4));
        if (err != cudaSuccess) {
            printf("CUDA malloc framebuffer failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles);
            return;
        }

        err = cudaMalloc(&d_ldr, (size_t)out_width * out_height * 3);
        if (err != cudaSuccess) {
            printf("CUDA malloc display frame failed: %s", cudaGetErrorString(err));
            cudaFree(d_triangles); cudaFree(d_hdr);
            return;
        }

        err = cudaMalloc(&d_zbuffer, width * height * sizeof(int));
        if (err != cudaSuccess) {
            printf("CUDA malloc zbuffer failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles);
            cudaFree(d_hdr); cudaFree(d_ldr);
            return;
        }

        err = cudaMalloc(&d_tile_counts, num_tiles * sizeof(int));
        if (err != cudaSuccess) {
            printf("CUDA malloc tile counts failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_triangles); cudaFree(d_hdr); cudaFree(d_ldr); cudaFree(d_zbuffer);
            return;
        }

        cudaMalloc(&d_tile_offsets, num_tiles * sizeof(int));
        cudaMalloc(&d_tile_cursor,  num_tiles * sizeof(int));

        // 8 tile-entries per triangle to start, grown on demand.
        idx_capacity = (size_t)MAX_BATCH * 8;
        cudaMalloc(&d_tri_indices, idx_capacity * sizeof(int));

        cudaMalloc(&d_tri_count, sizeof(int));
        cudaMemset(d_tri_count, 0, sizeof(int));
        cudaMalloc(&d_stats, 8 * sizeof(int));
        cudaMemset(d_stats, 0, 8 * sizeof(int));
        mat_capacity = MATERIALS_INITIAL;
        cudaMalloc(&d_materials, mat_capacity * sizeof(CudaMaterial));
        cudaMalloc(&d_shadowbuf, (size_t)w * h * sizeof(int));

        cudaMalloc(&d_visbuffer, (size_t)w * h * sizeof(unsigned long long));

        cudaMalloc(&d_normalbuf, (size_t)w * h * 3 * sizeof(float));
        cudaMalloc(&d_ao,        (size_t)w * h * sizeof(float));
        cudaMalloc(&d_ao_blur,   (size_t)w * h * sizeof(float));
        cudaMalloc(&d_ssao_kernel, SSAO_SAMPLES * 3 * sizeof(float));
        cudaMalloc(&d_ssao_inv_vp, 16 * sizeof(float));
        cudaMalloc(&d_ssao_vp,     16 * sizeof(float));
        if (ss > 1) {
            err = cudaMalloc(&d_hdr_resolved, (size_t)out_width * out_height * sizeof(float4));
            if (err != cudaSuccess) {
                printf("CUDA malloc resolve buffer failed: %s\n", cudaGetErrorString(err));
                return;
            }
        }
        uploadSSAOKernel();
        cudaMemset(d_shadowbuf, 0, (size_t)w * h * sizeof(int));
        resetMaterials();

        d_scan_temp = NULL;
        scan_temp_bytes = cudaBinScanTempBytes(num_tiles);
        cudaMalloc(&d_scan_temp, scan_temp_bytes);
        stat_bin_entries = 0;

        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_upload);
        cudaEventCreate(&ev_bin);
        cudaEventCreate(&ev_raster);
        timing_ready = false;

        h_batch.reserve(4096);
        initialized = true;
        if (ss > 1)
            printf("CUDA rasterizer initialized: %dx%d -> %dx%d (%dx SSAA), "
                   "%dx%d tiles of %dx%d\n",
                   width, height, out_width, out_height, ss,
                   tiles_x, tiles_y, TILE_W, TILE_H);
        else
            printf("CUDA rasterizer initialized: %dx%d, %dx%d tiles of %dx%d\n",
                   width, height, tiles_x, tiles_y, TILE_W, TILE_H);
    }

    ~CudaTriangleRasterizer() {
        if (initialized) {
            cudaFree(d_triangles);
            cudaFree(d_hdr); cudaFree(d_ldr);
            cudaFree(d_zbuffer);
            cudaFree(d_tile_counts);
            cudaFree(d_tile_offsets);
            cudaFree(d_tile_cursor);
            cudaFree(d_tri_indices);
            cudaFree(d_scan_temp);
            cudaFree(d_tri_count);
            cudaFree(d_stats);
            cudaFree(d_materials);
            if (d_draws) cudaFree(d_draws);
            cudaFree(d_shadowbuf);
            cudaFree(d_visbuffer);
            cudaFree(d_normalbuf);
            cudaFree(d_ao);
            cudaFree(d_ao_blur);
            cudaFree(d_ssao_kernel);
            cudaFree(d_ssao_inv_vp);
            cudaFree(d_ssao_vp);
            if (d_hdr_resolved) cudaFree(d_hdr_resolved);
            clearEnvironment();
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_upload);
            cudaEventDestroy(ev_bin);
            cudaEventDestroy(ev_raster);
        }
    }

    bool isInitialized() const { return initialized; }

    // A fixed hemisphere sample set, built once. Samples are pushed toward the
    // origin so most of them land close to the shaded point, which is where
    // occlusion actually matters; spreading them evenly through the hemisphere
    // wastes most of the budget on distant geometry that barely darkens
    // anything.
    void uploadSSAOKernel() {
        float h_kernel[SSAO_SAMPLES * 3];
        unsigned int seed = 12345u;
        auto rnd = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)((seed >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
        };
        for (int i = 0; i < SSAO_SAMPLES; i++) {
            float x, y, z, len;
            do {
                x = rnd() * 2.0f - 1.0f;
                y = rnd() * 2.0f - 1.0f;
                z = rnd();                       // hemisphere: z >= 0
                len = sqrtf(x * x + y * y + z * z);
            } while (len < 1e-4f || len > 1.0f);
            x /= len; y /= len; z /= len;

            // Clustered toward the shaded point, but not right on top of it:
            // a sample only a pixel or two away cannot see past the surface's
            // own curvature and contributes nothing but noise.
            float t = (float)i / (float)SSAO_SAMPLES;
            float scale = 0.35f + 0.65f * t * t;
            h_kernel[i * 3 + 0] = x * scale;
            h_kernel[i * 3 + 1] = y * scale;
            h_kernel[i * 3 + 2] = z * scale;
        }
        cudaMemcpy(d_ssao_kernel, h_kernel, sizeof(h_kernel),
                   cudaMemcpyHostToDevice);
    }

    // Occludes the finished frame in place. inv_vp16 and vp16 are
    // inverse(Viewport*Projection) and Viewport*Projection for this frame,
    // row-major; both are mesh-independent, which is what lets the kernel
    // rebuild eye-space positions from the depth buffer alone.
    void applySSAO(const float* inv_vp16, const float* vp16,
                   float radius, float intensity, float bias,
                   int ssao_debug = 0) {
        if (!initialized || intensity <= 0.0f) return;
        flush();

        tone_passthrough = (ssao_debug != 0);

        cudaMemcpy(d_ssao_inv_vp, inv_vp16, 16 * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_ssao_vp, vp16, 16 * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaLaunchSSAO(d_zbuffer, d_normalbuf, d_ao, d_ao_blur, d_hdr,
                       d_ssao_inv_vp, d_ssao_vp, d_ssao_kernel,
                       radius, bias, intensity, ssao_debug, width, height);
    }

    void clear() {
        if (!initialized) return;

        tone_passthrough = false;

        if (has_env) {
            cudaLaunchEnvironment(d_hdr, d_env_tex, env_inv_vp, width, height);
        } else {
            // not a memset: black is four floats, and only three of them are
            // zero. alpha stays 1 so the buffer is always a valid colour.
            cudaLaunchClearColour(d_hdr, width * height);
        }
        cudaMemset(d_normalbuf, 0, (size_t)width * height * 3 * sizeof(float));

        // zbuffer stores ints (float-as-int). nearest is the MAXIMUM depth here,
        // so the "empty" value is 0 (+0.0f) and any visible fragment beats it.
        union { float f; int i; } far_val;
        far_val.f = 0.0f;

        cudaLaunchZbufferFill(d_zbuffer, far_val.i, width * height);

        // Deferred: 0 means "no triangle covers this pixel", which is why the
        // packed index is stored biased by one.
        cudaLaunchVisbufferFill(d_visbuffer, width * height);

        h_batch.clear();
        h_draws.clear();
        h_draw_faces = 0;
        stat_submitted = stat_culled_back = stat_culled_offscreen = 0;
        cudaMemset(d_tri_count, 0, sizeof(int));
        cudaMemset(d_stats, 0, 8 * sizeof(int));
        // zero the table so slot 0 is a valid neutral material: the host
        // staging path points every triangle at it, and reading an
        // uninitialized cudaTextureObject_t would fault.
        cudaMemset(d_shadowbuf, 0, (size_t)width * height * sizeof(int));
        resetMaterials();
    }

    // drops every per-mesh material, leaving slot 0 as the neutral
    // pass-through cudaRenderTriangle points at. safe after any flush: the
    // triangles referencing the old slots have been consumed.
    void resetMaterials() {
        h_materials.clear();
        CudaMaterial neutral;
        memset(&neutral, 0, sizeof(neutral));
        neutral.unlit = 1;
        h_materials.push_back(neutral);
    }

    // milliseconds for the most recent flush. syncs on the last event only.
    void getKernelTimings(float* upload_ms, float* bin_ms, float* raster_ms) {
        if (upload_ms) *upload_ms = 0.f;
        if (bin_ms)    *bin_ms    = 0.f;
        if (raster_ms) *raster_ms = 0.f;
        if (!initialized || !timing_ready) return;
        cudaEventSynchronize(ev_raster);
        if (upload_ms) cudaEventElapsedTime(upload_ms, ev_start,  ev_upload);
        if (bin_ms)    cudaEventElapsedTime(bin_ms,    ev_upload, ev_bin);
        if (raster_ms) cudaEventElapsedTime(raster_ms, ev_bin,    ev_raster);
    }

    void getStats(int* submitted, int* culled_back, int* culled_offscreen,
                  int* bin_entries) {
        int ds[4] = {0,0,0,0};
        if (initialized)
            cudaMemcpy(ds, d_stats, 4 * sizeof(int), cudaMemcpyDeviceToHost);
        if (submitted)        *submitted        = stat_submitted        + ds[3];
        if (culled_back)      *culled_back      = stat_culled_back      + ds[0];
        if (culled_offscreen) *culled_offscreen = stat_culled_offscreen + ds[1];
        // Not a cull count and not an error: bins are sized exactly by the
        // prefix sum. This is (triangle, tile) pairs, i.e. how far binning
        // fans out, and is normally larger than the triangle count.
        if (bin_entries) *bin_entries = stat_bin_entries;
    }

    // just queue on CPU, no GPU work yet
    void submitTriangle(float v0x, float v0y, float v0z, float v0w,
                        float v1x, float v1y, float v1z, float v1w,
                        float v2x, float v2y, float v2z, float v2w,
                        unsigned char r, unsigned char g, unsigned char b) {
        if (!initialized) return;
        stat_submitted++;

        CudaTriangle tri;
        tri.v[0] = CudaVec4(v0x, v0y, v0z, v0w);
        tri.v[1] = CudaVec4(v1x, v1y, v1z, v1w);
        tri.v[2] = CudaVec4(v2x, v2y, v2z, v2w);
        tri.color = CudaColor(r, g, b);

        // compute screen-space bbox from perspective-divided coords
        float sx0 = v0x / v0w, sy0 = v0y / v0w;
        float sx1 = v1x / v1w, sy1 = v1y / v1w;
        float sx2 = v2x / v2w, sy2 = v2y / v2w;

        // backface cull, same sign convention as mesh_setup_kernel above
        float facing = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0);
        if (facing <= 0.0f) {
            stat_culled_back++;
            return;
        }

        float fmin_x = fminf(fminf(sx0, sx1), sx2);
        float fmax_x = fmaxf(fmaxf(sx0, sx1), sx2);
        float fmin_y = fminf(fminf(sy0, sy1), sy2);
        float fmax_y = fmaxf(fmaxf(sy0, sy1), sy2);

        tri.bbox_min_x = std::max(0, (int)floorf(fmin_x));
        tri.bbox_max_x = std::min(width - 1, (int)ceilf(fmax_x));
        tri.bbox_min_y = std::max(0, (int)floorf(fmin_y));
        tri.bbox_max_y = std::min(height - 1, (int)ceilf(fmax_y));

        // completely offscreen, don't even queue it
        if (tri.bbox_min_x > tri.bbox_max_x || tri.bbox_min_y > tri.bbox_max_y) {
            stat_culled_offscreen++;
            return;
        }

        // the host staging path has no material of its own; slot 0 is reserved
        // as a neutral one so cudaRenderTriangle keeps working for the tests
        tri.mat = 0;
        for (int k = 0; k < 3; k++) {
            tri.nx[k] = 0.f; tri.ny[k] = 0.f; tri.nz[k] = 1.f;
            tri.u[k] = 0.f;  tri.vt[k] = 0.f;
        }
        h_batch.push_back(tri);

        if ((int)h_batch.size() >= MAX_BATCH)
            flush();
    }

    // upload queued triangles and rasterize in one kernel launch
    // shadow_pass: bin exactly the same way, then write depth only.
    void flush(bool shadow_pass = false) {
        if (!initialized) return;

        cudaEventRecord(ev_start);

        // one launch for every mesh queued since the last flush
        if (!h_draws.empty()) {
            if (h_draws.size() > draw_capacity) {
                if (d_draws) cudaFree(d_draws);
                draw_capacity = h_draws.size() + h_draws.size() / 2;
                cudaError_t de = cudaMalloc(&d_draws,
                                            draw_capacity * sizeof(MeshDraw));
                if (de != cudaSuccess) {
                    printf("CUDA draw table alloc failed (%zu slots): %s\n",
                           draw_capacity, cudaGetErrorString(de));
                    d_draws = NULL; draw_capacity = 0;
                    h_draws.clear(); h_draw_faces = 0;
                    h_batch.clear(); resetMaterials();
                    return;
                }
            }
            cudaMemcpy(d_draws, h_draws.data(),
                       h_draws.size() * sizeof(MeshDraw), cudaMemcpyHostToDevice);

            cudaLaunchMeshSetup(d_draws, (int)h_draws.size(), h_draw_faces,
                                d_triangles, d_tri_count, MAX_BATCH,
                                width, height, d_stats);
        }

        // the setup kernel appends straight into d_triangles, so the device
        // owns the count now. the host staging path (still used by the tests
        // and cudaRenderTriangle) appends after whatever the meshes wrote.
        int num_tris = 0;
        cudaMemcpy(&num_tris, d_tri_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (!h_batch.empty()) {
            int n = (int)h_batch.size();
            if (num_tris + n > MAX_BATCH) n = MAX_BATCH - num_tris;
            if (n > 0) {
                cudaMemcpy(d_triangles + num_tris, h_batch.data(),
                           n * sizeof(CudaTriangle), cudaMemcpyHostToDevice);
                num_tris += n;
                cudaMemcpy(d_tri_count, &num_tris, sizeof(int),
                           cudaMemcpyHostToDevice);
            }
            h_batch.clear();
        }

        // one upload per flush for the whole material table, grown if this
        // frame drew more meshes than any previous one
        if (h_materials.size() > mat_capacity) {
            if (d_materials) cudaFree(d_materials);
            mat_capacity = h_materials.size() + h_materials.size() / 2;
            cudaError_t me = cudaMalloc(&d_materials,
                                        mat_capacity * sizeof(CudaMaterial));
            if (me != cudaSuccess) {
                printf("CUDA material table alloc failed (%zu slots): %s\n",
                       mat_capacity, cudaGetErrorString(me));
                d_materials = NULL; mat_capacity = 0;
                h_batch.clear(); resetMaterials();
                return;
            }
        }
        cudaMemcpy(d_materials, h_materials.data(),
                   h_materials.size() * sizeof(CudaMaterial),
                   cudaMemcpyHostToDevice);

        cudaEventRecord(ev_upload);

        if (num_tris > MAX_BATCH) num_tris = MAX_BATCH;   // setup kernel overran
        if (num_tris == 0) {
            timing_ready = false;
            h_draws.clear(); h_draw_faces = 0;
            resetMaterials();
            return;
        }

        // pass 1a: count (triangle, tile) pairs. nothing is written yet, so
        // there is no capacity to exceed.
        cudaMemset(d_tile_counts, 0, num_tiles * sizeof(int));

        cudaBinCount(d_triangles, num_tris, d_tile_counts, tiles_x, tiles_y);

        // pass 1b: exclusive prefix sum gives each tile its exact slice
        cudaBinScan(d_scan_temp, scan_temp_bytes, d_tile_counts, d_tile_offsets,
                    num_tiles);

        // total entries = last offset + last count. one 8-byte readback per
        // flush; it goes away once geometry lives on the device permanently.
        int last_off = 0, last_cnt = 0;
        cudaMemcpy(&last_off, d_tile_offsets + num_tiles - 1, sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_cnt, d_tile_counts  + num_tiles - 1, sizeof(int),
                   cudaMemcpyDeviceToHost);
        size_t total = (size_t)last_off + (size_t)last_cnt;
        stat_bin_entries = (int)total;

        if (total > idx_capacity) {
            cudaFree(d_tri_indices);
            idx_capacity = total + total / 2;      // grow with headroom
            cudaError_t ge = cudaMalloc(&d_tri_indices, idx_capacity * sizeof(int));
            if (ge != cudaSuccess) {
                printf("CUDA tile index buffer alloc failed (%zu entries): %s\n",
                       idx_capacity, cudaGetErrorString(ge));
                d_tri_indices = NULL; idx_capacity = 0;
                h_batch.clear();
                return;
            }
        }

        cudaMemset(d_tile_cursor, 0, num_tiles * sizeof(int));
        cudaBinScatter(d_triangles, num_tris, d_tile_offsets, d_tile_cursor,
                       d_tri_indices, tiles_x, tiles_y);

        cudaEventRecord(ev_bin);

        // pass 2: one block per tile, one thread per pixel in it
        if (shadow_pass) {
            cudaLaunchShadowRaster(d_triangles, d_tile_counts, d_tile_offsets,
                                   d_tri_indices, d_shadowbuf,
                                   width, height, tiles_x, tiles_y);
        } else if (g_deferred_shading) {
            // Front half: decide who is visible, shade nothing.
            cudaLaunchVisibilityRaster(d_triangles, d_tile_counts,
                                       d_tile_offsets, d_tri_indices,
                                       d_visbuffer,
                                       width, height, tiles_x, tiles_y);
            // Back half: one shade per covered pixel, whatever the overdraw was.
            cudaLaunchDeferredShade(d_triangles, d_materials, d_visbuffer,
                                    d_hdr, d_zbuffer, d_shadowbuf,
                                    d_normalbuf, width, height, d_stats);
        } else {
            cudaLaunchTiledRaster(d_triangles, d_tile_counts, d_tile_offsets,
                                  d_tri_indices, d_materials, d_hdr,
                                  d_zbuffer, d_shadowbuf, d_normalbuf,
                                  width, height, tiles_x, tiles_y, d_stats);
        }

        cudaEventRecord(ev_raster);
        timing_ready = true;

        // consumed: the next flush accumulates from zero. the material ids
        // baked into those triangles die with them, so the table resets too
        // and the shadow pass does not eat the colour pass's slots.
        cudaMemset(d_tri_count, 0, sizeof(int));
        h_draws.clear();
        h_draw_faces = 0;
        resetMaterials();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        }

        h_batch.clear();
    }

    // DMA the finished frame straight into a locked SDL texture. the kernel
    // already wrote it top-down in R,G,B, so no conversion is needed and one
    // 2D memcpy replaces ~2M per-pixel host operations a frame.
    // dst_pitch is SDL's row stride, which may be wider than width * 3.
    // Resolve, then tone map. Always out_width x out_height, and the single
    // place linear radiance becomes displayable: everything upstream of it is
    // float and everything downstream is the 8-bit frame.
    unsigned char* resolvedFramebuffer() {
        const float4* src = d_hdr;
        if (ss > 1) {
            cudaLaunchDownsample(d_hdr, d_hdr_resolved, out_width, out_height, ss);
            src = d_hdr_resolved;
        }
        int passthrough = (tone_passthrough || !tone_enabled) ? 1 : 0;
        cudaLaunchToneMap(src, d_ldr, out_width, out_height, tone_exposure,
                          passthrough);
        return d_ldr;
    }

    // Exposure is a per-frame display control, not scene state, so it lives
    // here rather than travelling with the environment.
    void setExposure(float e) { tone_exposure = e > 0.f ? e : 0.f; }
    void setToneMapping(bool on) { tone_enabled = on; }

    unsigned char* deviceFramebuffer(int* w, int* h) {
        if (w) *w = out_width;
        if (h) *h = out_height;
        if (!initialized) return NULL;
        flush();
        return resolvedFramebuffer();
    }

    void blitToTexture(void* dst, int dst_pitch) {
        if (!initialized || !dst) return;

        flush();

        const unsigned char* src = resolvedFramebuffer();
        cudaError_t err = cudaMemcpy2D(dst, (size_t)dst_pitch,
                                       src, (size_t)out_width * 3,
                                       (size_t)out_width * 3, (size_t)out_height,
                                       cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA blit failed: %s\n", cudaGetErrorString(err));
        }
    }

    // slow path, only for writing a TGA. nothing composites on the host, so
    // the interactive path never calls this.
    // host_framebuffer is out_width * out_height * 3, top-down R,G,B.
    void copyToCPU(unsigned char* host_framebuffer) {
        if (!initialized) return;

        flush();

        cudaMemcpy(host_framebuffer, resolvedFramebuffer(),
                   (size_t)out_width * out_height * 3,
                   cudaMemcpyDeviceToHost);
    }

    // ---- persistent device geometry -------------------------------------
    std::vector<DeviceMesh>& meshes() { static std::vector<DeviceMesh> m; return m; }

    int createMesh(const float* verts, int nverts, const int* faces, int nfaces,
                   const float* cnorms, const float* cuvs) {
        if (!initialized) return -1;
        DeviceMesh dm;
        memset(&dm, 0, sizeof(dm));
        dm.nverts = nverts; dm.nfaces = nfaces; dm.alive = true;
        if (cudaMalloc(&dm.d_verts, (size_t)nverts * 3 * sizeof(float)) != cudaSuccess)
            return -1;
        if (cudaMalloc(&dm.d_faces, (size_t)nfaces * 3 * sizeof(int)) != cudaSuccess) {
            cudaFree(dm.d_verts); return -1;
        }
        cudaMemcpy(dm.d_verts, verts, (size_t)nverts * 3 * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dm.d_faces, faces, (size_t)nfaces * 3 * sizeof(int),
                   cudaMemcpyHostToDevice);

        // per-corner normals and uvs, stored unindexed: avoids separate
        // vt/vn index arrays and matches how the OBJ addresses them
        if (cnorms) {
            cudaMalloc(&dm.d_norms, (size_t)nfaces * 9 * sizeof(float));
            cudaMemcpy(dm.d_norms, cnorms, (size_t)nfaces * 9 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
        if (cuvs) {
            cudaMalloc(&dm.d_uvs, (size_t)nfaces * 6 * sizeof(float));
            cudaMemcpy(dm.d_uvs, cuvs, (size_t)nfaces * 6 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        meshes().push_back(dm);
        return (int)meshes().size() - 1;
    }

    // meshes still holding device memory. slots are never reused, so this is
    // a count of the live ones, not of meshes().size().
    int liveMeshCount() {
        int n = 0;
        for (size_t i = 0; i < meshes().size(); i++)
            if (meshes()[i].alive) n++;
        return n;
    }

    void destroyMesh(int h) {
        if (h < 0 || h >= (int)meshes().size()) return;
        DeviceMesh& dm = meshes()[h];
        if (!dm.alive) return;
        cudaFree(dm.d_verts); cudaFree(dm.d_faces);
        if (dm.d_norms) cudaFree(dm.d_norms);
        if (dm.d_uvs)   cudaFree(dm.d_uvs);
        for (int i = 0; i < 3; i++) {
            if (dm.has_tex[i]) {
                cudaDestroyTextureObject(dm.tex[i]);
                cudaFreeArray(dm.tex_arr[i]);
            }
        }
        dm.alive = false;
    }

    bool setEnvironment(const float* rgba, int w, int h) {
        if (!initialized) return false;
        clearEnvironment();
        has_env = uploadFloatTexture(rgba, w, h, &d_env_arr, &d_env_tex);
        if (has_env) buildIrradiance(w, h);
        return has_env;
    }

    // One convolution per environment load. The result is small enough that
    // the round trip through host memory costs less than it would take to
    // write a device-to-array path for it.
    void buildIrradiance(int env_w, int env_h) {
        const int cw = cudaIrradianceConvWidth();
        const int ch = cudaIrradianceConvHeight();
        const int iw = IRRADIANCE_W, ih = IRRADIANCE_H;

        float4 *d_conv = NULL, *d_irr = NULL;
        if (cudaMalloc(&d_conv, (size_t)cw * ch * sizeof(float4)) != cudaSuccess)
            return;
        if (cudaMalloc(&d_irr, (size_t)iw * ih * sizeof(float4)) != cudaSuccess) {
            cudaFree(d_conv);
            return;
        }

        cudaLaunchEnvReduce(d_env_tex, d_conv, cw, ch, env_w, env_h);
        cudaLaunchIrradiance(d_conv, cw, ch, d_irr, iw, ih);

        std::vector<float> host((size_t)iw * ih * 4);
        cudaMemcpy(host.data(), d_irr, host.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_conv);
        cudaFree(d_irr);

        has_irr = uploadFloatTexture(host.data(), iw, ih, &d_irr_arr, &d_irr_tex);
    }

    void clearEnvironment() {
        if (d_env_tex) cudaDestroyTextureObject(d_env_tex);
        if (d_env_arr) cudaFreeArray(d_env_arr);
        d_env_tex = 0;
        d_env_arr = NULL;
        has_env = false;

        if (d_irr_tex) cudaDestroyTextureObject(d_irr_tex);
        if (d_irr_arr) cudaFreeArray(d_irr_arr);
        d_irr_tex = 0;
        d_irr_arr = NULL;
        has_irr = false;
    }

    // eye -> world rotation for this frame, and how much of the environment's
    // light to admit. Both are frame state, sent alongside the view.
    void setIBL(const float* e2w9, float intensity) {
        if (e2w9)
            for (int i = 0; i < 9; i++) ibl_e2w[i] = e2w9[i];
        ibl_intensity = intensity > 0.f ? intensity : 0.f;
    }

    void setEnvironmentView(const float* inv16) {
        if (inv16)
            for (int i = 0; i < 16; i++) env_inv_vp.m[i] = inv16[i];
    }

    void setMeshTexture(int h, int slot, const unsigned char* px,
                        int w, int hgt, int bpp) {
        if (h < 0 || h >= (int)meshes().size() || slot < 0 || slot > 2) return;
        DeviceMesh& dm = meshes()[h];
        if (!dm.alive || dm.has_tex[slot]) return;
        dm.has_tex[slot] = uploadTexture(px, w, hgt, bpp,
                                         &dm.tex_arr[slot], &dm.tex[slot]);
    }

    void renderShadowPass() { flush(true); }

    void shadowStats(int* nonzero, float* mn, float* mx) {
        if (nonzero) *nonzero = 0;
        if (mn) *mn = 0.f;
        if (mx) *mx = 0.f;
        if (!initialized) return;
        std::vector<int> h((size_t)width * height);
        cudaMemcpy(h.data(), d_shadowbuf, h.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        int nz = 0; float lo = 1e30f, hi = -1e30f;
        for (size_t i = 0; i < h.size(); i++) {
            if (h[i] == 0) continue;
            nz++;
            union { int i; float f; } c; c.i = h[i];
            if (c.f < lo) lo = c.f;
            if (c.f > hi) hi = c.f;
        }
        if (nonzero) *nonzero = nz;
        if (nz) { if (mn) *mn = lo; if (mx) *mx = hi; }
    }

    void drawMesh(int h, const float* mvp16, const float* clip16,
                  const float* mit16,
                  const float* light3, const float* light_rgb3, float light_intensity,
                  const float* mshadow16, float shadow_bias,
                  unsigned char r, unsigned char g, unsigned char b) {
        if (!initialized || h < 0 || h >= (int)meshes().size()) return;
        DeviceMesh& dm = meshes()[h];
        if (!dm.alive || dm.nfaces == 0) return;

        // the raster kernel draws every mesh in one launch, so per-mesh state
        // (textures, the eye-space light) goes into a table the triangle indexes
        CudaMaterial m;
        memset(&m, 0, sizeof(m));
        for (int i = 0; i < 3; i++) {
            if (dm.has_tex[i]) {
                if (i == 0) { m.diffuse = dm.tex[0]; m.has_diffuse = 1; }
                if (i == 1) { m.nm      = dm.tex[1]; m.has_nm      = 1; }
                if (i == 2) { m.spec    = dm.tex[2]; m.has_spec    = 1; }
            }
        }
        m.lx = light3[0]; m.ly = light3[1]; m.lz = light3[2];
        m.lcr = light_rgb3 ? fmaxf(light_rgb3[0], 0.0f) : 1.0f;
        m.lcg = light_rgb3 ? fmaxf(light_rgb3[1], 0.0f) : 1.0f;
        m.lcb = light_rgb3 ? fmaxf(light_rgb3[2], 0.0f) : 1.0f;
        m.lintensity = fmaxf(light_intensity, 0.0f);
        if (mshadow16) {
            m.has_shadow = 1;
            m.shadow_bias = shadow_bias;
            m.debug_shadow = (shadow_bias < 0.0f) ? 1 : 0;   // negative bias = debug
            for (int i = 0; i < 16; i++) m.mshadow[i] = mshadow16[i];
        }
        for (int r2 = 0; r2 < 3; r2++)
            for (int c2 = 0; c2 < 3; c2++)
                m.mit[r2 * 3 + c2] = mit16[r2 * 4 + c2];

        if (has_irr) {
            m.irradiance = d_irr_tex;
            m.has_irradiance = 1;
            m.ibl_intensity = ibl_intensity;
            for (int i = 0; i < 9; i++) m.e2w[i] = ibl_e2w[i];
        }
        // staged, not uploaded: flush() sends the whole table in one memcpy
        h_materials.push_back(m);
        int mat_id = (int)h_materials.size() - 1;

        MeshDraw dr;
        dr.verts = dm.d_verts;
        dr.faces = dm.d_faces;
        dr.cnorms = dm.d_norms;
        dr.cuvs = dm.d_uvs;
        dr.nfaces = dm.nfaces;
        dr.face_begin = h_draw_faces;
        for (int i = 0; i < 16; i++) dr.mvp.m[i] = mvp16[i];
        for (int i = 0; i < 16; i++) dr.clip.m[i] = clip16[i];
        for (int i = 0; i < 16; i++) dr.mit.m[i] = mit16[i];
        dr.mat_id = mat_id;
        dr.color = CudaColor(r, g, b);

        // queued, not launched: flush() transforms the whole scene at once
        h_draws.push_back(dr);
        h_draw_faces += dm.nfaces;
    }

    int shadeCount() {
        if (!initialized) return 0;
        int n = 0;
        cudaMemcpy(&n, d_stats + 4, sizeof(int), cudaMemcpyDeviceToHost);
        return n;
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

// C interface, matching the signatures Engine.cpp calls
extern "C" {
    // ss is the supersampling factor: the frame is rendered at
    // (width*ss) x (height*ss) and box-filtered back down on the device.
    // Existing callers declare the two-argument form, so that stays as-is and
    // means ss = 1.
    bool initCudaRasterizerSS(int width, int height, int ss) {
        if (ss < 1) ss = 1;
        if (g_cuda_rasterizer) delete g_cuda_rasterizer;

        g_cuda_rasterizer = new CudaTriangleRasterizer(width, height, ss);
        return g_cuda_rasterizer->isInitialized();
    }

    bool initCudaRasterizer(int width, int height) {
        return initCudaRasterizerSS(width, height, 1);
    }

    void cleanupCudaRasterizer() {
        if (g_cuda_rasterizer) {
            delete g_cuda_rasterizer;
            g_cuda_rasterizer = nullptr;
        }
    }

    void cudaSetLinearTextureFiltering(int enabled) {
        g_use_linear_filter = enabled != 0;
    }

    // 1 = visibility buffer then one shade per pixel, 0 = shade inside the
    // depth loop. Takes effect on the next flush; nothing is reallocated.
    void cudaSetDeferredShading(int enabled) {
        g_deferred_shading = enabled != 0;
    }

    int cudaGetDeferredShading() {
        return g_deferred_shading ? 1 : 0;
    }

    // Fragment-shader invocations since the last clear. Forward counts one per
    // fragment that won the depth test; deferred one per covered pixel.
    int cudaGetShadeCount() {
        if (!g_cuda_rasterizer) return 0;
        return g_cuda_rasterizer->shadeCount();
    }

    void cudaClearBuffers() {
        if (g_cuda_rasterizer) {
            g_cuda_rasterizer->clear();
        }
    }

    // called per-triangle from Engine.cpp. queues only, no GPU work yet.
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

    // ---- persistent geometry: upload once, transform on the GPU each frame
    int cudaCreateMesh(const float* verts, int nverts, const int* faces, int nfaces,
                       const float* corner_normals, const float* corner_uvs) {
        return g_cuda_rasterizer
             ? g_cuda_rasterizer->createMesh(verts, nverts, faces, nfaces,
                                             corner_normals, corner_uvs) : -1;
    }
    // slot: 0 diffuse, 1 normal map, 2 specular. px is a TGAImage buffer.
    void cudaSetMeshTexture(int handle, int slot, const unsigned char* px,
                            int w, int h, int bpp) {
        if (g_cuda_rasterizer)
            g_cuda_rasterizer->setMeshTexture(handle, slot, px, w, h, bpp);
    }
    void cudaSetEnvironment(const float* rgba, int w, int h) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->setEnvironment(rgba, w, h);
    }
    void cudaClearEnvironment() {
        if (g_cuda_rasterizer) g_cuda_rasterizer->clearEnvironment();
    }
    void cudaSetEnvironmentView(const float* inv16) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->setEnvironmentView(inv16);
    }
    void cudaSetIBL(const float* e2w9, float intensity) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->setIBL(e2w9, intensity);
    }
    void cudaSetExposure(float exposure) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->setExposure(exposure);
    }
    void cudaSetToneMapping(int enabled) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->setToneMapping(enabled != 0);
    }
    int cudaLiveMeshCount() {
        return g_cuda_rasterizer ? g_cuda_rasterizer->liveMeshCount() : 0;
    }
    void cudaDestroyMesh(int handle) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->destroyMesh(handle);
    }
    // mshadow16 may be NULL for an unshadowed draw
    void cudaDrawMesh(int handle, const float* mvp16, const float* clip16,
                      const float* mit16,
                      const float* light3, const float* light_rgb3,
                      float light_intensity, const float* mshadow16,
                      float shadow_bias,
                      unsigned char r, unsigned char g, unsigned char b) {
        if (g_cuda_rasterizer)
            g_cuda_rasterizer->drawMesh(handle, mvp16, clip16, mit16, light3,
                                        light_rgb3, light_intensity,
                                        mshadow16, shadow_bias, r, g, b);
    }
    void cudaGetShadowStats(int* nonzero, float* mn, float* mx) {
        if (g_cuda_rasterizer) g_cuda_rasterizer->shadowStats(nonzero, mn, mx);
    }
    // rasterizes everything queued so far into the shadow depth buffer
    void cudaRenderShadowPass() {
        if (g_cuda_rasterizer) g_cuda_rasterizer->renderShadowPass();
    }

    void cudaGetKernelTimings(float* upload_ms, float* bin_ms, float* raster_ms) {
        if (g_cuda_rasterizer)
            g_cuda_rasterizer->getKernelTimings(upload_ms, bin_ms, raster_ms);
    }

    void cudaGetRasterStats(int* submitted, int* culled_back, int* culled_offscreen,
                            int* bin_entries) {
        if (g_cuda_rasterizer) {
            g_cuda_rasterizer->getStats(submitted, culled_back, culled_offscreen,
                                        bin_entries);
        }
    }

    // Screen space ambient occlusion over the finished frame. inv_vp16 and
    // vp16 are inverse(Viewport*Projection) and Viewport*Projection, row-major.
    // radius is in world units, intensity 0 disables the effect.
    void cudaApplySSAO(const float* inv_vp16, const float* vp16,
                       float radius, float intensity, float bias, int debug) {
        if (g_cuda_rasterizer)
            g_cuda_rasterizer->applySSAO(inv_vp16, vp16, radius, intensity, bias, debug);
    }

    // The finished frame, still on the device. Pending kernels are flushed
    // first so anything that composites into it lands on top of a complete
    // raster frame. Both modules launch on the default stream, so a kernel
    // queued after this call is ordered after the raster work.
    unsigned char* cudaGetDeviceFramebuffer(int* w, int* h) {
        if (!g_cuda_rasterizer) {
            if (w) *w = 0;
            if (h) *h = 0;
            return NULL;
        }
        return g_cuda_rasterizer->deviceFramebuffer(w, h);
    }

    void cudaBlitToTexture(void* dst, int dst_pitch) {
        if (g_cuda_rasterizer) {
            g_cuda_rasterizer->blitToTexture(dst, dst_pitch);
        }
    }

    // pull the frame back into the host TGAImage so host-side code can
    // composite onto it. TGAImage is bottom-up and stores B,G,R while the
    // device buffer is top-down R,G,B, hence the row flip.
    void cudaCopyResults(TGAImage& framebuffer) {
        if (!g_cuda_rasterizer) return;

        int width = framebuffer.get_width();
        int height = framebuffer.get_height();

        unsigned char* h_framebuffer = new unsigned char[width * height * 3];
        g_cuda_rasterizer->copyToCPU(h_framebuffer);

        for (int y = 0; y < height; y++) {
            const unsigned char* row = h_framebuffer + (size_t)(height - 1 - y) * width * 3;
            for (int x = 0; x < width; x++) {
                TGAColor color(row[x * 3 + 0], row[x * 3 + 1], row[x * 3 + 2]);
                framebuffer.set(x, y, color);
            }
        }

        delete[] h_framebuffer;
    }
}
