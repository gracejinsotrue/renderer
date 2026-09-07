// Host entry points for the pipeline stages. Each launcher owns its own grid
// and block geometry, so the sizes stay with the kernel they belong to rather
// than at the call site.
#pragma once

#include "common.cuh"

// setup.cu: transform every queued mesh, cull, append survivors
void cudaLaunchMeshSetup(const MeshDraw* draws, int ndraws, int total_faces,
                         CudaTriangle* out, int* out_count, int max_out,
                         int width, int height, int* stats);

// binning.cu: count -> exclusive scan -> scatter, giving each tile its own
// exact slice of one flat index buffer. The scan's scratch space is sized by
// the first call and owned by the caller.
size_t cudaBinScanTempBytes(int num_tiles);
void cudaBinCount(const CudaTriangle* triangles, int num_triangles,
                  int* tile_counts, int tiles_x, int tiles_y);
void cudaBinScan(void* temp, size_t temp_bytes, const int* tile_counts,
                 int* tile_offsets, int num_tiles);
void cudaBinScatter(const CudaTriangle* triangles, int num_triangles,
                    const int* tile_offsets, int* tile_cursor, int* tri_indices,
                    int tiles_x, int tiles_y);

// raster.cu: depth clear, then one block per tile for either pass
void cudaLaunchZbufferFill(int* zbuffer, int fill_val, int count);
void cudaLaunchTiledRaster(const CudaTriangle* triangles, const int* tile_counts,
                           const int* tile_offsets, const int* tri_indices,
                           const CudaMaterial* materials,
                           float4* framebuffer, int* zbuffer,
                           const int* shadowbuf, float* normalbuf,
                           int width, int height, int tiles_x, int tiles_y,
                           int* stats);
void cudaLaunchShadowRaster(const CudaTriangle* triangles, const int* tile_counts,
                            const int* tile_offsets, const int* tri_indices,
                            int* shadowbuf,
                            int width, int height, int tiles_x, int tiles_y);

// raster.cu: the deferred front half. Same tile walk as the colour pass, but
// it records the winning triangle per pixel instead of shading it. Depth and
// the winner's identity move together in one 64-bit atomic.
void cudaLaunchVisibilityRaster(const CudaTriangle* triangles,
                                const int* tile_counts, const int* tile_offsets,
                                const int* tri_indices,
                                unsigned long long* visbuffer,
                                int width, int height, int tiles_x, int tiles_y);
void cudaLaunchVisbufferFill(unsigned long long* visbuffer, int count);

// shade.cu: the deferred back half. One thread per pixel, one shade per pixel,
// so cost follows screen area rather than depth complexity.
void cudaLaunchDeferredShade(const CudaTriangle* triangles,
                             const CudaMaterial* materials,
                             const unsigned long long* visbuffer,
                             float4* framebuffer, int* zbuffer,
                             const int* shadowbuf, float* normalbuf,
                             int width, int height, int* stats);

// ssao.cu: occlude the finished frame in place. ao and ao_blur are scratch.
void cudaLaunchSSAO(const int* zbuffer, const float* normalbuf,
                    float* ao, float* ao_blur, float4* framebuffer,
                    const float* inv_vp, const float* vp,
                    const float* kernel_samples,
                    float radius, float bias, float intensity,
                    int ssao_debug, int width, int height);

// resolve.cu: supersample resolve and background composite. Everything up to
// the tone map works in the linear float target; only cudaLaunchToneMap
// produces the 8-bit frame the presenter and the capture path read.
void cudaLaunchClearColour(float4* framebuffer, int count);
void cudaLaunchDownsample(const float4* src, float4* dst,
                          int out_w, int out_h, int ss);
void cudaLaunchBackground(float4* framebuffer, cudaTextureObject_t bg,
                          int width, int height);

// resolve.cu: the equirectangular environment, in place of the background.
// inv_vp is the row-major inverse of Viewport*Projection*ModelView for this
// frame, which is what turns a pixel back into a world-space view ray.
void cudaLaunchEnvironment(float4* framebuffer, cudaTextureObject_t env,
                           Mat4 inv_vp, int width, int height);

// resolve.cu: linear radiance -> the 8-bit frame. passthrough skips exposure
// and the curve for the SSAO debug views, which carry normals and occlusion
// rather than light and would be misreported by a tone curve.
void cudaLaunchToneMap(const float4* src, unsigned char* dst,
                       int width, int height, float exposure, int passthrough);
