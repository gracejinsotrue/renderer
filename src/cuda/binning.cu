// Binning stage: which triangles touch which tile. Two passes over the
// triangle list with a prefix sum between them, so every tile gets a slice
// sized to exactly what it holds and nothing can be dropped for want of room.
//
// This is the only translation unit that pulls in cub.
#include <cub/cub.cuh>

#include "common.cuh"
#include "stages.cuh"

// pass 1a: count how many tiles each triangle lands in. counting only, so
// there is no per-tile capacity that could drop geometry.
__global__
void count_kernel(const CudaTriangle* triangles, int num_triangles,
                  int* tile_counts, int tiles_x, int tiles_y)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    const CudaTriangle& tri = triangles[t];
    int tx0 = tri.bbox_min_x / TILE_W, tx1 = tri.bbox_max_x / TILE_W;
    int ty0 = tri.bbox_min_y / TILE_H, ty1 = tri.bbox_max_y / TILE_H;
    if (tx0 < 0) tx0 = 0;
    if (ty0 < 0) ty0 = 0;
    if (tx1 >= tiles_x) tx1 = tiles_x - 1;
    if (ty1 >= tiles_y) ty1 = tiles_y - 1;

    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++)
            atomicAdd(&tile_counts[ty * tiles_x + tx], 1);
}

// pass 1b: with an exclusive prefix sum of the counts in hand, every triangle
// writes into its tile's exact slice of one flat index buffer.
__global__
void scatter_kernel(const CudaTriangle* triangles, int num_triangles,
                    const int* tile_offsets, int* tile_cursor, int* tri_indices,
                    int tiles_x, int tiles_y)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    const CudaTriangle& tri = triangles[t];
    int tx0 = tri.bbox_min_x / TILE_W, tx1 = tri.bbox_max_x / TILE_W;
    int ty0 = tri.bbox_min_y / TILE_H, ty1 = tri.bbox_max_y / TILE_H;
    if (tx0 < 0) tx0 = 0;
    if (ty0 < 0) ty0 = 0;
    if (tx1 >= tiles_x) tx1 = tiles_x - 1;
    if (ty1 >= tiles_y) ty1 = tiles_y - 1;

    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++) {
            int tile = ty * tiles_x + tx;
            tri_indices[tile_offsets[tile] + atomicAdd(&tile_cursor[tile], 1)] = t;
        }
}

size_t cudaBinScanTempBytes(int num_tiles)
{
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(NULL, bytes, (int*)NULL, (int*)NULL, num_tiles);
    return bytes;
}

void cudaBinCount(const CudaTriangle* triangles, int num_triangles,
                  int* tile_counts, int tiles_x, int tiles_y)
{
    int block = 256;
    int grid = (num_triangles + block - 1) / block;
    count_kernel<<<grid, block>>>(triangles, num_triangles,
                                  tile_counts, tiles_x, tiles_y);
}

void cudaBinScan(void* temp, size_t temp_bytes, const int* tile_counts,
                 int* tile_offsets, int num_tiles)
{
    cub::DeviceScan::ExclusiveSum(temp, temp_bytes, tile_counts, tile_offsets,
                                  num_tiles);
}

void cudaBinScatter(const CudaTriangle* triangles, int num_triangles,
                    const int* tile_offsets, int* tile_cursor, int* tri_indices,
                    int tiles_x, int tiles_y)
{
    int block = 256;
    int grid = (num_triangles + block - 1) / block;
    scatter_kernel<<<grid, block>>>(triangles, num_triangles,
                                    tile_offsets, tile_cursor, tri_indices,
                                    tiles_x, tiles_y);
}
