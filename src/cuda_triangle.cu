// cuda_triangle.cu
//CUDA-compatible version
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// forward declare what we need from the rasterizer
class TGAImage;
class TGAColor;
template<size_t DIM, typename T> struct vec;
typedef vec<3, float> Vec3f;
typedef vec<4, float> Vec4f;

// CUDA-compatible 3D vector
struct CudaVec3 {
    float x, y, z;
    
    __device__ __host__ CudaVec3() : x(0), y(0), z(0) {}
    __device__ __host__ CudaVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

//  4D vector  
struct CudaVec4 {
    float x, y, z, w;
    
    __device__ __host__ CudaVec4() : x(0), y(0), z(0), w(1) {}
    __device__ __host__ CudaVec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

// simple color structure
struct CudaColor {
    unsigned char r, g, b;
    
    __device__ __host__ CudaColor() : r(0), g(0), b(0) {}
    __device__ __host__ CudaColor(unsigned char r_, unsigned char g_, unsigned char b_) 
        : r(r_), g(g_), b(b_) {}
};

// GPU barycentric calculation - returns simple struct
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
        return CudaVec3(-1, 1, 1);  // Degenerate triangle
    }
    
    float u = (s1y * s2x - s1x * s2y) / cross_z;
    float v = (s0x * s2y - s0y * s2x) / cross_z;
    
    return CudaVec3(1.0f - u - v, v, u);
}

// CUDA kernel for triangle rasterization
__global__
void triangle_raster_kernel(CudaVec4* vertices,
                           unsigned char* framebuffer,
                           float* zbuffer,
                           int width, int height,
                           CudaColor color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Perspective divide
    float v0x = vertices[0].x / vertices[0].w;
    float v0y = vertices[0].y / vertices[0].w;
    float v0z = vertices[0].z;
    
    float v1x = vertices[1].x / vertices[1].w;
    float v1y = vertices[1].y / vertices[1].w;
    float v1z = vertices[1].z;
    
    float v2x = vertices[2].x / vertices[2].w;
    float v2y = vertices[2].y / vertices[2].w;
    float v2z = vertices[2].z;
    
    // Barycentric coordinates
    CudaVec3 bary = cuda_barycentric(v0x, v0y, v1x, v1y, v2x, v2y, (float)x, (float)y);
    
    if (bary.x < 0 || bary.y < 0 || bary.z < 0) return;
    
    // Depth interpolation
    float z = v0z * bary.x + v1z * bary.y + v2z * bary.z;
    float w = vertices[0].w * bary.x + vertices[1].w * bary.y + vertices[2].w * bary.z;
    float depth = z / w;
    
    int idx = y * width + x;
    
    // Simple depth test
    if (depth > zbuffer[idx]) return;
    
    zbuffer[idx] = depth;
    framebuffer[idx * 3 + 0] = color.r;
    framebuffer[idx * 3 + 1] = color.g;
    framebuffer[idx * 3 + 2] = color.b;
}

// C++ wrapper class (host-only)
class CudaTriangleRasterizer {
private:
    CudaVec4* d_vertices;
    unsigned char* d_framebuffer;
    float* d_zbuffer;
    int width, height;
    bool initialized;
    
public:
    CudaTriangleRasterizer(int w, int h) : width(w), height(h), initialized(false) {
        cudaError_t err;
        
        err = cudaMalloc(&d_vertices, 3 * sizeof(CudaVec4));
        if (err != cudaSuccess) {
            printf("CUDA malloc vertices failed: %s\n", cudaGetErrorString(err));
            return;
        }
        
        err = cudaMalloc(&d_framebuffer, width * height * 3);
        if (err != cudaSuccess) {
            printf("CUDA malloc framebuffer failed: %s\n", cudaGetErrorString(err));
            return;
        }
        
        err = cudaMalloc(&d_zbuffer, width * height * sizeof(float));
        if (err != cudaSuccess) {
            printf("CUDA malloc zbuffer failed: %s\n", cudaGetErrorString(err));
            return;
        }
        
        initialized = true;
        printf("CUDA rasterizer initialized: %dx%d\n", width, height);
    }
    
    ~CudaTriangleRasterizer() {
        if (initialized) {
            cudaFree(d_vertices);
            cudaFree(d_framebuffer);
            cudaFree(d_zbuffer);
        }
    }
    
    bool isInitialized() const { return initialized; }
    
    void clear() {
        if (!initialized) return;
        
        cudaMemset(d_framebuffer, 0, width * height * 3);
        
        // Initialize zbuffer to far plane
        float far_value = 1000.0f;
        float* h_init = new float[width * height];
        for (int i = 0; i < width * height; i++) {
            h_init[i] = far_value;
        }
        cudaMemcpy(d_zbuffer, h_init, width * height * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_init;
    }
    
    void renderTriangle(float v0x, float v0y, float v0z, float v0w,
                       float v1x, float v1y, float v1z, float v1w,
                       float v2x, float v2y, float v2z, float v2w,
                       unsigned char r, unsigned char g, unsigned char b) {
        if (!initialized) return;
        
        CudaVec4 vertices[3] = {
            CudaVec4(v0x, v0y, v0z, v0w),
            CudaVec4(v1x, v1y, v1z, v1w),
            CudaVec4(v2x, v2y, v2z, v2w)
        };
        CudaColor color(r, g, b);
        
        cudaMemcpy(d_vertices, vertices, 3 * sizeof(CudaVec4), cudaMemcpyHostToDevice);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        triangle_raster_kernel<<<gridSize, blockSize>>>(
            d_vertices, d_framebuffer, d_zbuffer, width, height, color
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        }
    }
    
    void copyToCPU(unsigned char* host_framebuffer, float* host_zbuffer) {
        if (!initialized) return;
        
        cudaMemcpy(host_framebuffer, d_framebuffer, width * height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_zbuffer, d_zbuffer, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    void synchronize() {
        if (initialized) {
            cudaDeviceSynchronize();
        }
    }
};

// global instance for Engine to use
static CudaTriangleRasterizer* g_cuda_rasterizer = nullptr;

// include the rasterizer headers here so we can use the types
#include "geometry.h"
#include "tgaimage.h"

// C interface functions for Engine.cpp
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
    
    void cudaRenderTriangle(const Vec4f& v0, const Vec4f& v1, const Vec4f& v2, const TGAColor& tga_color) {
        if (g_cuda_rasterizer) {
            // extract values from Vec4f and TGAColor
            TGAColor& color = const_cast<TGAColor&>(tga_color);
            
            g_cuda_rasterizer->renderTriangle(
                v0[0], v0[1], v0[2], v0[3],
                v1[0], v1[1], v1[2], v1[3], 
                v2[0], v2[1], v2[2], v2[3],
                color.bgra[2], color.bgra[1], color.bgra[0]  // BGR to RGB
            );
        }
    }
    
    void cudaCopyResults(TGAImage& framebuffer, TGAImage& zbuffer) {
        if (!g_cuda_rasterizer) return;
        
        int width = framebuffer.get_width();
        int height = framebuffer.get_height();
        
        unsigned char* h_framebuffer = new unsigned char[width * height * 3];
        float* h_zbuffer = new float[width * height];
        
        g_cuda_rasterizer->synchronize();
        g_cuda_rasterizer->copyToCPU(h_framebuffer, h_zbuffer);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                
                TGAColor color(h_framebuffer[idx * 3 + 0],   // R
                              h_framebuffer[idx * 3 + 1],   // G  
                              h_framebuffer[idx * 3 + 2]);  // B
                framebuffer.set(x, y, color);
                
                unsigned char depth = (unsigned char)(h_zbuffer[idx] / 10.0f);
                zbuffer.set(x, y, TGAColor(depth));
            }
        }
        
        delete[] h_framebuffer;
        delete[] h_zbuffer;
    }
}