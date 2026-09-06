// The two stages that touch the finished frame rather than geometry: the
// supersample resolve, and the background the frame is cleared to.
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
