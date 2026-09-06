// Presenting the finished frame without sending it through the host.
//
// The rest of the pipeline already leaves the frame resolved and in device
// memory. Every path before this one then paid to drag it back across the bus
// so SDL could upload it again: device -> host -> GPU texture, for pixels that
// never left the card conceptually. Registering an OpenGL pixel buffer with
// CUDA lets the copy stay device-to-device, and the driver then feeds the
// texture from that buffer without the host touching a byte.
//
// This is the path that was impossible under WSL: WSLg's OpenGL is llvmpipe, a
// CPU software rasterizer, so cudaGLGetDevices reports no devices and
// cudaGraphicsGLRegisterBuffer cannot succeed. It needs a native build.
//
// Note this file is NOT a pipeline stage -- it takes the finished frame rather
// than producing part of one -- so it does not live behind stages.cuh.

#ifdef _WIN32
// GL/gl.h on Windows needs the windows.h typedefs (APIENTRY, WINGDIAPI) first,
// and windows.h without these two drags in a great deal and defines min/max.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

// the frame, still on the device, resolved and flushed. owned by rasterizer.cu.
extern "C" unsigned char *cudaGetDeviceFramebuffer(int *w, int *h);

// The registered buffer. One is enough: the presenter owns a single PBO for
// the life of the window and re-registers it on resize.
static cudaGraphicsResource *g_pbo_resource = NULL;
static unsigned int g_registered_pbo = 0;

// Set once by cudaGLInteropSupported so the probe cost is paid a single time.
// -1 not yet probed, 0 unavailable, 1 available.
static int g_interop_probe = -1;

extern "C" {

// Declared ahead of use: cudaGLRegisterPBO drops a previous registration
// before making a new one, and is defined above the function that does it.
void cudaGLUnregisterPBO();

// Does the GL context that is current on THIS thread belong to a device CUDA
// can share with? Must be called with a context current: cudaGLGetDevices
// queries the driver through it, which is exactly why it returns 0 devices
// under a software GL like llvmpipe.
int cudaGLInteropSupported()
{
    if (g_interop_probe >= 0) return g_interop_probe;

    unsigned int gl_device_count = 0;
    int gl_devices[8];
    cudaError_t err = cudaGLGetDevices(&gl_device_count, gl_devices, 8,
                                       cudaGLDeviceListAll);

    if (err != cudaSuccess || gl_device_count == 0) {
        // Not a failure worth aborting over: the caller falls back to the
        // copy path. Clear the sticky error so the next real CUDA call is
        // not blamed for it.
        cudaGetLastError();
        printf("CUDA/GL interop unavailable (%s, %u devices); "
               "presenting via host copy\n",
               cudaGetErrorString(err), gl_device_count);
        g_interop_probe = 0;
        return 0;
    }

    g_interop_probe = 1;
    return 1;
}

// Register a GL pixel-unpack buffer. WriteDiscard says the previous contents
// are never read back, which lets the driver skip preserving them.
int cudaGLRegisterPBO(unsigned int pbo)
{
    if (g_pbo_resource) {
        if (g_registered_pbo == pbo) return 1;   // already registered
        cudaGLUnregisterPBO();
    }

    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &g_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    if (err != cudaSuccess) {
        printf("cudaGraphicsGLRegisterBuffer failed: %s\n",
               cudaGetErrorString(err));
        g_pbo_resource = NULL;
        g_registered_pbo = 0;
        return 0;
    }

    g_registered_pbo = pbo;
    return 1;
}

void cudaGLUnregisterPBO()
{
    if (!g_pbo_resource) return;
    cudaGraphicsUnregisterResource(g_pbo_resource);
    g_pbo_resource = NULL;
    g_registered_pbo = 0;
}

// Copy the finished frame into the registered PBO, device to device.
//
// cudaGetDeviceFramebuffer flushes the pipeline and resolves supersampling
// first, so what comes back is the same bytes the host-copy path would have
// received -- the two present paths differ only in where those bytes go, which
// is what makes them comparable.
//
// Returns the number of bytes copied, or 0 on failure so the caller can fall
// back rather than present a stale frame.
size_t cudaGLBlitToPBO()
{
    if (!g_pbo_resource) return 0;

    int w = 0, h = 0;
    unsigned char *src = cudaGetDeviceFramebuffer(&w, &h);
    if (!src || w <= 0 || h <= 0) return 0;

    size_t frame_bytes = (size_t)w * h * 3;

    cudaError_t err = cudaGraphicsMapResources(1, &g_pbo_resource, 0);
    if (err != cudaSuccess) {
        printf("cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    void *dst = NULL;
    size_t dst_bytes = 0;
    err = cudaGraphicsResourceGetMappedPointer(&dst, &dst_bytes, g_pbo_resource);

    if (err != cudaSuccess || !dst) {
        printf("cudaGraphicsResourceGetMappedPointer failed: %s\n",
               cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, &g_pbo_resource, 0);
        return 0;
    }

    // The PBO is allocated by the presenter at exactly this size. If it is
    // ever smaller the copy would run off the end of a driver allocation, so
    // check rather than trust: a resize that misses a re-register lands here.
    if (dst_bytes < frame_bytes) {
        printf("PBO too small: %zu bytes for a %dx%d frame needing %zu\n",
               dst_bytes, w, h, frame_bytes);
        cudaGraphicsUnmapResources(1, &g_pbo_resource, 0);
        return 0;
    }

    // Both buffers are tightly packed top-down RGB, so this is one flat copy
    // that never leaves the device.
    err = cudaMemcpy(dst, src, frame_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("device-to-device frame copy failed: %s\n",
               cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, &g_pbo_resource, 0);
        return 0;
    }

    // Unmapping is what orders the copy against GL's later read of the buffer.
    err = cudaGraphicsUnmapResources(1, &g_pbo_resource, 0);
    if (err != cudaSuccess) {
        printf("cudaGraphicsUnmapResources failed: %s\n",
               cudaGetErrorString(err));
        return 0;
    }

    return frame_bytes;
}

}  // extern "C"
