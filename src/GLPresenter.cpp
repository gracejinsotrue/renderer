#include "GLPresenter.h"

#include <SDL2/SDL_opengl.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>

extern "C"
{
    // rasterizer.cu: the frame, device -> host, one DMA
    void cudaBlitToTexture(void *dst, int dst_pitch);
    // present.cu: the frame, device -> registered PBO, no host involvement
    int cudaGLInteropSupported();
    int cudaGLRegisterPBO(unsigned int pbo);
    void cudaGLUnregisterPBO();
    size_t cudaGLBlitToPBO();
}

// Buffer objects are GL 1.5, and Windows' opengl32 exports only 1.1, so these
// have to come from the driver at run time. The texture and quad calls below
// are all 1.1 and link directly.
static PFNGLGENBUFFERSPROC    p_glGenBuffers = nullptr;
static PFNGLBINDBUFFERPROC    p_glBindBuffer = nullptr;
static PFNGLBUFFERDATAPROC    p_glBufferData = nullptr;
static PFNGLDELETEBUFFERSPROC p_glDeleteBuffers = nullptr;

using Clock = std::chrono::high_resolution_clock;
static float msSince(Clock::time_point a)
{
    return std::chrono::duration<float, std::milli>(Clock::now() - a).count();
}

GLPresenter::GLPresenter()
    : window(nullptr), context(nullptr), texture(0), pbo(0),
      width(0), height(0), interop_ok(false), mode(HostCopy),
      last_upload_ms(0.f), staging(nullptr), staging_bytes(0)
{
}

GLPresenter::~GLPresenter()
{
    destroy();
}

bool GLPresenter::loadEntryPoints()
{
    p_glGenBuffers = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    p_glBindBuffer = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    p_glBufferData = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    p_glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteBuffers");

    return p_glGenBuffers && p_glBindBuffer && p_glBufferData && p_glDeleteBuffers;
}

bool GLPresenter::create(SDL_Window *win, int frameWidth, int frameHeight)
{
    window = win;

    context = SDL_GL_CreateContext(window);
    if (!context)
    {
        // Expected headlessly: the dummy video driver has no GL at all.
        printf("no GL context (%s); falling back to the SDL renderer\n",
               SDL_GetError());
        return false;
    }

    if (SDL_GL_MakeCurrent(window, context) != 0)
    {
        printf("SDL_GL_MakeCurrent failed: %s\n", SDL_GetError());
        destroy();
        return false;
    }

    // Present is paced by the engine loop, not the display. Leaving vsync on
    // would clamp every measurement below to the refresh rate and make the
    // two upload paths look identical.
    SDL_GL_SetSwapInterval(0);

    const char *renderer = (const char *)glGetString(GL_RENDERER);
    printf("GL renderer: %s\n", renderer ? renderer : "(unknown)");

    if (!loadEntryPoints())
    {
        printf("GL buffer objects unavailable; interop path disabled\n");
    }

    // Ask CUDA whether it can share a device with this context. Under a
    // software GL this is where the answer comes back no.
    interop_ok = p_glGenBuffers && cudaGLInteropSupported() != 0;

    if (!allocateTargets(frameWidth, frameHeight))
    {
        destroy();
        return false;
    }

    mode = interop_ok ? Interop : HostCopy;

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);

    return true;
}

bool GLPresenter::allocateTargets(int w, int h)
{
    if (w <= 0 || h <= 0) return false;

    width = w;
    height = h;

    if (!texture) glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // The frame is already at the render resolution and gets scaled to the
    // window by the quad, so linear filtering here is the only smoothing.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // RGB8, matching the device buffer's packed top-down R,G,B exactly, so
    // neither path does any per-pixel conversion.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    // Rows are tightly packed, not padded to 4 bytes, and at an odd width a
    // 3-byte format is not 4-byte aligned. Without this the driver reads each
    // row at the wrong offset and the image shears.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    size_t frame_bytes = (size_t)width * height * 3;

    if (staging_bytes < frame_bytes)
    {
        delete[] staging;
        staging = new unsigned char[frame_bytes];
        staging_bytes = frame_bytes;
    }

    if (interop_ok)
    {
        cudaGLUnregisterPBO();
        if (!pbo) p_glGenBuffers(1, &pbo);

        p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        p_glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)frame_bytes,
                       nullptr, GL_STREAM_DRAW);
        p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        if (!cudaGLRegisterPBO(pbo))
        {
            printf("PBO registration failed; interop path disabled\n");
            interop_ok = false;
            mode = HostCopy;
        }
    }

    return true;
}

bool GLPresenter::resize(int frameWidth, int frameHeight)
{
    if (!context) return false;
    if (frameWidth == width && frameHeight == height) return true;
    return allocateTargets(frameWidth, frameHeight);
}

void GLPresenter::destroy()
{
    if (interop_ok) cudaGLUnregisterPBO();

    if (pbo && p_glDeleteBuffers)
    {
        p_glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (texture)
    {
        glDeleteTextures(1, &texture);
        texture = 0;
    }

    delete[] staging;
    staging = nullptr;
    staging_bytes = 0;

    if (context)
    {
        SDL_GL_DeleteContext(context);
        context = nullptr;
    }
}

void GLPresenter::setMode(Mode m)
{
    if (m == Interop && !interop_ok) return;
    mode = m;
}

const char *GLPresenter::modeName() const
{
    return mode == Interop ? "interop" : "host copy";
}

// The frame crosses the bus into host memory, then crosses back as a texture
// upload. This is what the engine did before interop existed.
void GLPresenter::uploadHostCopy()
{
    cudaBlitToTexture(staging, width * 3);

    p_glBindBuffer ? p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) : (void)0;

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGB, GL_UNSIGNED_BYTE, staging);
}

// The frame is copied device-to-device into the PBO, and the texture is fed
// from that buffer. The host never sees a pixel: the last argument is an
// offset into the bound buffer, not a pointer.
void GLPresenter::uploadInterop()
{
    if (cudaGLBlitToPBO() == 0)
    {
        // Rather than present a stale frame, fall back for this frame. If it
        // keeps failing the mode is wrong and should stop claiming otherwise.
        uploadHostCopy();
        return;
    }

    p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GLPresenter::drawQuad()
{
    int winW = 0, winH = 0;
    SDL_GL_GetDrawableSize(window, &winW, &winH);
    if (winW <= 0 || winH <= 0) return;

    glViewport(0, 0, winW, winH);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Letterbox: preserve the frame's aspect inside the window, matching what
    // the SDL_Renderer path did with its destination rect.
    float scale = std::min((float)winW / width, (float)winH / height);
    float qw = (width * scale) / winW;    // as a fraction of the window
    float qh = (height * scale) / winH;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBindTexture(GL_TEXTURE_2D, texture);

    // The frame is stored top-down, and GL's texture origin is bottom-left, so
    // v is flipped here rather than the image being flipped on the device.
    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 1.f); glVertex2f(-qw, -qh);
    glTexCoord2f(1.f, 1.f); glVertex2f(qw, -qh);
    glTexCoord2f(1.f, 0.f); glVertex2f(qw, qh);
    glTexCoord2f(0.f, 0.f); glVertex2f(-qw, qh);
    glEnd();
}

void GLPresenter::present()
{
    if (!context) return;

    auto t0 = Clock::now();
    if (mode == Interop) uploadInterop();
    else uploadHostCopy();
    last_upload_ms = msSince(t0);

    drawQuad();
    SDL_GL_SwapWindow(window);
}
