#ifndef __GLPRESENTER_H__
#define __GLPRESENTER_H__

#include <SDL2/SDL.h>

// Gets the finished frame onto the screen, and exists mainly so the two ways
// of doing that can be compared.
//
// Both paths end identically: the same GL texture, drawn on the same quad, in
// the same context, swapped the same way. They differ only in how the pixels
// reach that texture --
//
//   HostCopy: device -> host staging buffer -> glTexSubImage2D
//   Interop:  device -> registered PBO (device to device) -> glTexSubImage2D
//
// Keeping everything downstream of the upload identical is the point. The
// earlier version of this comparison would have been the interop path against
// SDL_Renderer, which differs in the driver path, the texture format and the
// swap, and would have credited interop for all of it.
class GLPresenter
{
public:
    enum Mode
    {
        HostCopy = 0,   // the pre-interop path: pixels cross the bus twice
        Interop = 1     // pixels never leave the device
    };

    GLPresenter();
    ~GLPresenter();

    // Needs a window created with SDL_WINDOW_OPENGL. Returns false when a GL
    // context or the entry points cannot be had, which is the headless case
    // (SDL's dummy video driver) and is not an error worth failing init over.
    bool create(SDL_Window *window, int frameWidth, int frameHeight);
    void destroy();

    bool isValid() const { return context != nullptr; }

    // Whether Interop is actually usable. False under a software GL, where
    // CUDA cannot share a device with the context.
    bool interopAvailable() const { return interop_ok; }

    // Silently stays in HostCopy if interop is unavailable, so a toggle in the
    // UI cannot put the presenter into a mode that does not work.
    void setMode(Mode m);
    Mode getMode() const { return mode; }
    const char *modeName() const;

    // The render target changed size: reallocate the texture and PBO, and
    // re-register the PBO with CUDA.
    bool resize(int frameWidth, int frameHeight);

    // One frame. Uploads by the current mode, then draws the quad letterboxed
    // into the window and swaps.
    void present();

    // Milliseconds spent getting the frame into the texture, which is the only
    // part the two modes do differently. Excludes the quad and the swap.
    float lastUploadMs() const { return last_upload_ms; }

private:
    bool loadEntryPoints();
    bool allocateTargets(int w, int h);
    void uploadHostCopy();
    void uploadInterop();
    void drawQuad();

    SDL_Window *window;
    SDL_GLContext context;

    unsigned int texture;   // GLuint
    unsigned int pbo;       // GLuint, 0 when interop is unavailable

    int width, height;      // frame (render target) size, not window size

    bool interop_ok;
    Mode mode;
    float last_upload_ms;

    // Staging for HostCopy. Held rather than allocated per frame so the
    // comparison measures the transfer and not an allocator.
    unsigned char *staging;
    size_t staging_bytes;
};

#endif // __GLPRESENTER_H__
