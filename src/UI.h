#ifndef __UI_H__
#define __UI_H__

#include <SDL2/SDL.h>
#include <string>

class Engine;

// The control panel.
//
// Presentation only: everything it touches goes through Engine's public
// interface, so no renderer state lives here and nothing here is needed to
// render a frame. The headless tests never construct one, and init() fails
// cleanly when there is no backend, so the engine runs unchanged without it.
//
// Which backend depends on how the frame reaches the screen, matching the
// split GLPresenter already makes: OpenGL when there is a GL context, and
// SDL_Renderer when the presenter fell back.
class UI
{
public:
    UI();
    ~UI();

    // gl_context is the presenter's, or NULL to drive the SDL_Renderer path
    // instead. window has to outlive this.
    bool init(SDL_Window *window, SDL_GLContext gl_context, SDL_Renderer *renderer);
    void shutdown();
    bool isValid() const { return active; }

    void handleEvent(const SDL_Event &e);

    // True while the pointer or the keyboard belongs to a panel. The engine
    // has to skip its own handling then, or a drag on a slider also orbits
    // the camera and typing in a path field flies it around.
    bool wantsMouse() const;
    bool wantsKeyboard() const;

    // Builds this frame's widgets. Between update() and present().
    void build(Engine &engine);
    // Issues the draw calls, inside the present path: after the frame itself
    // and before the swap, so the panel sits over the render.
    void render();

private:
    void buildFrameStats(Engine &engine);
    void buildDisplay(Engine &engine);
    void buildLighting(Engine &engine);
    void buildScene(Engine &engine);

    bool active;
    bool use_gl;
    SDL_Window *window;
    SDL_Renderer *renderer;

    // Text fields need somewhere to live between frames.
    char env_path[256];
    char model_path[256];

    // Smoothed for readability: the raw per-frame numbers flicker too fast to
    // read, and the thing being watched is a trend, not one frame.
    float fps_smooth;
    float upload_ms_smooth, bin_ms_smooth, raster_ms_smooth;
};

#endif // __UI_H__
