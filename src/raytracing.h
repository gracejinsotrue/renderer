#ifndef RAYTRACING_H
#define RAYTRACING_H

#include "unified_math.h"
#include "Scene.h"
#include "tgaimage.h"
#include "ray_tracer_integration.h" 
#include <iomanip>
#include <chrono>
#include <vector>

// Progressive ray tracer that runs alongside the rasterizer and blends
// over its frame. Traces on the GPU when CUDA is available, otherwise a
// tile at a time on the CPU so the editor stays responsive.
class RealtimeRayTracer
{
private:
    // this is for basic ray tracing settings
    int rt_width, rt_height;            
    int tile_size;                      // size of tiles to ray trace per frame
    int current_tile_x, current_tile_y; // current tile being ray traced
    int total_tiles_x, total_tiles_y;  
    bool is_active;                    

    // Buffers
    TGAImage rt_framebuffer; 
    TGAImage progress_overlay; 
    rt_hittable_list world;    // R=ray traced scene
    std::shared_ptr<rt_bvh> bvh_world;  // BVH
    bool world_needs_update;  

    rt_camera rt_cam;

    int frames_since_camera_move;
    Vec3f last_camera_position;
    Vec3f last_camera_target;

    // Cheap fingerprint of everything the traced scene depends on: which
    // meshes exist, where they sit, and what their geometry version is.
    // Comparing it every frame catches sculpting, blend shapes, node moves
    // and add/delete without having to find every mutation site by hand.
    unsigned long long scene_signature;
    unsigned long long compute_scene_signature(Scene &scene) const;

    // A rebuild is convert + BVH build + upload, all on the CPU, and on a
    // heavy model that is seconds rather than milliseconds. Sculpting fires a
    // change every frame, so rebuilding immediately would freeze the editor
    // solid. Instead the rebuild waits until the scene has been still for a
    // moment, which is what a progressive renderer does anyway.
    bool rebuild_pending;
    std::chrono::high_resolution_clock::time_point last_scene_change;
    static const int REBUILD_DELAY_MS = 200;

    // samples per pixel accumulated into the current image
    int accumulated_samples;
    void restart_accumulation();

    int quality_level;
    float blend_strength;      
    bool show_progress_overlay; 
    bool adaptive_quality;     

    // performance racking
    std::chrono::high_resolution_clock::time_point last_frame_time;
    float average_frame_time; 
    int performance_samples;  

    bool show_tile_boundaries;

    // CUDA path tracer. When the device is available the whole frame is traced
    // in one kernel launch, so the tile-at-a-time scheme the CPU needs is
    // bypassed entirely. cuda_scene_tris is 0 until a scene has been uploaded.
    bool cuda_available;
    bool use_cuda;
    bool gpu_composite;
    int cuda_scene_tris;
    std::vector<unsigned char> cuda_readback;

    bool upload_scene_to_gpu();
    bool render_frame_gpu(bool readback);

    // Private methods
    void update_quality_settings();
    void ray_trace_tile_enhanced(int start_x, int start_y, int end_x, int end_y);
    void update_progress_overlay(int start_x, int start_y, int end_x, int end_y);
    void advance_tile();
    void update_performance_stats(float tile_time);
    void adjust_quality_based_on_performance();
    rt_color ray_color(const rt_ray &r, int depth) const;

public:
    RealtimeRayTracer(int render_width, int render_height);

    void toggle();

    bool is_enabled() const { return is_active; }

    void increase_quality();

    void decrease_quality();

    void adjust_blend_strength(float delta);

    void toggle_progress_overlay();

    void toggle_adaptive_quality();

    void toggle_tile_boundaries();

    void update_scene(Scene &scene);

    void render_one_tile();

    void blend_with_framebuffer(TGAImage &main_framebuffer);

    void reset_tiles();

    void mark_scene_dirty();

    void print_detailed_status() const;

    void print_status() const;

    // Traces and composites entirely on the device, leaving the finished
    // frame in the rasterizer's own framebuffer. Returns false if the GPU
    // path is unavailable, in which case the caller must fall back to
    // render_one_tile() plus blend_with_framebuffer().
    bool render_and_blend_on_gpu();

    // Compositing on the device keeps the frame there. Turning this off falls
    // back to pulling both images down and blending on the host, which is the
    // only route when something else needs the frame in host memory anyway.
    void set_gpu_composite(bool on) { gpu_composite = on; }
    bool uses_gpu_composite() const { return gpu_composite; }

    // the traced image before compositing, for tests and for the host blend
    const TGAImage &get_rt_framebuffer() const { return rt_framebuffer; }
    int get_rt_width() const { return rt_width; }
    int get_rt_height() const { return rt_height; }
    float get_blend_strength() const { return blend_strength; }

    int get_accumulated_samples() const { return accumulated_samples; }

    // vertical field of view in degrees, derived from the raster viewport
    double get_rt_vfov() const { return rt_cam.vfov; }

    void toggle_cuda();
    bool is_cuda_enabled() const { return cuda_available && use_cuda; }
    bool is_cuda_available() const { return cuda_available; }
};

#endif // RAYTRACING_H
