#ifndef RAYTRACING_H
#define RAYTRACING_H

#include "unified_math.h"
#include "Scene.h"
#include "tgaimage.h"
#include "ray_tracer_integration.h" 
#include <iomanip>                  
#include <chrono>

// TODO: kind of a big WIP
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

    int quality_level;
    float blend_strength;      
    bool show_progress_overlay; 
    bool adaptive_quality;     

    // performance racking
    std::chrono::high_resolution_clock::time_point last_frame_time;
    float average_frame_time; 
    int performance_samples;  

    bool show_tile_boundaries;

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
};

#endif // RAYTRACING_H