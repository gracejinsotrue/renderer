#ifndef REALTIME_RAYTRACER_H
#define REALTIME_RAYTRACER_H

#include "unified_math.h"
#include "Scene.h"
#include "tgaimage.h"
#include "ray_tracer_integration.h" // For existing ray tracing classes
#include <iomanip>                  // For std::setprecision
#include <chrono>

// TODO: kind of a big WIP
class RealtimeRayTracer
{
private:
    // Basic ray tracing settings
    int rt_width, rt_height;            // Ray trace resolution (lower than main framebuffer)
    int tile_size;                      // size of tiles to ray trace per frame
    int current_tile_x, current_tile_y; // current tile being ray traced
    int total_tiles_x, total_tiles_y;   // Total number of tiles
    bool is_active;                     // Whether real-time ray tracing is enabled

    // Buffers
    TGAImage rt_framebuffer;   // Low-res ray traced image
    TGAImage progress_overlay; // Visual indicator of ray traced areas
    rt_hittable_list world;    // Ray traced scene
    bool world_needs_update;   // Flag to rebuild scene when needed

    // Camera
    rt_camera rt_cam;

    // Performance tracking
    int frames_since_camera_move; // Reset tile progress when camera moves
    Vec3f last_camera_position;   // To detect camera movement

    // Quality and performance settings
    int quality_level;
    float blend_strength;       // How much to blend RT with rasterizer (0.0-1.0)
    bool show_progress_overlay; // Show visual progress indicator
    bool adaptive_quality;      // automatically adjust quality based on performance

    // performance racking
    std::chrono::high_resolution_clock::time_point last_frame_time;
    float average_frame_time; // Moving average of frame times
    int performance_samples;  // Number of samples for averaging

    bool show_tile_boundaries;

public:
    RealtimeRayTracer(int render_width, int render_height)
        : rt_width(render_width / 2), rt_height(render_height / 2), // half resolution for speed
          tile_size(16),                                            // 16x16 pixel tiles
          current_tile_x(0), current_tile_y(0),
          is_active(false),
          rt_framebuffer(rt_width, rt_height, TGAImage::RGB),
          progress_overlay(render_width, render_height, TGAImage::RGB),
          world_needs_update(true),
          frames_since_camera_move(0),
          last_camera_position(0, 0, 0),
          quality_level(2),     // start with medium quality
          blend_strength(0.7f), // 70% ray traced, 30% rasterizer
          show_progress_overlay(true),
          adaptive_quality(false),
          average_frame_time(16.67f),
          performance_samples(0),
          show_tile_boundaries(false)
    {
        total_tiles_x = (rt_width + tile_size - 1) / tile_size;
        total_tiles_y = (rt_height + tile_size - 1) / tile_size;

        // Initialize buffers
        rt_framebuffer.clear();
        progress_overlay.clear();
        last_frame_time = std::chrono::high_resolution_clock::now();

        update_quality_settings();

        std::cout << "Enhanced RealtimeRayTracer initialized:" << std::endl;
        std::cout << "  RT Resolution: " << rt_width << "x" << rt_height << std::endl;
        std::cout << "  Quality level: " << quality_level << "/4" << std::endl;
        std::cout << "  Blend strength: " << (blend_strength * 100) << "%" << std::endl;
        std::cout << "  Total tiles: " << total_tiles_x << "x" << total_tiles_y << " = "
                  << (total_tiles_x * total_tiles_y) << std::endl;
    }

    void toggle()
    {
        is_active = !is_active;
        if (is_active)
        {
            std::cout << "Real-time ray tracing ENABLED" << std::endl;
            reset_tiles(); // start fresh
        }
        else
        {
            std::cout << "Real-time ray tracing DISABLED" << std::endl;
        }
    }

    bool is_enabled() const { return is_active; }

    void increase_quality()
    {
        if (quality_level < 4)
        {
            quality_level++;
            update_quality_settings();
            reset_tiles();
            std::cout << "Ray tracing quality increased to " << quality_level << "/4" << std::endl;
        }
    }

    void decrease_quality()
    {
        if (quality_level > 1)
        {
            quality_level--;
            update_quality_settings();
            reset_tiles();
            std::cout << "Ray tracing quality decreased to " << quality_level << "/4" << std::endl;
        }
    }

    void adjust_blend_strength(float delta)
    {
        blend_strength = std::max(0.0f, std::min(1.0f, blend_strength + delta));
        std::cout << "Ray tracing blend strength: " << (blend_strength * 100) << "%" << std::endl;
    }

    void toggle_progress_overlay()
    {
        show_progress_overlay = !show_progress_overlay;
        if (!show_progress_overlay)
        {
            progress_overlay.clear(); // Clear overlay when disabled
        }
        std::cout << "Progress overlay: " << (show_progress_overlay ? "ON" : "OFF") << std::endl;
    }

    void toggle_adaptive_quality()
    {
        adaptive_quality = !adaptive_quality;
        std::cout << "Adaptive quality: " << (adaptive_quality ? "ON" : "OFF") << std::endl;
    }

    void toggle_tile_boundaries()
    {
        show_tile_boundaries = !show_tile_boundaries;
        std::cout << "Tile boundaries: " << (show_tile_boundaries ? "ON" : "OFF") << std::endl;
    }

    void update_scene(Scene &scene)
    {
        if (!is_active)
            return;

        // Check if camera moved (reset tiles if so)
        Vec3f current_camera_pos = scene.camera.position;
        float camera_move_threshold = 0.01f;
        if ((current_camera_pos - last_camera_position).norm() > camera_move_threshold)
        {
            reset_tiles();
            last_camera_position = current_camera_pos;
            frames_since_camera_move = 0;
        }
        else
        {
            frames_since_camera_move++;
        }

        // Update ray traced scene if needed
        if (world_needs_update)
        {
            world = SceneToRayTracer::convert_scene(scene);
            world_needs_update = false;
            std::cout << "Ray traced scene updated" << std::endl;
        }

        // Update ray tracer camera to match rasterizer
        rt_cam.aspect_ratio = float(rt_width) / float(rt_height);
        rt_cam.image_width = rt_width;
        rt_cam.vfov = 45;

        rt_cam.lookfrom = raster_to_point(scene.camera.position);
        rt_cam.lookat = raster_to_point(scene.camera.target);
        rt_cam.vup = raster_to_rt(scene.camera.up);
    }

    // better render method with performance tracking
    void render_one_tile()
    {
        if (!is_active)
            return;

        auto start_time = std::chrono::high_resolution_clock::now();

        // only ray trace if we have tiles left to do
        if (current_tile_y < total_tiles_y)
        {
            int tile_start_x = current_tile_x * tile_size;
            int tile_start_y = current_tile_y * tile_size;
            int tile_end_x = std::min(tile_start_x + tile_size, rt_width);
            int tile_end_y = std::min(tile_start_y + tile_size, rt_height);

            // ray trace this tile with current quality settings
            ray_trace_tile_enhanced(tile_start_x, tile_start_y, tile_end_x, tile_end_y);

            // Update progress overlay
            if (show_progress_overlay)
            {
                update_progress_overlay(tile_start_x, tile_start_y, tile_end_x, tile_end_y);
            }

            // Move to next tile
            advance_tile();
        }

        // Performance tracking
        auto end_time = std::chrono::high_resolution_clock::now();
        float tile_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        update_performance_stats(tile_time);

        if (adaptive_quality)
        {
            adjust_quality_based_on_performance();
        }
    }

    // Better blending with multiple modes
    void blend_with_framebuffer(TGAImage &main_framebuffer)
    {
        if (!is_active)
            return;

        int main_width = main_framebuffer.get_width();
        int main_height = main_framebuffer.get_height();

        for (int y = 0; y < main_height; y++)
        {
            for (int x = 0; x < main_width; x++)
            {
                // Map main framebuffer coordinates to ray trace coordinates
                int rt_x = (x * rt_width) / main_width;
                int rt_y = (y * rt_height) / main_height;

                rt_x = std::max(0, std::min(rt_width - 1, rt_x));
                rt_y = std::max(0, std::min(rt_height - 1, rt_y));

                // Get colors
                TGAColor rt_color = rt_framebuffer.get(rt_x, rt_y);
                TGAColor raster_color = main_framebuffer.get(x, y);

                bool has_rt_data = (rt_color[0] > 0 || rt_color[1] > 0 || rt_color[2] > 0);

                if (has_rt_data)
                {

                    float current_blend = blend_strength;

                    // reduce blend strength for lower quality to avoid artifacts
                    if (quality_level == 1)
                        current_blend *= 0.6f;

                    TGAColor blended(
                        int(rt_color[2] * current_blend + raster_color[2] * (1.0f - current_blend)), // R
                        int(rt_color[1] * current_blend + raster_color[1] * (1.0f - current_blend)), // G
                        int(rt_color[0] * current_blend + raster_color[0] * (1.0f - current_blend))  // B
                    );

                    main_framebuffer.set(x, y, blended);
                }

                // add progress overlay if enabled
                if (show_progress_overlay)
                {
                    TGAColor overlay_color = progress_overlay.get(x, y);
                    if (overlay_color[0] > 0 || overlay_color[1] > 0 || overlay_color[2] > 0)
                    {
                        // Blend progress overlay
                        TGAColor current_color = main_framebuffer.get(x, y);
                        TGAColor with_overlay(
                            std::min(255, current_color[2] + overlay_color[2] / 4), // R
                            std::min(255, current_color[1] + overlay_color[1] / 4), // G
                            std::min(255, current_color[0] + overlay_color[0] / 4)  // B
                        );
                        main_framebuffer.set(x, y, with_overlay);
                    }
                }
            }
        }
    }

    void reset_tiles()
    {
        current_tile_x = 0;
        current_tile_y = 0;
        rt_framebuffer.clear(); // clear previous ray traced image
        if (show_progress_overlay)
        {
            progress_overlay.clear(); // Clear progress overlay
        }
        std::cout << "Ray trace tiles reset" << std::endl;
    }

    void mark_scene_dirty()
    {
        world_needs_update = true;
        reset_tiles(); // Start over when scene changes
    }

    // status
    void print_detailed_status() const
    {
        if (!is_active)
        {
            std::cout << "Real-time ray tracing: DISABLED" << std::endl;
            return;
        }

        float progress = float(current_tile_y * total_tiles_x + current_tile_x) /
                         float(total_tiles_x * total_tiles_y) * 100.0f;

        std::cout << "=== REAL-TIME RAY TRACING STATUS ===" << std::endl;
        std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "%" << std::endl;
        std::cout << "Quality: " << quality_level << "/4 ";

        switch (quality_level)
        {
        case 1:
            std::cout << "(Fast)";
            break;
        case 2:
            std::cout << "(Medium)";
            break;
        case 3:
            std::cout << "(High)";
            break;
        case 4:
            std::cout << "(Ultra)";
            break;
        }
        std::cout << std::endl;

        std::cout << "Blend strength: " << (blend_strength * 100) << "%" << std::endl;
        std::cout << "Avg frame time: " << average_frame_time << "ms" << std::endl;
        std::cout << "Adaptive quality: " << (adaptive_quality ? "ON" : "OFF") << std::endl;
        std::cout << "Progress overlay: " << (show_progress_overlay ? "ON" : "OFF") << std::endl;
        std::cout << "Tile boundaries: " << (show_tile_boundaries ? "ON" : "OFF") << std::endl;
        std::cout << "====================================" << std::endl;
    }

    // debug info (simplified)
    void print_status() const
    {
        if (is_active)
        {
            float progress = float(current_tile_y * total_tiles_x + current_tile_x) /
                             float(total_tiles_x * total_tiles_y) * 100.0f;
            std::cout << "RT Progress: " << std::fixed << std::setprecision(1)
                      << progress << "% (Quality: " << quality_level << "/4)" << std::endl;
        }
    }

private:
    void update_quality_settings()
    {
        // Adjust settings based on quality level
        switch (quality_level)
        {
        case 1: // fast
            rt_cam.samples_per_pixel = 1;
            rt_cam.max_depth = 2;
            tile_size = 32;
            break;
        case 2: // medium
            rt_cam.samples_per_pixel = 1;
            rt_cam.max_depth = 3;
            tile_size = 16;
            break;
        case 3: // high
            rt_cam.samples_per_pixel = 2;
            rt_cam.max_depth = 4;
            tile_size = 8;
            break;
        case 4: // lutra
            rt_cam.samples_per_pixel = 4;
            rt_cam.max_depth = 6;
            tile_size = 8;
            break;
        }

        // recalculate tile counts
        total_tiles_x = (rt_width + tile_size - 1) / tile_size;
        total_tiles_y = (rt_height + tile_size - 1) / tile_size;
    }

    void ray_trace_tile_enhanced(int start_x, int start_y, int end_x, int end_y)
    {

        // initialize ray tracer camera
        rt_point3 center = rt_cam.lookfrom;

        auto theta = degrees_to_radians(rt_cam.vfov);
        auto h = std::tan(theta / 2);
        auto viewport_height = 2 * h;
        auto viewport_width = viewport_height * rt_cam.aspect_ratio;

        rt_vec3 w = unit_vector(rt_cam.lookfrom - rt_cam.lookat);
        rt_vec3 u = unit_vector(cross(rt_cam.vup, w));
        rt_vec3 v = cross(w, u);

        rt_vec3 viewport_u = viewport_width * u;
        rt_vec3 viewport_v = viewport_height * -v;

        rt_vec3 pixel_delta_u = viewport_u / rt_width;
        rt_vec3 pixel_delta_v = viewport_v / rt_height;

        auto viewport_upper_left = center - w - viewport_u / 2 - viewport_v / 2;
        rt_point3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // ray trace each pixel in the tile with multiple samples
        for (int y = start_y; y < end_y; y++)
        {
            for (int x = start_x; x < end_x; x++)
            {
                rt_color pixel_color(0, 0, 0);

                for (int sample = 0; sample < rt_cam.samples_per_pixel; sample++)
                {
                    // small, random offset for anti-aliasing
                    float offset_x = (sample > 0) ? (random_double() - 0.5f) : 0.0f;
                    float offset_y = (sample > 0) ? (random_double() - 0.5f) : 0.0f;

                    auto pixel_sample = pixel00_loc +
                                        ((x + offset_x) * pixel_delta_u) +
                                        ((y + offset_y) * pixel_delta_v);

                    rt_ray r(center, pixel_sample - center);
                    pixel_color += ray_color(r, rt_cam.max_depth);
                }

                pixel_color = pixel_color / rt_cam.samples_per_pixel;

                // Convert and store
                TGAColor tga_color = ColorConversion::rt_color_to_tga(pixel_color);
                rt_framebuffer.set(x, y, tga_color);
            }
        }
    }

    void update_progress_overlay(int start_x, int start_y, int end_x, int end_y)
    {
        // Map RT coordinates to main framebuffer coordinates
        int main_width = progress_overlay.get_width();
        int main_height = progress_overlay.get_height();

        int main_start_x = (start_x * main_width) / rt_width;
        int main_start_y = (start_y * main_height) / rt_height;
        int main_end_x = (end_x * main_width) / rt_width;
        int main_end_y = (end_y * main_height) / rt_height;

        // Add colored overlay for completed tiles
        TGAColor overlay_color(0, 50, 0); // Subtle green tint

        for (int y = main_start_y; y < main_end_y; y++)
        {
            for (int x = main_start_x; x < main_end_x; x++)
            {
                if (x >= 0 && x < main_width && y >= 0 && y < main_height)
                {
                    progress_overlay.set(x, y, overlay_color);
                }
            }
        }

        if (show_tile_boundaries)
        {
            TGAColor boundary_color(100, 100, 0); // yellow boundaries

            // Draw boundary lines
            for (int x = main_start_x; x < main_end_x; x++)
            {
                if (main_start_y >= 0 && main_start_y < main_height)
                    progress_overlay.set(x, main_start_y, boundary_color);
                if (main_end_y - 1 >= 0 && main_end_y - 1 < main_height)
                    progress_overlay.set(x, main_end_y - 1, boundary_color);
            }
            for (int y = main_start_y; y < main_end_y; y++)
            {
                if (main_start_x >= 0 && main_start_x < main_width)
                    progress_overlay.set(main_start_x, y, boundary_color);
                if (main_end_x - 1 >= 0 && main_end_x - 1 < main_width)
                    progress_overlay.set(main_end_x - 1, y, boundary_color);
            }
        }
    }

    void advance_tile()
    {
        current_tile_x++;
        if (current_tile_x >= total_tiles_x)
        {
            current_tile_x = 0;
            current_tile_y++;
            if (current_tile_y >= total_tiles_y)
            {
                current_tile_y = 0; // Start over
            }
        }
    }

    void update_performance_stats(float tile_time)
    {
        // update moving average of frame times
        performance_samples++;
        float alpha = 1.0f / std::min(60, performance_samples); // 60-sample moving average
        average_frame_time = average_frame_time * (1.0f - alpha) + tile_time * alpha;
    }

    void adjust_quality_based_on_performance()
    {
        // target: keep tile time under 2ms for smooth interaction
        const float target_tile_time = 2.0f;

        if (average_frame_time > target_tile_time * 2.0f && quality_level > 1)
        {
            // too slow, decrease quality
            decrease_quality();
        }
        else if (average_frame_time < target_tile_time * 0.5f && quality_level < 4)
        {
            // fast enough, increase quality
            increase_quality();
        }
    }

    rt_color ray_color(const rt_ray &r, int depth) const
    {
        if (depth <= 0)
            return rt_color(0, 0, 0);

        rt_hit_record rec;
        if (world.hit(r, 0.001, rt_infinity, rec))
        {
            rt_ray scattered;
            rt_color attenuation;
            if (rec.mat->scatter(r, rec.p, rec.normal, attenuation, scattered))
                return attenuation * ray_color(scattered, depth - 1);
            return rt_color(0, 0, 0);
        }

        // black background
        return rt_color(0, 0, 0);
    }
};

#endif // REALTIME_RAYTRACER_H