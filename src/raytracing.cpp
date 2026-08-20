#include "raytracing.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <map>

extern "C" {
    bool initCudaRayTracer(int width, int height);
    void cleanupCudaRayTracer();
    bool cudaRTSetScene(const float *verts, const int *tri_mat, int ntri,
                        const float *albedos, int nmat);
    void cudaRTSetCamera(const float *origin3, const float *pixel00_3,
                         const float *du3, const float *dv3);
    void cudaRTSetSky(const float *lo3, const float *hi3);
    void cudaRTRender(int spp, int max_depth, int mode, unsigned int seed);
    void cudaRTGetResults(unsigned char *host_rgb);
    void cudaRTResetAccumulation();
    bool cudaRTSetSceneWithMaterials(const float *verts, const int *tri_mat, int ntri,
                                     const float *albedos, int nmat,
                                     const int *types, const float *fuzz,
                                     const float *iors);
    void cudaRTBlendToFramebuffer(unsigned char *dst, int dst_w, int dst_h, float blend);
    unsigned char *cudaGetDeviceFramebuffer(int *w, int *h);
}

// =============================================================================
// CONSTRUCTOR
// =============================================================================

RealtimeRayTracer::RealtimeRayTracer(int render_width, int render_height)
    : rt_width(render_width / 2), rt_height(render_height / 2), // half resolution for speed
      tile_size(16),                                            // 16x16 pixel tiles
      current_tile_x(0), current_tile_y(0),
      is_active(false),
      rt_framebuffer(rt_width, rt_height, TGAImage::RGB),
      progress_overlay(render_width, render_height, TGAImage::RGB),
      world_needs_update(true),
      frames_since_camera_move(0),
      last_camera_position(0, 0, 0),
      last_camera_target(0, 0, 0),
      scene_signature(0),
      rebuild_pending(false),
      accumulated_samples(0),
      quality_level(2),     
      blend_strength(0.7f), 
      show_progress_overlay(true),
      adaptive_quality(false),
      average_frame_time(16.67f),
      performance_samples(0),
      show_tile_boundaries(false),
      cuda_available(false),
      use_cuda(false),
      gpu_composite(true),
      cuda_scene_tris(0)
{
    total_tiles_x = (rt_width + tile_size - 1) / tile_size;
    total_tiles_y = (rt_height + tile_size - 1) / tile_size;

    rt_framebuffer.clear();
    progress_overlay.clear();
    last_frame_time = std::chrono::high_resolution_clock::now();

    update_quality_settings();

    cuda_available = initCudaRayTracer(rt_width, rt_height);
    use_cuda = cuda_available;

    std::cout << "Enhanced RealtimeRayTracer initialized:" << std::endl;
    std::cout << "  CUDA: " << (cuda_available ? "available" : "not available") << std::endl;
    std::cout << "  RT Resolution: " << rt_width << "x" << rt_height << std::endl;
    std::cout << "  Quality level: " << quality_level << "/4" << std::endl;
    std::cout << "  Blend strength: " << (blend_strength * 100) << "%" << std::endl;
    std::cout << "  Total tiles: " << total_tiles_x << "x" << total_tiles_y << " = "
              << (total_tiles_x * total_tiles_y) << std::endl;
}

// =============================================================================
// PUBLIC METHODS
// =============================================================================

void RealtimeRayTracer::toggle()
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

void RealtimeRayTracer::increase_quality()
{
    if (quality_level < 4)
    {
        quality_level++;
        update_quality_settings();
        reset_tiles();
        std::cout << "Ray tracing quality increased to " << quality_level << "/4" << std::endl;
    }
}

void RealtimeRayTracer::decrease_quality()
{
    if (quality_level > 1)
    {
        quality_level--;
        update_quality_settings();
        reset_tiles();
        std::cout << "Ray tracing quality decreased to " << quality_level << "/4" << std::endl;
    }
}

void RealtimeRayTracer::adjust_blend_strength(float delta)
{
    blend_strength = std::max(0.0f, std::min(1.0f, blend_strength + delta));
    std::cout << "Ray tracing blend strength: " << (blend_strength * 100) << "%" << std::endl;
}

void RealtimeRayTracer::toggle_progress_overlay()
{
    show_progress_overlay = !show_progress_overlay;
    if (!show_progress_overlay)
    {
        progress_overlay.clear(); // Clear overlay when disabled
    }
    std::cout << "Progress overlay: " << (show_progress_overlay ? "ON" : "OFF") << std::endl;
}

void RealtimeRayTracer::toggle_adaptive_quality()
{
    adaptive_quality = !adaptive_quality;
    std::cout << "Adaptive quality: " << (adaptive_quality ? "ON" : "OFF") << std::endl;
}

void RealtimeRayTracer::toggle_tile_boundaries()
{
    show_tile_boundaries = !show_tile_boundaries;
    std::cout << "Tile boundaries: " << (show_tile_boundaries ? "ON" : "OFF") << std::endl;
}

// Everything the traced image depends on, folded into one value. Node
// transforms are included via their world matrix, and geometry edits via
// Model::geometryVersion(), which every method that writes vertices bumps.
unsigned long long RealtimeRayTracer::compute_scene_signature(Scene &scene) const
{
    std::vector<SceneNode *> nodes;
    scene.getAllMeshNodes(nodes);

    unsigned long long h = 1469598103934665603ULL;
    auto mix = [&h](unsigned long long v)
    {
        h ^= v;
        h *= 1099511628211ULL;
    };

    mix((unsigned long long)nodes.size());
    for (size_t i = 0; i < nodes.size(); i++)
    {
        SceneNode *n = nodes[i];
        if (!n || !n->model)
            continue;
        mix((unsigned long long)n->model->geometryVersion());
        mix(n->isVisible() ? 1ULL : 2ULL);

        Matrix m = n->getWorldMatrix();
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
            {
                float f = (float)m[r][c];
                unsigned int bits;
                memcpy(&bits, &f, sizeof(bits));
                mix((unsigned long long)bits);
            }
    }
    return h;
}

// Throws away the samples gathered so far. Anything that changes what the
// image should look like has to call this, or old samples smear into the new
// picture.
void RealtimeRayTracer::restart_accumulation()
{
    accumulated_samples = 0;
    frames_since_camera_move = 0;
    if (cuda_available)
        cudaRTResetAccumulation();
}

void RealtimeRayTracer::update_scene(Scene &scene)
{
    if (!is_active)
        return;

    // Camera moves invalidate the image. Target as well as position: orbiting
    // can swing the target while the eye barely moves.
    Vec3f current_camera_pos = scene.camera.position;
    Vec3f current_camera_target = scene.camera.target;
    float camera_move_threshold = 0.01f;
    bool camera_moved =
        (current_camera_pos - last_camera_position).norm() > camera_move_threshold ||
        (current_camera_target - last_camera_target).norm() > camera_move_threshold;

    if (camera_moved)
    {
        reset_tiles();
        last_camera_position = current_camera_pos;
        last_camera_target = current_camera_target;
        restart_accumulation();
    }
    else
    {
        frames_since_camera_move++;
    }

    // Sculpting, blend shapes, node moves and add/delete all land here.
    // Nothing else marked the traced scene dirty, so before this the ray
    // tracer kept drawing the mesh as it was when it was switched on.
    unsigned long long sig = compute_scene_signature(scene);
    if (sig != scene_signature)
    {
        scene_signature = sig;
        rebuild_pending = true;
        last_scene_change = std::chrono::high_resolution_clock::now();
        reset_tiles();
        restart_accumulation();
    }

    // hold off until the edits stop, so a sculpt stroke does not pay for a
    // full rebuild on every frame of the drag
    if (rebuild_pending)
    {
        float since = std::chrono::duration<float, std::milli>(
                          std::chrono::high_resolution_clock::now() - last_scene_change)
                          .count();
        if (since >= (float)REBUILD_DELAY_MS)
        {
            rebuild_pending = false;
            world_needs_update = true;
        }
    }

    if (world_needs_update)
    {
        world = SceneToRayTracer::convert_scene(scene);
        // The CPU BVH is only ever read by the CPU tracer. Building it when
        // the GPU is doing the tracing is the single most expensive part of a
        // rebuild and nothing consumes it, so leave it null and build lazily
        // if the CPU path is ever switched back on.
        bvh_world.reset();
        if (cuda_available && use_cuda)
        {
            upload_scene_to_gpu();
        }
        else
        {
            bvh_world = world.build_bvh();
            if (cuda_available)
                upload_scene_to_gpu();
        }
        world_needs_update = false;
        // samples gathered during the debounce window traced the old
        // geometry, so they cannot be blended with what comes next
        restart_accumulation();
        std::cout << "Ray traced scene updated" << std::endl;
    }

    rt_cam.aspect_ratio = float(rt_width) / float(rt_height);
    rt_cam.image_width = rt_width;

    rt_cam.lookfrom = raster_to_point(scene.camera.position);
    rt_cam.lookat = raster_to_point(scene.camera.target);
    rt_cam.vup = raster_to_rt(scene.camera.up);
    // The traced image composites over the rasterized frame, so it has to
    // cover exactly the same view volume. The rasterizer puts one world unit
    // at the target plane across (renderHeight * 3/8) pixels: Engine uses a
    // viewport three quarters the size of the frame, and Projection is the
    // identity at the target plane, so ndc y = 1 lands one world unit up.
    // The tracer covers the whole frame, which is (renderHeight / 2) pixels
    // from the centre, hence 4/3 world units. A hardcoded vfov cannot track
    // that and leaves the two images at different scales, which shows up as
    // a bright fringe wherever the silhouettes disagree.
    const double RASTER_HALF_EXTENT = 4.0 / 3.0;
    double cam_dist = (rt_cam.lookfrom - rt_cam.lookat).length();
    if (cam_dist > 1e-6)
        rt_cam.vfov = 2.0 * std::atan(RASTER_HALF_EXTENT / cam_dist) * 180.0 / 3.14159265358979323846;
}

// Pulls the triangles back out of the converted scene. Going through
// rt_hittable_list rather than re-walking the scene graph keeps
// SceneToRayTracer as the single place that applies world transforms.
bool RealtimeRayTracer::upload_scene_to_gpu()
{
    std::vector<float> verts;
    std::vector<int> tri_mat;
    std::vector<float> albedos;
    std::vector<int> mat_types;
    std::vector<float> mat_fuzz;
    std::vector<float> mat_ior;
    std::map<const rt_material *, int> mat_index;
    verts.reserve(world.objects.size() * 9);
    tri_mat.reserve(world.objects.size());

    for (size_t i = 0; i < world.objects.size(); i++)
    {
        rt_triangle *tri = dynamic_cast<rt_triangle *>(world.objects[i].get());
        if (!tri)
            continue;   // spheres and anything else stay CPU-only for now
        const rt_point3 *v[3] = { &tri->v0, &tri->v1, &tri->v2 };
        for (int k = 0; k < 3; k++)
            for (int a = 0; a < 3; a++)
                verts.push_back((float)(*v[k])[a]);

        // one device material per distinct CPU material, so meshes keep the
        // colour SceneToRayTracer gave them instead of sharing one albedo
        const rt_material *mp = tri->mat.get();
        std::map<const rt_material *, int>::iterator it = mat_index.find(mp);
        int idx;
        if (it == mat_index.end())
        {
            idx = (int)(albedos.size() / 3);
            rt_color a(0.7, 0.3, 0.3);
            int type = 0;
            float fz = 0.f, ior = 1.5f;

            if (const rt_lambertian *lam = dynamic_cast<const rt_lambertian *>(mp))
            {
                a = lam->albedo;
                type = 0;
            }
            else if (const rt_metal *met = dynamic_cast<const rt_metal *>(mp))
            {
                a = met->albedo;
                fz = (float)met->fuzz;
                type = 1;
            }
            else if (const rt_dielectric *die = dynamic_cast<const rt_dielectric *>(mp))
            {
                ior = (float)die->refraction_index;
                type = 2;
            }

            albedos.push_back((float)a.x());
            albedos.push_back((float)a.y());
            albedos.push_back((float)a.z());
            mat_types.push_back(type);
            mat_fuzz.push_back(fz);
            mat_ior.push_back(ior);
            mat_index[mp] = idx;
        }
        else
        {
            idx = it->second;
        }
        tri_mat.push_back(idx);
    }

    cuda_scene_tris = (int)(verts.size() / 9);
    if (cuda_scene_tris == 0)
        return false;
    if (albedos.empty())
    {
        albedos.push_back(0.7f);
        albedos.push_back(0.3f);
        albedos.push_back(0.3f);
        mat_types.push_back(0);
        mat_fuzz.push_back(0.f);
        mat_ior.push_back(1.5f);
    }

    if (!cudaRTSetSceneWithMaterials(verts.data(), tri_mat.data(), cuda_scene_tris,
                                     albedos.data(), (int)(albedos.size() / 3),
                                     mat_types.data(), mat_fuzz.data(), mat_ior.data()))
    {
        cuda_scene_tris = 0;
        return false;
    }

    // The realtime CPU tracer returns black on a miss and nothing in the scene
    // emits, so with a black sky every path terminates at black. Use the same
    // gradient the offline rt_camera does, which is the scene's only light.
    const float sky_lo[3] = { 0.0f, 0.0f, 0.0f };
    const float sky_hi[3] = { 0.8f, 0.8f, 0.8f };
    cudaRTSetSky(sky_lo, sky_hi);

    std::cout << "Ray tracer scene uploaded to GPU: " << cuda_scene_tris
              << " triangles, " << (albedos.size() / 3) << " materials" << std::endl;
    return true;
}

// One kernel launch for the whole frame. At these rates there is nothing to
// gain from spreading the work across frames the way the CPU path has to.
bool RealtimeRayTracer::render_frame_gpu(bool readback)
{
    if (!cuda_available || cuda_scene_tris == 0)
        return false;

    rt_point3 center = rt_cam.lookfrom;
    double theta = degrees_to_radians(rt_cam.vfov);
    double h = std::tan(theta / 2);
    double viewport_height = 2 * h;
    double viewport_width = viewport_height * rt_cam.aspect_ratio;

    rt_vec3 w = unit_vector(rt_cam.lookfrom - rt_cam.lookat);
    rt_vec3 u = unit_vector(cross(rt_cam.vup, w));
    rt_vec3 v = cross(w, u);

    rt_vec3 viewport_u = viewport_width * u;
    rt_vec3 viewport_v = viewport_height * -v;
    rt_vec3 pixel_delta_u = viewport_u / rt_width;
    rt_vec3 pixel_delta_v = viewport_v / rt_height;
    rt_point3 pixel00 = center - w - viewport_u / 2 - viewport_v / 2 +
                        0.5 * (pixel_delta_u + pixel_delta_v);

    float c_o[3]  = { (float)center.x(), (float)center.y(), (float)center.z() };
    float c_p[3]  = { (float)pixel00.x(), (float)pixel00.y(), (float)pixel00.z() };
    float c_du[3] = { (float)pixel_delta_u.x(), (float)pixel_delta_u.y(), (float)pixel_delta_u.z() };
    float c_dv[3] = { (float)pixel_delta_v.x(), (float)pixel_delta_v.y(), (float)pixel_delta_v.z() };
    cudaRTSetCamera(c_o, c_p, c_du, c_dv);

    // the seed advances every pass so successive passes add new samples
    // rather than repeating the ones already accumulated
    cudaRTRender(rt_cam.samples_per_pixel, rt_cam.max_depth, 0,
                 1u + (unsigned int)accumulated_samples * 2654435761u);
    accumulated_samples += rt_cam.samples_per_pixel;

    if (!readback)
        return true;   // the caller will composite on the device

    cuda_readback.resize((size_t)rt_width * rt_height * 3);
    cudaRTGetResults(cuda_readback.data());

    for (int y = 0; y < rt_height; y++)
        for (int x = 0; x < rt_width; x++)
        {
            const unsigned char *p = &cuda_readback[((size_t)y * rt_width + x) * 3];
            rt_framebuffer.set(x, y, TGAColor(p[0], p[1], p[2]));
        }
    return true;
}

// Traces and blends without either image touching the host. The CPU route
// costs two device-to-host DMAs and a per-pixel pass over the whole frame,
// which dwarfs the trace itself now that the trace is on the GPU.
bool RealtimeRayTracer::render_and_blend_on_gpu()
{
    if (!is_active || !use_cuda || !cuda_available || cuda_scene_tris == 0)
        return false;
    if (!gpu_composite)
        return false;

    int fw = 0, fh = 0;
    unsigned char *dev_fb = cudaGetDeviceFramebuffer(&fw, &fh);
    if (!dev_fb || fw <= 0 || fh <= 0)
        return false;

    auto start_time = std::chrono::high_resolution_clock::now();

    if (!render_frame_gpu(false))
        return false;

    float b = blend_strength;
    if (quality_level == 1)
        b *= 0.6f;
    cudaRTBlendToFramebuffer(dev_fb, fw, fh, b);

    // the whole image is done, so there is no tile cursor to advance
    current_tile_x = 0;
    current_tile_y = total_tiles_y;

    auto end = std::chrono::high_resolution_clock::now();
    update_performance_stats(
        std::chrono::duration<float, std::milli>(end - start_time).count());
    return true;
}

void RealtimeRayTracer::toggle_cuda()
{
    if (!cuda_available)
    {
        std::cout << "Ray tracer CUDA not available" << std::endl;
        return;
    }
    use_cuda = !use_cuda;
    // the CPU tracer needs a BVH, which the GPU path skips building
    if (!use_cuda && !bvh_world && !world.objects.empty())
        bvh_world = world.build_bvh();
    reset_tiles();
    std::cout << "Ray tracer CUDA: " << (use_cuda ? "ENABLED" : "DISABLED") << std::endl;
}

void RealtimeRayTracer::render_one_tile()
{
    if (!is_active)
        return;

    auto start_time = std::chrono::high_resolution_clock::now();

    if (use_cuda && cuda_available && cuda_scene_tris > 0)
    {
        if (render_frame_gpu(true))
        {
            // the whole image is done, so there is no tile cursor to advance
            current_tile_x = 0;
            current_tile_y = total_tiles_y;

            auto end = std::chrono::high_resolution_clock::now();
            update_performance_stats(
                std::chrono::duration<float, std::milli>(end - start_time).count());
            return;
        }
    }

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

void RealtimeRayTracer::blend_with_framebuffer(TGAImage &main_framebuffer)
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

void RealtimeRayTracer::reset_tiles()
{
    current_tile_x = 0;
    current_tile_y = 0;
    rt_framebuffer.clear(); // clear previous ray traced image
    if (show_progress_overlay)
    {
        progress_overlay.clear(); 
    }
    std::cout << "Ray trace tiles reset" << std::endl;
}

void RealtimeRayTracer::mark_scene_dirty()
{
    world_needs_update = true;
    reset_tiles(); // Start over when scene changes
    restart_accumulation();
}

void RealtimeRayTracer::print_detailed_status() const
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

void RealtimeRayTracer::print_status() const
{
    if (is_active)
    {
        float progress = float(current_tile_y * total_tiles_x + current_tile_x) /
                         float(total_tiles_x * total_tiles_y) * 100.0f;
        std::cout << "RT Progress: " << std::fixed << std::setprecision(1)
                  << progress << "% (Quality: " << quality_level << "/4)" << std::endl;
    }
}

// =============================================================================
// PRIVATE METHODS
// =============================================================================

void RealtimeRayTracer::update_quality_settings()
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

void RealtimeRayTracer::ray_trace_tile_enhanced(int start_x, int start_y, int end_x, int end_y)
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
                // anti alialisicg strat 
                float offset_x = (sample > 0) ? (random_double() - 0.5f) : 0.0f;
                float offset_y = (sample > 0) ? (random_double() - 0.5f) : 0.0f;

                auto pixel_sample = pixel00_loc +
                                    ((x + offset_x) * pixel_delta_u) +
                                    ((y + offset_y) * pixel_delta_v);

                rt_ray r(center, pixel_sample - center);
                pixel_color += ray_color(r, rt_cam.max_depth);
            }

            pixel_color = pixel_color / rt_cam.samples_per_pixel;

            TGAColor tga_color = ColorConversion::rt_color_to_tga(pixel_color);
            rt_framebuffer.set(x, y, tga_color);
        }
    }
}

void RealtimeRayTracer::update_progress_overlay(int start_x, int start_y, int end_x, int end_y)
{
    // Map RT coordinates to main framebuffer coordinates
    int main_width = progress_overlay.get_width();
    int main_height = progress_overlay.get_height();

    int main_start_x = (start_x * main_width) / rt_width;
    int main_start_y = (start_y * main_height) / rt_height;
    int main_end_x = (end_x * main_width) / rt_width;
    int main_end_y = (end_y * main_height) / rt_height;

    TGAColor overlay_color(0, 50, 0); 

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

void RealtimeRayTracer::advance_tile()
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

void RealtimeRayTracer::update_performance_stats(float tile_time)
{
    // update moving average of frame times
    performance_samples++;
    float alpha = 1.0f / std::min(60, performance_samples);
    average_frame_time = average_frame_time * (1.0f - alpha) + tile_time * alpha;
}

void RealtimeRayTracer::adjust_quality_based_on_performance()
{

    const float target_tile_time = 2.0f;

    if (average_frame_time > target_tile_time * 2.0f && quality_level > 1)
    {
        // too slow, decrease quality
        decrease_quality();
    }
    else if (average_frame_time < target_tile_time * 0.5f && quality_level < 4)
    {
        // fast enough
        increase_quality();
    }
}

rt_color RealtimeRayTracer::ray_color(const rt_ray &r, int depth) const
{
    if (depth <= 0)
        return rt_color(0, 0, 0);

    rt_hit_record rec;
    // Use BVH for faster intersection testing
    if (!bvh_world)
        return rt_color(0, 0, 0);
    if (bvh_world->hit(r, 0.001, rt_infinity, rec))
    {
        rt_ray scattered;
        rt_color attenuation;
        if (rec.mat->scatter(r, rec.p, rec.normal, rec.front_face, attenuation, scattered))
            return attenuation * ray_color(scattered, depth - 1);
        return rt_color(0, 0, 0);
    }

    // black background
    return rt_color(0, 0, 0);
}
