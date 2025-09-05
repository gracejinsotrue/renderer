#ifndef RAY_TRACER_INTEGRATION_H
#define RAY_TRACER_INTEGRATION_H

#include "unified_math.h"
#include "Scene.h"
#include "model.h"
#include <vector>
#include <memory>
#include <fstream>

// =============================================================================
// RAY TRACER MATERIALS
// =============================================================================

class rt_material
{
public:
    virtual ~rt_material() = default;
    virtual bool scatter(const rt_ray &r_in, const rt_vec3 &hit_point, const rt_vec3 &normal,
                         rt_color &attenuation, rt_ray &scattered) const = 0;
};

class rt_lambertian : public rt_material
{
public:
    rt_color albedo;

    rt_lambertian(const rt_color &a) : albedo(a) {}

    bool scatter(const rt_ray &r_in, const rt_vec3 &hit_point, const rt_vec3 &normal,
                 rt_color &attenuation, rt_ray &scattered) const override
    {
        rt_vec3 scatter_direction = normal + random_unit_vector();

        if (scatter_direction.near_zero())
            scatter_direction = normal;

        scattered = rt_ray(hit_point, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

// =============================================================================
// RAY TRACER GEOMETRY
// =============================================================================

struct rt_hit_record
{
    rt_point3 p;
    rt_vec3 normal;
    std::shared_ptr<rt_material> mat;
    double t;
    bool front_face;

    void set_face_normal(const rt_ray &r, const rt_vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class rt_hittable
{
public:
    virtual ~rt_hittable() = default;
    virtual bool hit(const rt_ray &r, double t_min, double t_max, rt_hit_record &rec) const = 0;
};

class rt_triangle : public rt_hittable
{
public:
    rt_point3 v0, v1, v2;
    rt_vec3 edge1, edge2, normal;
    std::shared_ptr<rt_material> mat;

    rt_triangle(const rt_point3 &vertex0, const rt_point3 &vertex1, const rt_point3 &vertex2,
                std::shared_ptr<rt_material> material)
        : v0(vertex0), v1(vertex1), v2(vertex2), mat(material)
    {
        edge1 = v1 - v0;
        edge2 = v2 - v0;
        normal = unit_vector(cross(edge1, edge2));
    }

    bool hit(const rt_ray &r, double t_min, double t_max, rt_hit_record &rec) const override
    {
        // Möller-Trumbore intersection algorithm
        const double EPSILON = 1e-8;
        rt_vec3 h = cross(r.direction(), edge2);
        double a = dot(edge1, h);

        if (a > -EPSILON && a < EPSILON)
            return false; // Ray is parallel to triangle

        double f = 1.0 / a;
        rt_vec3 s = r.origin() - v0;
        double u = f * dot(s, h);

        if (u < 0.0 || u > 1.0)
            return false;

        rt_vec3 q = cross(s, edge1);
        double v = f * dot(r.direction(), q);

        if (v < 0.0 || u + v > 1.0)
            return false;

        double t = f * dot(edge2, q);

        if (t < t_min || t > t_max)
            return false;

        rec.t = t;
        rec.p = r.at(t);
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }
};

class rt_sphere : public rt_hittable
{
public:
    rt_point3 center;
    double radius;
    std::shared_ptr<rt_material> mat;

    rt_sphere(const rt_point3 &c, double r, std::shared_ptr<rt_material> material)
        : center(c), radius(r), mat(material) {}

    bool hit(const rt_ray &r, double t_min, double t_max, rt_hit_record &rec) const override
    {
        rt_vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);
        auto root = (h - sqrtd) / a;
        if (root < t_min || root > t_max)
        {
            root = (h + sqrtd) / a;
            if (root < t_min || root > t_max)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rt_vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }
};

class rt_hittable_list : public rt_hittable
{
public:
    std::vector<std::shared_ptr<rt_hittable>> objects;

    void clear() { objects.clear(); }
    void add(std::shared_ptr<rt_hittable> object) { objects.push_back(object); }

    bool hit(const rt_ray &r, double t_min, double t_max, rt_hit_record &rec) const override
    {
        rt_hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = t_max;

        for (const auto &object : objects)
        {
            if (object->hit(r, t_min, closest_so_far, temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

// =============================================================================
// RAY TRACER CAMERA
// =============================================================================

class rt_camera
{
public:
    double aspect_ratio;
    int image_width;
    int samples_per_pixel;
    int max_depth;
    double vfov;

    rt_point3 lookfrom = rt_point3(0, 0, 0);
    rt_point3 lookat = rt_point3(0, 0, -1);
    rt_vec3 vup = rt_vec3(0, 1, 0);

    void render_to_file(const rt_hittable &world, const std::string &filename)
    {
        initialize();

        std::ofstream file(filename);
        file << "P3\n"
             << image_width << ' ' << image_height << "\n255\n";

        std::cout << "Ray tracing to " << filename << "..." << std::endl;

        for (int j = 0; j < image_height; j++)
        {
            std::cout << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++)
            {
                rt_color pixel_color(0, 0, 0);
                for (int sample = 0; sample < samples_per_pixel; sample++)
                {
                    rt_ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }
                write_color(file, pixel_samples_scale * pixel_color);
            }
        }

        std::cout << "\rRay tracing complete! Saved to " << filename << std::endl;
    }

private:
    int image_height;
    double pixel_samples_scale;
    rt_point3 center;
    rt_point3 pixel00_loc;
    rt_vec3 pixel_delta_u;
    rt_vec3 pixel_delta_v;
    rt_vec3 u, v, w;

    void initialize()
    {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta / 2);
        auto viewport_height = 2 * h;
        auto viewport_width = viewport_height * (double(image_width) / image_height);

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        rt_vec3 viewport_u = viewport_width * u;
        rt_vec3 viewport_v = viewport_height * -v;

        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        auto viewport_upper_left = center - w - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    rt_ray get_ray(int i, int j) const
    {
        auto offset = sample_square();
        auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
        return rt_ray(center, pixel_sample - center);
    }

    rt_vec3 sample_square() const
    {
        return rt_vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }

    rt_color ray_color(const rt_ray &r, int depth, const rt_hittable &world) const
    {
        if (depth <= 0)
            return rt_color(0, 0, 0);

        rt_hit_record rec;
        if (world.hit(r, 0.001, rt_infinity, rec))
        {
            rt_ray scattered;
            rt_color attenuation;
            if (rec.mat->scatter(r, rec.p, rec.normal, attenuation, scattered))
                return attenuation * ray_color(scattered, depth - 1, world);
            return rt_color(0, 0, 0);
        }
        // return rt_color(0, 0, 0);

        // gradient stuff

        rt_vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - a) * rt_color(0.0, 0.0, 0.0) + a * rt_color(0.8, 0.8, 0.8);
    }

    void write_color(std::ostream &out, const rt_color &pixel_color) const
    {
        auto r = pixel_color.x();
        auto g = pixel_color.y();
        auto b = pixel_color.z();

        // Apply gamma correction
        r = std::sqrt(r);
        g = std::sqrt(g);
        b = std::sqrt(b);

        // Clamp and convert to byte values
        int rbyte = int(256 * std::max(0.0, std::min(0.999, r)));
        int gbyte = int(256 * std::max(0.0, std::min(0.999, g)));
        int bbyte = int(256 * std::max(0.0, std::min(0.999, b)));

        out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
    }
};

// =============================================================================
// SCENE TO RAY TRACER CONVERTER
// =============================================================================

class SceneToRayTracer
{
public:
    static rt_hittable_list convert_scene(Scene &scene)
    {
        rt_hittable_list world;

        // get all mesh nodes from the scene
        std::vector<SceneNode *> mesh_nodes;
        scene.getAllMeshNodes(mesh_nodes);

        std::cout << "Converting " << mesh_nodes.size() << " mesh nodes to ray tracer format..." << std::endl;

        // Convert each mesh to triangles
        for (SceneNode *node : mesh_nodes)
        {
            if (node->hasModel())
            {
                convert_model_to_triangles(node, world);
            }
        }

        // add a ground plane
        // auto ground_material = std::make_shared<rt_lambertian>(rt_color(0.5, 0.5, 0.5));
        // world.add(std::make_shared<rt_sphere>(rt_point3(0, -1000, 0), 1000, ground_material));

        std::cout << "Scene conversion complete!" << std::endl;
        return world;
    }

private:
    static void convert_model_to_triangles(SceneNode *node, rt_hittable_list &world)
    {
        Model *model = node->model;
        if (!model)
            return;

        // get world transform matrix
        Matrix world_transform = node->getWorldMatrix();

        // create a simple lambertian material (we'll improve this later)
        auto material = std::make_shared<rt_lambertian>(rt_color(0.7, 0.3, 0.3));

        // Convert each face to a triangle
        for (int i = 0; i < model->nfaces(); i++)
        {
            // Get the three vertices of the face
            Vec3f v0_local = model->vert(i, 0);
            Vec3f v1_local = model->vert(i, 1);
            Vec3f v2_local = model->vert(i, 2);

            // Transform vertices to world space
            Vec3f v0_world = transform_point(world_transform, v0_local);
            Vec3f v1_world = transform_point(world_transform, v1_local);
            Vec3f v2_world = transform_point(world_transform, v2_local);

            // convert to ray tracer format
            rt_point3 rt_v0 = raster_to_point(v0_world);
            rt_point3 rt_v1 = raster_to_point(v1_world);
            rt_point3 rt_v2 = raster_to_point(v2_world);

            // create triangle and add to world
            auto triangle = std::make_shared<rt_triangle>(rt_v0, rt_v1, rt_v2, material);
            world.add(triangle);
        }

        std::cout << "  Converted " << node->name << " (" << model->nfaces() << " triangles)" << std::endl;
    }

    static Vec3f transform_point(const Matrix &transform, const Vec3f &point)
    {
        // convert point to homogeneous coordinates
        Vec4f homogeneous_point = embed<4>(point);
        homogeneous_point[3] = 1.0f; // w = 1 for points

        // apply transformation
        Vec4f transformed = transform * homogeneous_point;

        // Convert back to 3D via perspective divide
        if (transformed[3] != 0.0f)
        {
            return Vec3f(transformed[0] / transformed[3],
                         transformed[1] / transformed[3],
                         transformed[2] / transformed[3]);
        }

        return Vec3f(transformed[0], transformed[1], transformed[2]);
    }
};

// =============================================================================
// RAY TRACER INTERFACE FOR ENGINE
// =============================================================================

class RayTracerInterface
{
public:
    static void ray_trace_scene(Scene &scene)
    {
        std::cout << "\n=== STARTING RAY TRACE ===" << std::endl;

        // convert scene to ray tracer format
        rt_hittable_list world = SceneToRayTracer::convert_scene(scene);

        // QUALITY SETTINGS
        rt_camera cam;
        cam.aspect_ratio = 1.0;
        cam.image_width = 1200;     //  RESOLUTION
        cam.samples_per_pixel = 16; // MORE SAMPL E= LESS GRAINY BUT LONGER
        cam.max_depth = 10;         // BOUNCES FOR LIGTHING (MORE IS SLOWER BUT BETTER LIGHTING)
        cam.vfov = 20;

        // convert rasterizer camera to ray tracer camera
        Camera &raster_cam = scene.camera;
        cam.lookfrom = raster_to_point(raster_cam.position);
        cam.lookat = raster_to_point(raster_cam.target);
        cam.vup = raster_to_rt(raster_cam.up);

        std::cout << "ray tracing settings:" << std::endl;
        std::cout << "  resolution: " << cam.image_width << "x" << cam.image_width << std::endl;
        std::cout << "  samples per pixel: " << cam.samples_per_pixel << std::endl;
        std::cout << "  max depth: " << cam.max_depth << std::endl;
        std::cout << "  camera position: (" << cam.lookfrom.x() << ", " << cam.lookfrom.y() << ", " << cam.lookfrom.z() << ")" << std::endl;

        // writing to ppm file (need to remove)
        cam.render_to_file(world, "raytraced_output.ppm");

        std::cout << "=== RAY TRACE COMPLETE ===" << std::endl;
        std::cout << "Output saved to: raytraced_output.ppm" << std::endl;
    }
};

#endif // RAY_TRACER_INTEGRATION_H