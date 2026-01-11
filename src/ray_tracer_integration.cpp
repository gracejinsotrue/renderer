#include "ray_tracer_integration.h"
#include "bvh.h"
#include <algorithm>

// =============================================================================
// BVH IMPLEMENTATION
// =============================================================================

rt_aabb rt_bvh::compute_bounds(std::shared_ptr<rt_hittable> object)
{
    // Try to cast to known types
    auto triangle = std::dynamic_pointer_cast<rt_triangle>(object);
    if (triangle)
    {
        // compute the bounds for the triagnel
        double min_x = std::min({triangle->v0.x(), triangle->v1.x(), triangle->v2.x()});
        double min_y = std::min({triangle->v0.y(), triangle->v1.y(), triangle->v2.y()});
        double min_z = std::min({triangle->v0.z(), triangle->v1.z(), triangle->v2.z()});

        double max_x = std::max({triangle->v0.x(), triangle->v1.x(), triangle->v2.x()});
        double max_y = std::max({triangle->v0.y(), triangle->v1.y(), triangle->v2.y()});
        double max_z = std::max({triangle->v0.z(), triangle->v1.z(), triangle->v2.z()});

        rt_point3 min_point(min_x, min_y, min_z);
        rt_point3 max_point(max_x, max_y, max_z);

        // avoid zero-area boxes this is just good practice 
        const double padding = 1e-6;
        for (int i = 0; i < 3; i++)
        {
            if (max_point[i] - min_point[i] < padding)
            {
                min_point[i] -= padding / 2;
                max_point[i] += padding / 2;
            }
        }

        return rt_aabb(min_point, max_point);
    }

    auto sphere = std::dynamic_pointer_cast<rt_sphere>(object);
    if (sphere)
    {
        rt_vec3 radius_vec(sphere->radius, sphere->radius, sphere->radius);
        return rt_aabb(sphere->center - radius_vec, sphere->center + radius_vec);
    }
    return rt_aabb(rt_point3(0, 0, 0), rt_point3(0, 0, 0));
}

rt_bvh::rt_bvh(std::vector<std::shared_ptr<rt_hittable>> &objects)
{
    if (!objects.empty())
    {
        root = build_tree(objects, 0, objects.size());
    }
}

bool rt_bvh::hit(const rt_ray &r, double t_min, double t_max, rt_hit_record &rec) const
{
    if (!root)
        return false;
    return hit_node(root, r, t_min, t_max, rec);
}

std::shared_ptr<rt_bvh_node> rt_bvh::build_tree(
    std::vector<std::shared_ptr<rt_hittable>> &objects,
    int start, int end)
{
    auto node = std::make_shared<rt_bvh_node>();

    int object_span = end - start;

    if (object_span == 1)
    {
        // then it is a leaf node
        node->leaf_object = objects[start];
        node->bounds = compute_bounds(objects[start]);
        return node;
    }

    // compute bounding box for all objects in this range
    rt_aabb bounds = compute_bounds(objects[start]);
    for (int i = start + 1; i < end; i++)
    {
        bounds = rt_aabb::combine(bounds, compute_bounds(objects[i]));
    }
    node->bounds = bounds;

    // find the split axis (longest axis of bounding box)
    int axis = bounds.longest_axis();

    // Sort objects along the split axis
    std::sort(objects.begin() + start, objects.begin() + end,
              [axis](const std::shared_ptr<rt_hittable> &a,
                     const std::shared_ptr<rt_hittable> &b)
              {
                  rt_aabb box_a = compute_bounds(a);
                  rt_aabb box_b = compute_bounds(b);
                  return box_a.min[axis] < box_b.min[axis];
              });

    // recursively build left and right subtree
    int mid = start + object_span / 2;
    node->left = build_tree(objects, start, mid);
    node->right = build_tree(objects, mid, end);

    return node;
}

bool rt_bvh::hit_node(std::shared_ptr<rt_bvh_node> node, const rt_ray &r,
                      double t_min, double t_max, rt_hit_record &rec) const
{
    if (!node || !node->bounds.hit(r, t_min, t_max))
        return false;

    // if this is a leaf node, test intersection with the object
    if (node->leaf_object)
    {
        return node->leaf_object->hit(r, t_min, t_max, rec);
    }

    // we recusrively test both subtree
    bool hit_left = false;
    bool hit_right = false;
    double closest_t = t_max;

    if (node->left)
    {
        rt_hit_record temp_rec;
        if (hit_node(node->left, r, t_min, closest_t, temp_rec))
        {
            hit_left = true;
            closest_t = temp_rec.t;
            rec = temp_rec;
        }
    }

    if (node->right)
    {
        rt_hit_record temp_rec;
        if (hit_node(node->right, r, t_min, closest_t, temp_rec))
        {
            hit_right = true;
            rec = temp_rec;
        }
    }

    return hit_left || hit_right;
}

// =============================================================================
// RT_HITTABLE_LIST 
// =============================================================================

std::shared_ptr<rt_bvh> rt_hittable_list::build_bvh()
{
    return std::make_shared<rt_bvh>(objects);
}
