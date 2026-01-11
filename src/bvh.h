#ifndef BVH_H
#define BVH_H

#include "unified_math.h"
#include <memory>
#include <vector>
#include <algorithm>

// =============================================================================
// AABB (Axis-Aligned Bounding Box)
// =============================================================================

class rt_aabb
{
public:
    rt_point3 min, max;

    rt_aabb() {}

    rt_aabb(const rt_point3 &a, const rt_point3 &b)
        : min(a), max(b) {}

    // Get the bounding box of two AABBs
    static rt_aabb combine(const rt_aabb &box0, const rt_aabb &box1)
    {
        rt_point3 min_point(
            std::min(box0.min.x(), box1.min.x()),
            std::min(box0.min.y(), box1.min.y()),
            std::min(box0.min.z(), box1.min.z())
        );
        rt_point3 max_point(
            std::max(box0.max.x(), box1.max.x()),
            std::max(box0.max.y(), box1.max.y()),
            std::max(box0.max.z(), box1.max.z())
        );
        return rt_aabb(min_point, max_point);
    }

    // Check if ray intersects this bounding box
    bool hit(const rt_ray &r, double t_min, double t_max) const
    {
        for (int a = 0; a < 3; a++)
        {
            double t0 = std::min((min[a] - r.origin()[a]) / r.direction()[a],
                                 (max[a] - r.origin()[a]) / r.direction()[a]);
            double t1 = std::max((min[a] - r.origin()[a]) / r.direction()[a],
                                 (max[a] - r.origin()[a]) / r.direction()[a]);

            t_min = std::max(t0, t_min);
            t_max = std::min(t1, t_max);

            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    // Get the longest axis 
    int longest_axis() const
    {
        double x = max.x() - min.x();
        double y = max.y() - min.y();
        double z = max.z() - min.z();

        if (x > y && x > z)
            return 0;
        else if (y > z)
            return 1;
        else
            return 2;
    }

    double surface_area() const
    {
        double x = max.x() - min.x();
        double y = max.y() - min.y();
        double z = max.z() - min.z();
        return 2.0 * (x * y + y * z + z * x);
    }
};

// Forward declarations
class rt_hittable;
struct rt_hit_record;

// =============================================================================
// BVH NODE
// =============================================================================

struct rt_bvh_node
{
    rt_aabb bounds;
    std::shared_ptr<rt_bvh_node> left;
    std::shared_ptr<rt_bvh_node> right;
    std::shared_ptr<rt_hittable> leaf_object;

    rt_bvh_node() : left(nullptr), right(nullptr), leaf_object(nullptr) {}
};

#endif // BVH_H
