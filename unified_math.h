#ifndef UNIFIED_MATH_H
#define UNIFIED_MATH_H

// Include rasterizer math first
#include "geometry.h" // rasterizer math (Vec3f, Matrix, etc.)
#include "tgaimage.h" // for TGAColor

// ray tracer headers
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdlib>

// ray tracer constants and utilities
const double rt_infinity = std::numeric_limits<double>::infinity();
const double rt_pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees)
{
    return degrees * rt_pi / 180.0;
}

inline double random_double()
{
    return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max)
{
    return min + (max - min) * random_double();
}

// simple vec3 class for ray tracer
class rt_vec3
{
public:
    double e[3];

    rt_vec3() : e{0, 0, 0} {}
    rt_vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    rt_vec3 operator-() const { return rt_vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double &operator[](int i) { return e[i]; }

    rt_vec3 &operator+=(const rt_vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    rt_vec3 &operator*=(double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    double length() const { return std::sqrt(length_squared()); }
    double length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    bool near_zero() const
    {
        auto s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }

    static rt_vec3 random()
    {
        return rt_vec3(random_double(), random_double(), random_double());
    }

    static rt_vec3 random(double min, double max)
    {
        return rt_vec3(random_double(min, max), random_double(min, max), random_double(min, max));
    }
};

using rt_point3 = rt_vec3;
using rt_color = rt_vec3;

// Vector operations
inline rt_vec3 operator+(const rt_vec3 &u, const rt_vec3 &v)
{
    return rt_vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline rt_vec3 operator-(const rt_vec3 &u, const rt_vec3 &v)
{
    return rt_vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline rt_vec3 operator*(const rt_vec3 &u, const rt_vec3 &v)
{
    return rt_vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline rt_vec3 operator*(double t, const rt_vec3 &v)
{
    return rt_vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline rt_vec3 operator*(const rt_vec3 &v, double t)
{
    return t * v;
}

inline rt_vec3 operator/(const rt_vec3 &v, double t)
{
    return (1 / t) * v;
}

inline double dot(const rt_vec3 &u, const rt_vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline rt_vec3 cross(const rt_vec3 &u, const rt_vec3 &v)
{
    return rt_vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                   u.e[2] * v.e[0] - u.e[0] * v.e[2],
                   u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline rt_vec3 unit_vector(const rt_vec3 &v)
{
    return v / v.length();
}

inline rt_vec3 random_unit_vector()
{
    while (true)
    {
        auto p = rt_vec3::random(-1, 1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

inline rt_vec3 random_on_hemisphere(const rt_vec3 &normal)
{
    rt_vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

inline rt_vec3 reflect(const rt_vec3 &v, const rt_vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

inline rt_vec3 refract(const rt_vec3 &uv, const rt_vec3 &n, double etai_over_etat)
{
    auto cos_theta = std::fmin(dot(-uv, n), 1.0);
    rt_vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    rt_vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// Simple ray class
class rt_ray
{
public:
    rt_ray() {}
    rt_ray(const rt_point3 &origin, const rt_vec3 &direction) : orig(origin), dir(direction) {}

    const rt_point3 &origin() const { return orig; }
    const rt_vec3 &direction() const { return dir; }

    rt_point3 at(double t) const
    {
        return orig + t * dir;
    }

private:
    rt_point3 orig;
    rt_vec3 dir;
};

// =============================================================================
// CONVERSION FUNCTIONS BETWEEN MATH SYSTEMS
// =============================================================================

// convert ray tracer rt_vec3 -> rasterizer Vec3f
inline Vec3f rt_to_raster(const rt_vec3 &v)
{
    return Vec3f(static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z()));
}

// convert rasterizer Vec3f -> ray tracer rt_vec3
inline rt_vec3 raster_to_rt(const Vec3f &v)
{
    return rt_vec3(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
}

// ray tracer rt_point3 ->rasterizer Vec3f
inline Vec3f point_to_raster(const rt_point3 &p)
{
    return Vec3f(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
}

// rasterizer Vec3f -> ray tracer rt_point3
inline rt_point3 raster_to_point(const Vec3f &v)
{
    return rt_point3(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
}

// =============================================================================
// UNIFIED TYPES
// =============================================================================

using UnifiedVec3 = Vec3f;
using UnifiedPoint3 = Vec3f;
using UnifiedMatrix = Matrix;

// =============================================================================
// UNIFIED UTILITY FUNCTIONS
// =============================================================================

namespace UnifiedMath
{

    inline UnifiedVec3 normalize(const UnifiedVec3 &v)
    {
        UnifiedVec3 result = v;
        return result.normalize();
    }

    inline float dot(const UnifiedVec3 &a, const UnifiedVec3 &b)
    {
        return a * b;
    }

    inline UnifiedVec3 cross(const UnifiedVec3 &a, const UnifiedVec3 &b)
    {
        return ::cross(a, b);
    }

    inline float length(const UnifiedVec3 &v)
    {
        UnifiedVec3 temp = v;
        return temp.norm();
    }

    inline float length_squared(const UnifiedVec3 &v)
    {
        return v * v;
    }

    inline UnifiedVec3 random_unit_vector()
    {
        rt_vec3 rt_random = ::random_unit_vector();
        return rt_to_raster(rt_random);
    }

    inline UnifiedVec3 random_in_hemisphere(const UnifiedVec3 &normal)
    {
        rt_vec3 rt_normal = raster_to_rt(normal);
        rt_vec3 rt_random = ::random_on_hemisphere(rt_normal);
        return rt_to_raster(rt_random);
    }

    inline UnifiedVec3 reflect(const UnifiedVec3 &v, const UnifiedVec3 &n)
    {
        rt_vec3 rt_v = raster_to_rt(v);
        rt_vec3 rt_n = raster_to_rt(n);
        rt_vec3 rt_result = ::reflect(rt_v, rt_n);
        return rt_to_raster(rt_result);
    }

    inline UnifiedVec3 refract(const UnifiedVec3 &uv, const UnifiedVec3 &n, float etai_over_etat)
    {
        rt_vec3 rt_uv = raster_to_rt(uv);
        rt_vec3 rt_n = raster_to_rt(n);
        rt_vec3 rt_result = ::refract(rt_uv, rt_n, static_cast<double>(etai_over_etat));
        return rt_to_raster(rt_result);
    }

    inline float random_float()
    {
        return static_cast<float>(random_double());
    }

    inline float random_float(float min, float max)
    {
        return static_cast<float>(random_double(static_cast<double>(min), static_cast<double>(max)));
    }

    inline float degrees_to_radians_f(float degrees)
    {
        return degrees * 3.14159265359f / 180.0f;
    }
}

// =============================================================================
// UNIFIED RAY CLASS
// =============================================================================

class UnifiedRay
{
public:
    UnifiedVec3 origin;
    UnifiedVec3 direction;

    UnifiedRay() : origin(0, 0, 0), direction(0, 0, 1) {}
    UnifiedRay(const UnifiedVec3 &orig, const UnifiedVec3 &dir)
        : origin(orig), direction(dir) {}

    rt_ray to_rt_ray() const
    {
        return rt_ray(raster_to_point(origin), raster_to_rt(direction));
    }

    static UnifiedRay from_rt_ray(const rt_ray &r)
    {
        return UnifiedRay(point_to_raster(r.origin()), rt_to_raster(r.direction()));
    }

    UnifiedVec3 at(float t) const
    {
        return origin + direction * t;
    }
};

// =============================================================================
// COLOR CONVERSION
// =============================================================================

namespace ColorConversion
{

    inline double linear_to_gamma(double linear_component)
    {
        if (linear_component > 0)
            return std::sqrt(linear_component);
        return 0;
    }

    inline TGAColor rt_color_to_tga(const rt_color &c)
    {
        float r = static_cast<float>(linear_to_gamma(c.x()));
        float g = static_cast<float>(linear_to_gamma(c.y()));
        float b = static_cast<float>(linear_to_gamma(c.z()));

        int rbyte = static_cast<int>(std::max(0.0f, std::min(0.999f, r)) * 255);
        int gbyte = static_cast<int>(std::max(0.0f, std::min(0.999f, g)) * 255);
        int bbyte = static_cast<int>(std::max(0.0f, std::min(0.999f, b)) * 255);

        return TGAColor(rbyte, gbyte, bbyte);
    }

    inline rt_color tga_to_rt_color(const TGAColor &tga)
    {
        TGAColor &mutable_tga = const_cast<TGAColor &>(tga);
        double r = static_cast<double>(mutable_tga[2]) / 255.0;
        double g = static_cast<double>(mutable_tga[1]) / 255.0;
        double b = static_cast<double>(mutable_tga[0]) / 255.0;

        r = r * r;
        g = g * g;
        b = b * b;

        return rt_color(r, g, b);
    }

    inline UnifiedVec3 rt_color_to_vec3(const rt_color &c)
    {
        return UnifiedVec3(static_cast<float>(c.x()), static_cast<float>(c.y()), static_cast<float>(c.z()));
    }

    inline rt_color vec3_to_rt_color(const UnifiedVec3 &v)
    {
        return rt_color(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
    }
}

// =============================================================================
// TESTING UTILITIES
// =============================================================================

namespace UnifiedMathTest
{

    inline bool test_conversions()
    {
        Vec3f raster_vec(1.0f, 2.0f, 3.0f);
        rt_vec3 rt_vec = raster_to_rt(raster_vec);
        Vec3f converted_back = rt_to_raster(rt_vec);

        float epsilon = 1e-6f;
        bool test_passed =
            std::abs(raster_vec.x - converted_back.x) < epsilon &&
            std::abs(raster_vec.y - converted_back.y) < epsilon &&
            std::abs(raster_vec.z - converted_back.z) < epsilon;

        if (test_passed)
        {
            std::cout << "math conversion test passed" << std::endl;
        }
        else
        {
            std::cout << "math conversion test failed" << std::endl;
        }

        return test_passed;
    }

    inline void debug_print_vector(const Vec3f &v, const std::string &name)
    {
        std::cout << name << " (raster): (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
        rt_vec3 rt_v = raster_to_rt(v);
        std::cout << name << " (rt): (" << rt_v.x() << ", " << rt_v.y() << ", " << rt_v.z() << ")" << std::endl;
    }
}

#endif // UNIFIED_MATH_H