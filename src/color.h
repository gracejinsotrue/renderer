#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "vec3.h"

using color = vec3;
// literally just raise to power of 1/gamma, or square root of linear component
inline double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}
void write_color(std::ostream &out, const color &pixel_color)
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // apply gamma correction
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // simple clamping
    r = std::max(0.0, std::min(0.999, r));
    g = std::max(0.0, std::min(0.999, g));
    b = std::max(0.0, std::min(0.999, b));

    // Convert to byte values
    int rbyte = int(256 * r);
    int gbyte = int(256 * g);
    int bbyte = int(256 * b);

    // write out the pixel color components
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif