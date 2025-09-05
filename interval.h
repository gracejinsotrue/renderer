#ifndef INTERVAL_H
#define INTERVAL_H

#include <limits>

class interval
{
private:
    static constexpr double infinity = std::numeric_limits<double>::infinity();

public:
    double min, max;

    interval() : min(+infinity), max(-infinity) {} // default interval is empty

    interval(double min, double max) : min(min), max(max) {}

    double size() const
    {
        return max - min;
    }

    bool contains(double x) const
    {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const
    {
        return min < x && x < max;
    }

    double clamp(double x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

    static const interval empty, universe;
};

const interval interval::empty = interval(+interval::infinity, -interval::infinity);
const interval interval::universe = interval(-interval::infinity, +interval::infinity);

#endif