#ifndef __HDRI_H__
#define __HDRI_H__

#include <string>
#include <vector>

struct HDRImage
{
    int width = 0;
    int height = 0;

    std::vector<float> pixels;

    bool valid() const { return width > 0 && height > 0 && !pixels.empty(); }
};

bool load_hdr(const std::string &filename, HDRImage &out);

#endif //__HDRI_H__
