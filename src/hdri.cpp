#include "hdri.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

namespace
{

// RGBE packs one shared exponent across the three channels. The bias is 128,
// and the mantissas are 8-bit fractions rather than integers, which is the
// extra -8.
void rgbe_to_float(const unsigned char *rgbe, float *out)
{
    if (rgbe[3] == 0)
    {
        out[0] = out[1] = out[2] = 0.f;
        return;
    }
    float f = ldexpf(1.0f, (int)rgbe[3] - (128 + 8));
    out[0] = rgbe[0] * f;
    out[1] = rgbe[1] * f;
    out[2] = rgbe[2] * f;
}

bool read_whole_file(const std::string &path, std::vector<unsigned char> &buf)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        return false;
    std::streamoff size = in.tellg();
    if (size <= 0)
        return false;
    in.seekg(0, std::ios::beg);
    buf.resize((size_t)size);
    return (bool)in.read((char *)buf.data(), size);
}

// Consumes through the newline and hands back the line without it. Radiance
// headers are ASCII, but the pixel data that follows is not, so this works on
// the raw buffer rather than reopening the file in text mode.
bool next_line(const std::vector<unsigned char> &buf, size_t &p, std::string &line)
{
    if (p >= buf.size())
        return false;
    size_t start = p;
    while (p < buf.size() && buf[p] != '\n')
        p++;
    size_t end = p;
    if (end > start && buf[end - 1] == '\r')
        end--;
    line.assign((const char *)buf.data() + start, end - start);
    if (p < buf.size())
        p++;
    return true;
}

// One scanline of RLE, decoded component-plane at a time into interleaved
// RGBE. Returns false on any run that would overflow the row, which is the
// only corruption that turns into an out-of-bounds write rather than a
// visibly wrong pixel.
bool decode_rle_scanline(const std::vector<unsigned char> &buf, size_t &p,
                         int width, unsigned char *scan)
{
    for (int c = 0; c < 4; c++)
    {
        int x = 0;
        while (x < width)
        {
            if (p >= buf.size())
                return false;
            int count = buf[p++];
            if (count > 128)
            {
                int run = count - 128;
                if (p >= buf.size() || x + run > width)
                    return false;
                unsigned char val = buf[p++];
                for (int i = 0; i < run; i++)
                    scan[(size_t)(x++) * 4 + c] = val;
            }
            else
            {
                if (count == 0 || p + count > buf.size() || x + count > width)
                    return false;
                for (int i = 0; i < count; i++)
                    scan[(size_t)(x++) * 4 + c] = buf[p++];
            }
        }
    }
    return true;
}

} // namespace

bool load_hdr(const std::string &filename, HDRImage &out)
{
    std::vector<unsigned char> buf;
    if (!read_whole_file(filename, buf))
    {
        std::cerr << "hdri: cannot read " << filename << std::endl;
        return false;
    }

    size_t p = 0;
    std::string line;
    if (!next_line(buf, p, line) || line.compare(0, 2, "#?") != 0)
    {
        std::cerr << "hdri: " << filename << " is not a Radiance file" << std::endl;
        return false;
    }

    bool rgbe_format = false;
    while (next_line(buf, p, line))
    {
        if (line.empty())
            break;
        if (line.rfind("FORMAT=", 0) == 0)
            rgbe_format = (line == "FORMAT=32-bit_rle_rgbe");
    }
    if (!rgbe_format)
    {
        std::cerr << "hdri: " << filename
                  << " is not 32-bit_rle_rgbe (XYZE is not decoded)" << std::endl;
        return false;
    }

    int w = 0, h = 0;
    if (!next_line(buf, p, line) ||
        sscanf(line.c_str(), "-Y %d +X %d", &h, &w) != 2 || w <= 0 || h <= 0)
    {
        std::cerr << "hdri: unsupported resolution line \"" << line
                  << "\" in " << filename << " (only -Y H +X W)" << std::endl;
        return false;
    }

    out.width = w;
    out.height = h;
    out.pixels.assign((size_t)w * h * 4, 0.f);
    std::vector<unsigned char> scan((size_t)w * 4);

    for (int y = 0; y < h; y++)
    {
        if (p + 4 > buf.size())
        {
            std::cerr << "hdri: " << filename << " ends at row " << y
                      << " of " << h << std::endl;
            out = HDRImage();
            return false;
        }

        bool rle = buf[p] == 2 && buf[p + 1] == 2 &&
                   ((buf[p + 2] << 8) | buf[p + 3]) == w && w >= 8 && w < 32768;
        if (rle)
        {
            p += 4;
            if (!decode_rle_scanline(buf, p, w, scan.data()))
            {
                std::cerr << "hdri: corrupt run-length data at row " << y
                          << " of " << filename << std::endl;
                out = HDRImage();
                return false;
            }
        }
        else
        {
            // Not an RLE marker, so those four bytes are already the first
            // pixel. A file using the old (1,1,1,count) repeat would decode
            // as garbage here rather than fail; nothing writes them any more.
            if (p + (size_t)w * 4 > buf.size())
            {
                std::cerr << "hdri: " << filename << " is short at row " << y << std::endl;
                out = HDRImage();
                return false;
            }
            memcpy(scan.data(), buf.data() + p, (size_t)w * 4);
            p += (size_t)w * 4;
        }

        float *row = out.pixels.data() + (size_t)y * w * 4;
        for (int x = 0; x < w; x++)
        {
            rgbe_to_float(&scan[(size_t)x * 4], row + (size_t)x * 4);
            row[(size_t)x * 4 + 3] = 1.f;
        }
    }

    std::cout << "Loaded environment: " << filename << " (" << w << "x" << h << ")"
              << std::endl;
    return true;
}
