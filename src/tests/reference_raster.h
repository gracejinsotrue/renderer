#ifndef __REFERENCE_RASTER_H__
#define __REFERENCE_RASTER_H__

// The CPU rasterizer, kept as the reference the GPU kernels are checked
// against. Nothing in src/ links it. It lives here so test_cull and
// test_shaded have an independent implementation to disagree with: a
// differential test whose reference is derived from the kernel it checks
// agrees by construction and catches nothing.

#include <vector>
#include "geometry.h"
#include "tgaimage.h"
#include "transform.h"

// sized by whoever renders a shadow pass; index it with these, never with the
// colour target's dimensions, which need not match.
extern std::vector<float> shadowbuffer;
extern int shadowbuffer_w, shadowbuffer_h;

// clears to -inf: the map keeps the surface nearest the light, and depth grows
// toward the viewer here, so nearest means largest.
void resizeShadowBuffer(int w, int h);
void clearShadowBuffer();

struct IShader
{
    virtual ~IShader();
    virtual Vec4f vertex(int iface, int nthvert) = 0;
    virtual bool fragment(Vec3f screen_bar, Vec3f persp_bar, TGAColor &color) = 0;
};

void triangle(Vec4f *pts, IShader &shader, TGAImage &image, TGAImage &zbuffer);

#endif //__REFERENCE_RASTER_H__
