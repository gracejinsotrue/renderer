// CUDA path tracer.
//
// The CPU tracer in ray_tracer_integration.h keeps every triangle behind a
// shared_ptr<rt_hittable> and walks a BVH of shared_ptr nodes recursively, in
// double precision. None of that survives a port: this file holds the same
// scene as three flat float arrays, traverses the BVH iteratively with an
// explicit stack, and runs one thread per pixel.
//
// The scene is handed over as plain triangles. The BVH is built here on the
// host, straight into the flat layout the kernel reads, so no tree of pointers
// is ever constructed.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define RT_EPS 1e-8f
#define RT_TMIN 0.001f
#define RT_INF 1e30f

// Traversal stack depth. A median-split BVH over n triangles is about
// log2(n) deep; 64 leaves room for badly unbalanced trees without spilling.
#define RT_STACK 64

// Triangles are stored with edges precomputed, matching the Moller-Trumbore
// form the CPU rt_triangle uses.
struct RTTri
{
    float v0[3];
    float e1[3];
    float e2[3];
    float n[3];
    int mat;
};

// count > 0 marks a leaf holding triangles [first, first + count).
// count == 0 marks an interior node whose children are left and right.
struct RTNode
{
    float bmin[3];
    float bmax[3];
    int left;
    int right;
    int first;
    int count;
};

#define RT_MAT_LAMBERTIAN 0
#define RT_MAT_METAL 1
#define RT_MAT_DIELECTRIC 2

struct RTMat
{
    float albedo[3];
    int type;
    float fuzz;   // metal only: 0 is a perfect mirror
    float ior;    // dielectric only: refractive index
};

struct RTCam
{
    float origin[3];
    float pixel00[3];
    float du[3];
    float dv[3];
};

// Vertical gradient used for rays that escape the scene. This is the only
// light source: the only material is lambertian and nothing emits, so with a
// black sky every path terminates at black and the image is uniformly black.
// Defaults match rt_camera::ray_color in ray_tracer_integration.h.
struct RTSky
{
    float lo[3];
    float hi[3];
};

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float3 mk(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return mk(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return mk(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator-(float3 a) { return mk(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b)
{
    return mk(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 norm3(float3 v)
{
    float l = sqrtf(dot3(v, v));
    return (l > 0.f) ? v * (1.f / l) : v;
}
__device__ __forceinline__ float3 ld3(const float *p) { return mk(p[0], p[1], p[2]); }

// PCG. Seeded per pixel per sample, so a thread's sequence does not depend on
// what any other thread does and a render is reproducible for a given seed.
__device__ __forceinline__ unsigned int pcg(unsigned int &state)
{
    state = state * 747796405u + 2891336453u;
    unsigned int w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (w >> 22u) ^ w;
}

__device__ __forceinline__ float randf(unsigned int &state)
{
    // [0,1), matching random_double()
    return (pcg(state) >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float3 reflect3(float3 v, float3 n)
{
    return v - n * (2.f * dot3(v, n));
}

__device__ __forceinline__ float3 refract3(float3 uv, float3 n, float etai_over_etat)
{
    float cos_theta = fminf(dot3(-uv, n), 1.f);
    float3 perp = (uv + n * cos_theta) * etai_over_etat;
    float k = 1.f - dot3(perp, perp);
    float3 parallel = n * (-sqrtf(fabsf(k)));
    return perp + parallel;
}

// Schlick's approximation for angle-dependent reflectance
__device__ __forceinline__ float schlick(float cosine, float ri)
{
    float r0 = (1.f - ri) / (1.f + ri);
    r0 = r0 * r0;
    float m = 1.f - cosine;
    return r0 + (1.f - r0) * m * m * m * m * m;
}

// same rejection sampling as vec3.h random_unit_vector()
__device__ __forceinline__ float3 randUnitVec(unsigned int &state)
{
    for (int i = 0; i < 32; i++)
    {
        float3 p = mk(randf(state) * 2.f - 1.f, randf(state) * 2.f - 1.f, randf(state) * 2.f - 1.f);
        float ls = dot3(p, p);
        if (ls > 1e-30f && ls <= 1.f)
            return p * rsqrtf(ls);
    }
    return mk(0.f, 0.f, 1.f);
}

// ---------------------------------------------------------------------------
// intersection
// ---------------------------------------------------------------------------

// Slab test. inv_dir is precomputed once per ray; the IEEE infinities produced
// by a zero direction component behave correctly through the min/max.
__device__ __forceinline__ bool hitAABB(const RTNode &nd, float3 o, float3 inv, float tmax)
{
    float t0 = (nd.bmin[0] - o.x) * inv.x, t1 = (nd.bmax[0] - o.x) * inv.x;
    float tlo = fminf(t0, t1), thi = fmaxf(t0, t1);

    t0 = (nd.bmin[1] - o.y) * inv.y; t1 = (nd.bmax[1] - o.y) * inv.y;
    tlo = fmaxf(tlo, fminf(t0, t1)); thi = fminf(thi, fmaxf(t0, t1));

    t0 = (nd.bmin[2] - o.z) * inv.z; t1 = (nd.bmax[2] - o.z) * inv.z;
    tlo = fmaxf(tlo, fminf(t0, t1)); thi = fminf(thi, fmaxf(t0, t1));

    return thi >= fmaxf(tlo, RT_TMIN) && tlo < tmax;
}

struct Hit
{
    float t;
    float3 n;
    int mat;
};

__device__ bool traceRay(const RTTri *tris, const RTNode *nodes, int nnodes,
                         float3 o, float3 d, float tmax, Hit &hit)
{
    if (nnodes == 0)
        return false;

    float3 inv = mk(1.f / d.x, 1.f / d.y, 1.f / d.z);
    hit.t = tmax;
    bool found = false;

    int stack[RT_STACK];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0)
    {
        int ni = stack[--sp];
        const RTNode &nd = nodes[ni];
        if (!hitAABB(nd, o, inv, hit.t))
            continue;

        if (nd.count > 0)
        {
            for (int i = 0; i < nd.count; i++)
            {
                const RTTri &tr = tris[nd.first + i];
                float3 e1 = ld3(tr.e1), e2 = ld3(tr.e2);
                float3 h = cross3(d, e2);
                float a = dot3(e1, h);
                if (a > -RT_EPS && a < RT_EPS)
                    continue;

                float f = 1.f / a;
                float3 s = o - ld3(tr.v0);
                float u = f * dot3(s, h);
                if (u < 0.f || u > 1.f)
                    continue;

                float3 q = cross3(s, e1);
                float v = f * dot3(d, q);
                if (v < 0.f || u + v > 1.f)
                    continue;

                float t = f * dot3(e2, q);
                if (t < RT_TMIN || t > hit.t)
                    continue;

                hit.t = t;
                hit.n = ld3(tr.n);
                hit.mat = tr.mat;
                found = true;
            }
        }
        else if (sp + 2 <= RT_STACK)
        {
            stack[sp++] = nd.left;
            stack[sp++] = nd.right;
        }
    }
    return found;
}

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------

// mode 0 path traces. modes 1 and 2 are deterministic first-hit visualizations
// (normal, depth) used to diff against the CPU tracer, since a path trace can
// only ever be compared statistically.
#define RT_MODE_PATH 0
#define RT_MODE_NORMAL 1
#define RT_MODE_DEPTH 2

__device__ __forceinline__ float3 skyColor(const RTSky &sky, float3 d)
{
    float3 u = norm3(d);
    float a = 0.5f * (u.y + 1.0f);
    return mk(sky.lo[0], sky.lo[1], sky.lo[2]) * (1.0f - a) +
           mk(sky.hi[0], sky.hi[1], sky.hi[2]) * a;
}

__global__ void rt_kernel(const RTTri *tris, const RTNode *nodes, int nnodes,
                          const RTMat *mats, RTCam cam, RTSky sky,
                          int width, int height, int spp, int max_depth,
                          int mode, unsigned int seed, float *accum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int pix = y * width + x;
    float3 origin = ld3(cam.origin);
    float3 p00 = ld3(cam.pixel00);
    float3 du = ld3(cam.du);
    float3 dv = ld3(cam.dv);

    float3 total = mk(0.f, 0.f, 0.f);

    for (int s = 0; s < spp; s++)
    {
        unsigned int rng = seed ^ (pix * 9781u + s * 6271u);
        // seed already varies per accumulation pass, so successive passes
        // explore different paths rather than repeating the first one
        pcg(rng);

        // first sample is unjittered, matching ray_trace_tile_enhanced
        float ox = (s > 0) ? randf(rng) - 0.5f : 0.f;
        float oy = (s > 0) ? randf(rng) - 0.5f : 0.f;

        float3 target = p00 + du * ((float)x + ox) + dv * ((float)y + oy);
        float3 o = origin;
        float3 d = target - origin;

        if (mode != RT_MODE_PATH)
        {
            Hit h;
            if (traceRay(tris, nodes, nnodes, o, d, RT_INF, h))
            {
                float3 nrm = norm3(h.n);
                if (dot3(d, nrm) > 0.f)
                    nrm = -nrm;
                total = (mode == RT_MODE_NORMAL)
                            ? total + mk(nrm.x * .5f + .5f, nrm.y * .5f + .5f, nrm.z * .5f + .5f)
                            : total + mk(h.t, h.t, h.t);
            }
            continue;
        }

        // Iterative replacement for the recursive ray_color(): throughput is
        // carried forward instead of unwinding a call stack.
        float3 tp = mk(1.f, 1.f, 1.f);
        for (int depth = 0; depth < max_depth; depth++)
        {
            Hit h;
            if (!traceRay(tris, nodes, nnodes, o, d, RT_INF, h))
            {
                tp = tp * skyColor(sky, d);
                break;
            }

            float3 nrm = norm3(h.n);
            bool front_face = dot3(d, nrm) < 0.f;
            if (!front_face)
                nrm = -nrm;

            const RTMat &m = mats[h.mat];
            o = o + d * h.t;

            if (m.type == RT_MAT_METAL)
            {
                float3 refl = norm3(reflect3(norm3(d), nrm));
                refl = refl + randUnitVec(rng) * m.fuzz;
                // a fuzzed ray can end up below the surface; absorb it
                if (dot3(refl, nrm) <= 0.f)
                {
                    tp = mk(0.f, 0.f, 0.f);
                    break;
                }
                tp = tp * mk(m.albedo[0], m.albedo[1], m.albedo[2]);
                d = refl;
            }
            else if (m.type == RT_MAT_DIELECTRIC)
            {
                // glass does not tint what passes through it
                float ri = front_face ? (1.f / m.ior) : m.ior;
                float3 ud = norm3(d);
                float cos_theta = fminf(dot3(-ud, nrm), 1.f);
                float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));

                if (ri * sin_theta > 1.f || schlick(cos_theta, ri) > randf(rng))
                    d = reflect3(ud, nrm);
                else
                    d = refract3(ud, nrm, ri);
            }
            else
            {
                tp = tp * mk(m.albedo[0], m.albedo[1], m.albedo[2]);
                float3 sd = nrm + randUnitVec(rng);
                if (fabsf(sd.x) < 1e-8f && fabsf(sd.y) < 1e-8f && fabsf(sd.z) < 1e-8f)
                    sd = nrm;
                d = sd;
            }

            // depth exhausted without reaching the sky: the CPU returns black
            if (depth == max_depth - 1)
                tp = mk(0.f, 0.f, 0.f);
        }
        total = total + tp;
    }

    // Accumulate rather than overwrite. Every call adds its samples to what
    // is already there, so a still camera keeps refining the same image
    // instead of throwing it away and re-tracing at one sample per pixel.
    // The caller divides by the running sample count when resolving.
    accum[pix * 3 + 0] += total.x;
    accum[pix * 3 + 1] += total.y;
    accum[pix * 3 + 2] += total.z;
}

// Composites the traced frame straight into the rasterizer's device
// framebuffer, so neither image has to come down to the host. Both buffers are
// top-down RGB, so unlike the CPU blend there is no channel swap here. The RT
// buffer is usually half resolution, hence the nearest-neighbour lookup.
__global__ void rt_blend_kernel(const unsigned char *rt, int rw, int rh,
                                unsigned char *dst, int dw, int dh, float blend)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh)
        return;

    int sx = (x * rw) / dw;
    int sy = (y * rh) / dh;
    sx = min(rw - 1, max(0, sx));
    sy = min(rh - 1, max(0, sy));

    const unsigned char *s = rt + ((size_t)sy * rw + sx) * 3;
    // same rule as the CPU blend: a pixel the tracer never wrote stays raster
    if (s[0] == 0 && s[1] == 0 && s[2] == 0)
        return;

    unsigned char *d = dst + ((size_t)y * dw + x) * 3;
    for (int c = 0; c < 3; c++)
        d[c] = (unsigned char)(s[c] * blend + d[c] * (1.0f - blend));
}

// gamma + clamp, matching ColorConversion::rt_color_to_tga
__global__ void rt_resolve_kernel(const float *accum, unsigned char *rgb, int n,
                                  float inv_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    for (int c = 0; c < 3; c++)
    {
        float v = accum[i * 3 + c] * inv_samples;
        v = (v > 0.f) ? sqrtf(v) : 0.f;
        v = fminf(0.999f, fmaxf(0.f, v));
        rgb[i * 3 + c] = (unsigned char)(v * 255.f);
    }
}

// ---------------------------------------------------------------------------
// host: flat BVH build
// ---------------------------------------------------------------------------

namespace
{

struct BuildTri
{
    float c[3];
    float bmin[3];
    float bmax[3];
};

// Median split on the widest axis of the centroid bounds. Same shape of tree
// the CPU builder produces; the flat arrays are filled directly.
class BVHBuilder
{
public:
    std::vector<RTNode> nodes;
    std::vector<int> order;

    void build(const std::vector<BuildTri> &tri, int leaf_size)
    {
        int n = (int)tri.size();
        nodes.clear();
        order.resize(n);
        for (int i = 0; i < n; i++)
            order[i] = i;
        if (n == 0)
            return;
        nodes.reserve(2 * n);
        src = &tri;
        maxLeaf = leaf_size;
        recurse(0, n);
    }

private:
    const std::vector<BuildTri> *src;
    int maxLeaf;

    int recurse(int begin, int end)
    {
        int self = (int)nodes.size();
        nodes.push_back(RTNode());

        float bmin[3] = {RT_INF, RT_INF, RT_INF};
        float bmax[3] = {-RT_INF, -RT_INF, -RT_INF};
        float cmin[3] = {RT_INF, RT_INF, RT_INF};
        float cmax[3] = {-RT_INF, -RT_INF, -RT_INF};
        for (int i = begin; i < end; i++)
        {
            const BuildTri &t = (*src)[order[i]];
            for (int a = 0; a < 3; a++)
            {
                bmin[a] = std::min(bmin[a], t.bmin[a]);
                bmax[a] = std::max(bmax[a], t.bmax[a]);
                cmin[a] = std::min(cmin[a], t.c[a]);
                cmax[a] = std::max(cmax[a], t.c[a]);
            }
        }

        int count = end - begin;
        bool leaf = count <= maxLeaf;

        int axis = 0;
        float ext = cmax[0] - cmin[0];
        for (int a = 1; a < 3; a++)
            if (cmax[a] - cmin[a] > ext) { ext = cmax[a] - cmin[a]; axis = a; }

        // every centroid coincident: splitting cannot separate them
        if (!leaf && ext <= 0.f)
            leaf = true;

        int mid = (begin + end) / 2;
        if (!leaf)
        {
            const std::vector<BuildTri> &s = *src;
            std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                             [&s, axis](int a, int b) { return s[a].c[axis] < s[b].c[axis]; });
        }

        RTNode &nd = nodes[self];
        for (int a = 0; a < 3; a++) { nd.bmin[a] = bmin[a]; nd.bmax[a] = bmax[a]; }

        if (leaf)
        {
            nd.left = nd.right = -1;
            nd.first = begin;
            nd.count = count;
            return self;
        }

        nd.first = 0;
        nd.count = 0;
        int l = recurse(begin, mid);
        int r = recurse(mid, end);
        // the vector may have reallocated during recursion
        nodes[self].left = l;
        nodes[self].right = r;
        return self;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// host state
// ---------------------------------------------------------------------------

struct CudaRayTracer
{
    int width = 0, height = 0;
    RTTri *d_tris = NULL;
    RTNode *d_nodes = NULL;
    RTMat *d_mats = NULL;
    float *d_accum = NULL;
    unsigned char *d_rgb = NULL;
    int ntris = 0, nnodes = 0, nmats = 0;
    RTCam cam;
    RTSky sky;
    // total samples per pixel accumulated since the last reset
    int accum_samples = 0;
    bool ok = false;

    // reported so callers can see the build actually produced a tree
    int build_nodes = 0, build_leaves = 0, build_maxdepth = 0;

    CudaRayTracer(int w, int h) : width(w), height(h)
    {
        memset(&cam, 0, sizeof(cam));
        sky.lo[0] = sky.lo[1] = sky.lo[2] = 0.0f;
        sky.hi[0] = sky.hi[1] = sky.hi[2] = 0.8f;
        size_t px = (size_t)w * h;
        if (cudaMalloc(&d_accum, px * 3 * sizeof(float)) != cudaSuccess) return;
        if (cudaMalloc(&d_rgb, px * 3) != cudaSuccess) return;
        cudaMemset(d_accum, 0, px * 3 * sizeof(float));
        cudaMemset(d_rgb, 0, px * 3);
        ok = true;
        printf("CUDA ray tracer initialized: %dx%d\n", w, h);
    }

    ~CudaRayTracer()
    {
        freeScene();
        if (d_accum) cudaFree(d_accum);
        if (d_rgb) cudaFree(d_rgb);
    }

    void freeScene()
    {
        if (d_tris) { cudaFree(d_tris); d_tris = NULL; }
        if (d_nodes) { cudaFree(d_nodes); d_nodes = NULL; }
        if (d_mats) { cudaFree(d_mats); d_mats = NULL; }
        ntris = nnodes = nmats = 0;
    }

    // verts is ntri * 9 floats (v0, v1, v2), albedos is nmat * 3.
    // types/fuzz/iors may be NULL, in which case every material is lambertian.
    bool setScene(const float *verts, const int *tri_mat, int ntri,
                  const float *albedos, int nmat,
                  const int *types, const float *fuzz, const float *iors)
    {
        freeScene();
        if (ntri <= 0 || nmat <= 0)
            return false;

        std::vector<BuildTri> bt((size_t)ntri);
        for (int i = 0; i < ntri; i++)
        {
            const float *p = verts + i * 9;
            BuildTri &t = bt[i];
            for (int a = 0; a < 3; a++)
            {
                float lo = std::min(p[a], std::min(p[3 + a], p[6 + a]));
                float hi = std::max(p[a], std::max(p[3 + a], p[6 + a]));
                t.bmin[a] = lo;
                t.bmax[a] = hi;
                t.c[a] = (p[a] + p[3 + a] + p[6 + a]) / 3.f;
            }
        }

        BVHBuilder b;
        b.build(bt, 4);

        // triangles are written in traversal order so a leaf's range is
        // contiguous and the kernel reads them with one stride
        std::vector<RTTri> tris((size_t)ntri);
        for (int i = 0; i < ntri; i++)
        {
            const float *p = verts + b.order[i] * 9;
            RTTri &t = tris[i];
            for (int a = 0; a < 3; a++)
            {
                t.v0[a] = p[a];
                t.e1[a] = p[3 + a] - p[a];
                t.e2[a] = p[6 + a] - p[a];
            }
            float nx = t.e1[1] * t.e2[2] - t.e1[2] * t.e2[1];
            float ny = t.e1[2] * t.e2[0] - t.e1[0] * t.e2[2];
            float nz = t.e1[0] * t.e2[1] - t.e1[1] * t.e2[0];
            float l = sqrtf(nx * nx + ny * ny + nz * nz);
            if (l > 0.f) { nx /= l; ny /= l; nz /= l; }
            t.n[0] = nx; t.n[1] = ny; t.n[2] = nz;
            int m = tri_mat ? tri_mat[b.order[i]] : 0;
            t.mat = (m >= 0 && m < nmat) ? m : 0;
        }

        std::vector<RTMat> mats((size_t)nmat);
        for (int i = 0; i < nmat; i++)
        {
            for (int a = 0; a < 3; a++)
                mats[i].albedo[a] = albedos[i * 3 + a];
            mats[i].type = types ? types[i] : RT_MAT_LAMBERTIAN;
            mats[i].fuzz = fuzz ? fuzz[i] : 0.f;
            mats[i].ior = iors ? iors[i] : 1.5f;
        }

        build_nodes = (int)b.nodes.size();
        build_leaves = 0;
        for (size_t i = 0; i < b.nodes.size(); i++)
            if (b.nodes[i].count > 0) build_leaves++;
        build_maxdepth = depthOf(b.nodes, 0);

        if (cudaMalloc(&d_tris, tris.size() * sizeof(RTTri)) != cudaSuccess) return false;
        if (cudaMalloc(&d_nodes, b.nodes.size() * sizeof(RTNode)) != cudaSuccess) return false;
        if (cudaMalloc(&d_mats, mats.size() * sizeof(RTMat)) != cudaSuccess) return false;
        cudaMemcpy(d_tris, tris.data(), tris.size() * sizeof(RTTri), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nodes, b.nodes.data(), b.nodes.size() * sizeof(RTNode), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mats, mats.data(), mats.size() * sizeof(RTMat), cudaMemcpyHostToDevice);

        ntris = ntri;
        nnodes = (int)b.nodes.size();
        nmats = nmat;
        return true;
    }

    static int depthOf(const std::vector<RTNode> &nd, int i)
    {
        if (i < 0 || i >= (int)nd.size()) return 0;
        if (nd[i].count > 0) return 1;
        return 1 + std::max(depthOf(nd, nd[i].left), depthOf(nd, nd[i].right));
    }

    void resetAccum()
    {
        if (!ok) return;
        cudaMemset(d_accum, 0, (size_t)width * height * 3 * sizeof(float));
        accum_samples = 0;
    }

    void render(int spp, int max_depth, int mode, unsigned int seed)
    {
        if (!ok || nnodes == 0)
            return;
        dim3 block(8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        rt_kernel<<<grid, block>>>(d_tris, d_nodes, nnodes, d_mats, cam, sky,
                                   width, height, spp, max_depth, mode, seed, d_accum);
        accum_samples += spp;
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            printf("CUDA ray tracer kernel error: %s\n", cudaGetErrorString(e));
    }

    float invSamples() const
    {
        return (accum_samples > 0) ? 1.0f / (float)accum_samples : 0.0f;
    }

    void resolve(unsigned char *host_rgb)
    {
        if (!ok) return;
        int n = width * height;
        int blk = 256;
        rt_resolve_kernel<<<(n + blk - 1) / blk, blk>>>(d_accum, d_rgb, n, invSamples());
        cudaMemcpy(host_rgb, d_rgb, (size_t)n * 3, cudaMemcpyDeviceToHost);
    }

    // resolves into d_rgb, then blends that into dst without a readback
    void blendInto(unsigned char *dst, int dw, int dh, float blend)
    {
        if (!ok || !dst || dw <= 0 || dh <= 0)
            return;
        int n = width * height;
        int blk = 256;
        rt_resolve_kernel<<<(n + blk - 1) / blk, blk>>>(d_accum, d_rgb, n, invSamples());

        dim3 block(16, 16);
        dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
        rt_blend_kernel<<<grid, block>>>(d_rgb, width, height, dst, dw, dh, blend);

        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            printf("CUDA ray tracer blend error: %s\n", cudaGetErrorString(e));
    }

    // averaged, not the raw running sum, so callers see a radiance value
    // regardless of how many passes have accumulated
    void readAccum(float *host_accum)
    {
        if (!ok) return;
        size_t n = (size_t)width * height * 3;
        cudaMemcpy(host_accum, d_accum, n * sizeof(float), cudaMemcpyDeviceToHost);
        float inv = invSamples();
        for (size_t i = 0; i < n; i++)
            host_accum[i] *= inv;
    }
};

static CudaRayTracer *g_rt = NULL;

extern "C" {

bool initCudaRayTracer(int width, int height)
{
    if (g_rt) delete g_rt;
    g_rt = new CudaRayTracer(width, height);
    if (!g_rt->ok) { delete g_rt; g_rt = NULL; return false; }
    return true;
}

void cleanupCudaRayTracer()
{
    if (g_rt) { delete g_rt; g_rt = NULL; }
}

bool cudaRTAvailable() { return g_rt != NULL; }

// verts: ntri * 9 floats. tri_mat may be NULL, in which case every triangle
// uses material 0. albedos: nmat * 3 floats.
bool cudaRTSetScene(const float *verts, const int *tri_mat, int ntri,
                    const float *albedos, int nmat)
{
    return g_rt ? g_rt->setScene(verts, tri_mat, ntri, albedos, nmat,
                                 NULL, NULL, NULL) : false;
}

// types: 0 lambertian, 1 metal, 2 dielectric. fuzz applies to metal, ior to
// dielectric; both may be NULL for an all-lambertian scene.
bool cudaRTSetSceneWithMaterials(const float *verts, const int *tri_mat, int ntri,
                                 const float *albedos, int nmat,
                                 const int *types, const float *fuzz,
                                 const float *iors)
{
    return g_rt ? g_rt->setScene(verts, tri_mat, ntri, albedos, nmat,
                                 types, fuzz, iors) : false;
}

void cudaRTSetCamera(const float *origin3, const float *pixel00_3,
                     const float *du3, const float *dv3)
{
    if (!g_rt) return;
    for (int i = 0; i < 3; i++)
    {
        g_rt->cam.origin[i] = origin3[i];
        g_rt->cam.pixel00[i] = pixel00_3[i];
        g_rt->cam.du[i] = du3[i];
        g_rt->cam.dv[i] = dv3[i];
    }
}

// lo is the colour straight down, hi straight up. pass both black to
// reproduce the realtime CPU tracer, which renders an entirely black image.
void cudaRTSetSky(const float *lo3, const float *hi3)
{
    if (!g_rt) return;
    for (int i = 0; i < 3; i++)
    {
        g_rt->sky.lo[i] = lo3[i];
        g_rt->sky.hi[i] = hi3[i];
    }
}

// Every call ADDS its samples to whatever has accumulated since the last
// reset. Call cudaRTResetAccumulation() whenever the image is invalidated:
// the camera moved, the scene changed, or a deterministic visualization mode
// is about to be used.
void cudaRTRender(int spp, int max_depth, int mode, unsigned int seed)
{
    if (g_rt) g_rt->render(spp, max_depth, mode, seed);
}

void cudaRTResetAccumulation()
{
    if (g_rt) g_rt->resetAccum();
}

// samples per pixel accumulated so far, for progress reporting
int cudaRTGetSampleCount()
{
    return g_rt ? g_rt->accum_samples : 0;
}

// host_rgb is width * height * 3, top-down R,G,B
void cudaRTGetResults(unsigned char *host_rgb)
{
    if (g_rt) g_rt->resolve(host_rgb);
}

// raw linear values, before gamma. used by the differential tests.
void cudaRTGetAccum(float *host_accum)
{
    if (g_rt) g_rt->readAccum(host_accum);
}

// dst is a device pointer to a top-down RGB image, normally the rasterizer's
// own framebuffer via cudaGetDeviceFramebuffer.
void cudaRTBlendToFramebuffer(unsigned char *dst, int dst_w, int dst_h, float blend)
{
    if (g_rt) g_rt->blendInto(dst, dst_w, dst_h, blend);
}

void cudaRTGetBVHStats(int *nodes, int *leaves, int *max_depth, int *tris)
{
    if (nodes) *nodes = g_rt ? g_rt->build_nodes : 0;
    if (leaves) *leaves = g_rt ? g_rt->build_leaves : 0;
    if (max_depth) *max_depth = g_rt ? g_rt->build_maxdepth : 0;
    if (tris) *tris = g_rt ? g_rt->ntris : 0;
}

} // extern "C"
