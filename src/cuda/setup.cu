// First stage of the pipeline: mesh geometry and matrices in, screen-space
// triangles out. What it appends to is the buffer binning.cu then bins.
#include <cmath>
#include <cfloat>

#include "common.cuh"
#include "stages.cuh"

// One triangle corner in the form clipping needs: homogeneous position plus
// the attributes that travel with it. Positions stay pre-divide, which is what
// makes a corner introduced by clipping correct -- interpolating linearly in
// clip space is exactly right, because the rasterizer rebuilds
// perspective-correct weights from w further down the pipeline.
struct SetupVert {
    CudaVec4 v;
    float nx, ny, nz;
    float u, vt;
};

__device__ inline SetupVert lerp_vert(const SetupVert& a, const SetupVert& b,
                                      float t)
{
    SetupVert r;
    r.v.x = a.v.x + (b.v.x - a.v.x) * t;
    r.v.y = a.v.y + (b.v.y - a.v.y) * t;
    r.v.z = a.v.z + (b.v.z - a.v.z) * t;
    r.v.w = a.v.w + (b.v.w - a.v.w) * t;
    r.nx  = a.nx  + (b.nx  - a.nx)  * t;
    r.ny  = a.ny  + (b.ny  - a.ny)  * t;
    r.nz  = a.nz  + (b.nz  - a.nz)  * t;
    r.u   = a.u   + (b.u   - a.u)   * t;
    r.vt  = a.vt  + (b.vt  - a.vt)  * t;
    return r;
}

// Backface cull, bbox and append. Runs per OUTPUT triangle rather than per
// input face, since clipping can turn one face into two and each half needs
// its own bounding box.
__device__ inline void emit_tri(const SetupVert& a, const SetupVert& b,
                                const SetupVert& c, CudaColor color, int mat,
                                CudaTriangle* out, int* out_count, int max_out,
                                int width, int height, int* stats)
{
    float sx0 = a.v.x / a.v.w, sy0 = a.v.y / a.v.w;
    float sx1 = b.v.x / b.v.w, sy1 = b.v.y / b.v.w;
    float sx2 = c.v.x / c.v.w, sy2 = c.v.y / c.v.w;

    // Backface cull. The models wind front faces counter-clockwise in screen
    // space, so a front face has a POSITIVE edge cross product and a negative
    // one faces away. Culling is only ever an optimization: for a closed mesh
    // the image has to be identical with it on and off, which is the check
    // that settles the sign (it was inverted here, and from the front that
    // leaves the inside of the far side of the mesh, which keeps a plausible
    // silhouette while lighting it by normals that point away).
    float facing = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0);
    if (facing <= 0.0f) { atomicAdd(&stats[0], 1); return; }

    int bx0 = max(0,          (int)floorf(fminf(fminf(sx0, sx1), sx2)));
    int bx1 = min(width  - 1, (int)ceilf (fmaxf(fmaxf(sx0, sx1), sx2)));
    int by0 = max(0,          (int)floorf(fminf(fminf(sy0, sy1), sy2)));
    int by1 = min(height - 1, (int)ceilf (fmaxf(fmaxf(sy0, sy1), sy2)));
    if (bx0 > bx1 || by0 > by1) { atomicAdd(&stats[1], 1); return; }

    int slot = atomicAdd(out_count, 1);
    if (slot >= max_out) { atomicAdd(&stats[2], 1); return; }

    CudaTriangle& t = out[slot];
    t.v[0] = a.v;  t.v[1] = b.v;  t.v[2] = c.v;
    t.nx[0] = a.nx; t.ny[0] = a.ny; t.nz[0] = a.nz; t.u[0] = a.u; t.vt[0] = a.vt;
    t.nx[1] = b.nx; t.ny[1] = b.ny; t.nz[1] = b.nz; t.u[1] = b.u; t.vt[1] = b.vt;
    t.nx[2] = c.nx; t.ny[2] = c.ny; t.nz[2] = c.nz; t.u[2] = c.u; t.vt[2] = c.vt;
    t.color = color;
    t.mat = mat;
    t.bbox_min_x = bx0; t.bbox_max_x = bx1;
    t.bbox_min_y = by0; t.bbox_max_y = by1;
}

// Transforms every queued mesh and does setup (clip + cull + bbox), appending
// survivors straight into the shared triangle buffer. One launch covers the
// whole scene: a launch costs ~11us here, which a 40-mesh scene would
// otherwise pay 80 times a frame across the two passes.
__global__
void mesh_setup_kernel(const MeshDraw* draws, int ndraws, int total_faces,
                       CudaTriangle* out, int* out_count, int max_out,
                       int width, int height, int* stats)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_faces) return;

    // which mesh owns this face. face_begin is ascending, so this is an
    // upper bound search for the last draw starting at or before gid.
    int lo = 0, hi = ndraws - 1, d = 0;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (gid < draws[mid].face_begin) hi = mid - 1;
        else { d = mid; lo = mid + 1; }
    }

    const MeshDraw& dr = draws[d];
    const float* verts = dr.verts;
    const int* faces = dr.faces;
    const float* cnorms = dr.cnorms;
    const float* cuvs = dr.cuvs;
    const Mat4& mvp = dr.mvp;
    const Mat4& clip = dr.clip;
    const Mat4& mit = dr.mit;
    const int mat_id = dr.mat_id;
    const CudaColor color = dr.color;

    int f = gid - dr.face_begin;
    atomicAdd(&stats[3], 1);                       // submitted

    SetupVert in[3];
    CudaVec4 cv[3];
    float mx[3], my[3], mz[3];                     // model-space, for the normal
    for (int k = 0; k < 3; k++) {
        int vi = faces[f * 3 + k];
        float px = verts[vi * 3 + 0];
        float py = verts[vi * 3 + 1];
        float pz = verts[vi * 3 + 2];
        mx[k] = px; my[k] = py; mz[k] = pz;
        in[k].v.x = mvp.m[0]  * px + mvp.m[1]  * py + mvp.m[2]  * pz + mvp.m[3];
        in[k].v.y = mvp.m[4]  * px + mvp.m[5]  * py + mvp.m[6]  * pz + mvp.m[7];
        in[k].v.z = mvp.m[8]  * px + mvp.m[9]  * py + mvp.m[10] * pz + mvp.m[11];
        in[k].v.w = mvp.m[12] * px + mvp.m[13] * py + mvp.m[14] * pz + mvp.m[15];
        cv[k].x = clip.m[0]  * px + clip.m[1]  * py + clip.m[2]  * pz + clip.m[3];
        cv[k].y = clip.m[4]  * px + clip.m[5]  * py + clip.m[6]  * pz + clip.m[7];
        cv[k].z = clip.m[8]  * px + clip.m[9]  * py + clip.m[10] * pz + clip.m[11];
        cv[k].w = clip.m[12] * px + clip.m[13] * py + clip.m[14] * pz + clip.m[15];
    }

    // Corner attributes, computed before clipping so a corner introduced by it
    // can simply interpolate them. Normals go to eye space via the
    // inverse-transpose, using only the 3x3 block, so the 4th component is
    // implicitly 0.
    for (int k = 0; k < 3; k++) {
        float snx, sny, snz;
        if (cnorms) {
            snx = cnorms[(f * 3 + k) * 3 + 0];
            sny = cnorms[(f * 3 + k) * 3 + 1];
            snz = cnorms[(f * 3 + k) * 3 + 2];
        } else {
            // no vertex normals in the file: fall back to the face normal
            float e1x = mx[1] - mx[0], e1y = my[1] - my[0], e1z = mz[1] - mz[0];
            float e2x = mx[2] - mx[0], e2y = my[2] - my[0], e2z = mz[2] - mz[0];
            snx = e1y * e2z - e1z * e2y;
            sny = e1z * e2x - e1x * e2z;
            snz = e1x * e2y - e1y * e2x;
        }
        float ex = mit.m[0] * snx + mit.m[1] * sny + mit.m[2]  * snz;
        float ey = mit.m[4] * snx + mit.m[5] * sny + mit.m[6]  * snz;
        float ez = mit.m[8] * snx + mit.m[9] * sny + mit.m[10] * snz;
        float l = sqrtf(ex * ex + ey * ey + ez * ez);
        if (l > 1e-12f) { ex /= l; ey /= l; ez /= l; }
        in[k].nx = ex; in[k].ny = ey; in[k].nz = ez;
        in[k].u  = cuvs ? cuvs[(f * 3 + k) * 2 + 0] : 0.0f;
        in[k].vt = cuvs ? cuvs[(f * 3 + k) * 2 + 1] : 0.0f;
    }

    // The near plane, in w. lookat centres eye space on the TARGET rather than
    // the camera -- its translation is -R*center -- and projection(coeff) is
    // the identity with Projection[3][2] = coeff = -1/d. So w = 1 - z_eye/d,
    // which puts the target plane at w = 1 and the camera itself at w = 0.
    //
    // w > 0 is therefore the right plane, and the cull below always tested it.
    // What it could not do is act on a triangle that only partly fails: it
    // rejected one only when all three corners were behind the camera, so a
    // straddling triangle went through with a near-zero or negative w and
    // divided by it. That is the stretched wedge across a frame, and the
    // mirroring once w flips sign. From outside a model every corner has
    // w >= 0.3 and none of it shows; from inside a scene it is most of the
    // frame. Clipping is what the cull was missing, not a different plane.
    //
    // A small positive floor keeps the divide well away from the singularity,
    // and costs a near plane 0.001*d in front of the camera. The orthographic
    // shadow pass has coeff = 0, hence w identically 1, so it never clips.
    const float w_min = 1e-3f;

    int n_in = 0;
    for (int k = 0; k < 3; k++) if (in[k].v.w >= w_min) n_in++;
    if (n_in == 0) { atomicAdd(&stats[1], 1); return; }

    if (n_in == 3) {
        // Wholly in front. The x/y frustum tests are only meaningful once w is
        // known positive, which is why they sit on this side of the branch; a
        // straddling triangle reaches the same rejection through the clipped
        // halves and their bounding boxes instead.
        bool outside = true;
        for (int k = 0; k < 3; k++) if (cv[k].x >= -cv[k].w) outside = false;
        if (outside) { atomicAdd(&stats[1], 1); return; }
        outside = true;
        for (int k = 0; k < 3; k++) if (cv[k].x <= cv[k].w) outside = false;
        if (outside) { atomicAdd(&stats[1], 1); return; }
        outside = true;
        for (int k = 0; k < 3; k++) if (cv[k].y >= -cv[k].w) outside = false;
        if (outside) { atomicAdd(&stats[1], 1); return; }
        outside = true;
        for (int k = 0; k < 3; k++) if (cv[k].y <= cv[k].w) outside = false;
        if (outside) { atomicAdd(&stats[1], 1); return; }

        emit_tri(in[0], in[1], in[2], color, mat_id,
                 out, out_count, max_out, width, height, stats);
        return;
    }

    // Straddles the eye plane: Sutherland-Hodgman against the single plane
    // w >= w_min. One plane against a triangle yields a triangle or a quad and
    // never more, so four slots is the whole story.
    SetupVert poly[4];
    int n = 0;
    for (int k = 0; k < 3; k++) {
        const SetupVert& cur = in[k];
        const SetupVert& nxt = in[(k + 1) % 3];
        bool cur_in = cur.v.w >= w_min;
        bool nxt_in = nxt.v.w >= w_min;
        if (cur_in) poly[n++] = cur;
        if (cur_in != nxt_in) {
            float denom = nxt.v.w - cur.v.w;
            float t = (fabsf(denom) > 1e-20f) ? (w_min - cur.v.w) / denom : 0.0f;
            poly[n++] = lerp_vert(cur, nxt, fminf(1.0f, fmaxf(0.0f, t)));
        }
    }
    if (n < 3) { atomicAdd(&stats[1], 1); return; }

    emit_tri(poly[0], poly[1], poly[2], color, mat_id,
             out, out_count, max_out, width, height, stats);
    if (n == 4)
        emit_tri(poly[0], poly[2], poly[3], color, mat_id,
                 out, out_count, max_out, width, height, stats);
}

void cudaLaunchMeshSetup(const MeshDraw* draws, int ndraws, int total_faces,
                         CudaTriangle* out, int* out_count, int max_out,
                         int width, int height, int* stats)
{
    int block = 256;
    int grid = (total_faces + block - 1) / block;
    mesh_setup_kernel<<<grid, block>>>(draws, ndraws, total_faces,
                                       out, out_count, max_out,
                                       width, height, stats);
}
