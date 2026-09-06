// First stage of the pipeline: mesh geometry and matrices in, screen-space
// triangles out. What it appends to is the buffer binning.cu then bins.
#include <cmath>
#include <cfloat>

#include "common.cuh"
#include "stages.cuh"

// Transforms every queued mesh and does setup (cull + bbox), appending
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

    CudaVec4 v[3], cv[3];
    float sx[3], sy[3];
    float mx[3], my[3], mz[3];                     // model-space, for the normal
    for (int k = 0; k < 3; k++) {
        int vi = faces[f * 3 + k];
        float px = verts[vi * 3 + 0];
        float py = verts[vi * 3 + 1];
        float pz = verts[vi * 3 + 2];
        mx[k] = px; my[k] = py; mz[k] = pz;
        v[k].x = mvp.m[0]  * px + mvp.m[1]  * py + mvp.m[2]  * pz + mvp.m[3];
        v[k].y = mvp.m[4]  * px + mvp.m[5]  * py + mvp.m[6]  * pz + mvp.m[7];
        v[k].z = mvp.m[8]  * px + mvp.m[9]  * py + mvp.m[10] * pz + mvp.m[11];
        v[k].w = mvp.m[12] * px + mvp.m[13] * py + mvp.m[14] * pz + mvp.m[15];
        cv[k].x = clip.m[0]  * px + clip.m[1]  * py + clip.m[2]  * pz + clip.m[3];
        cv[k].y = clip.m[4]  * px + clip.m[5]  * py + clip.m[6]  * pz + clip.m[7];
        cv[k].z = clip.m[8]  * px + clip.m[9]  * py + clip.m[10] * pz + clip.m[11];
        cv[k].w = clip.m[12] * px + clip.m[13] * py + clip.m[14] * pz + clip.m[15];
        sx[k] = v[k].x / v[k].w;
        sy[k] = v[k].y / v[k].w;
    }

    bool outside = true;
    for (int k = 0; k < 3; k++) if (cv[k].w > 0.0f) outside = false;
    if (outside) { atomicAdd(&stats[1], 1); return; }
    outside = true;
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
    // Backface cull. The models wind front faces counter-clockwise in screen
    // space, so a front face has a POSITIVE edge cross product and a negative
    // one faces away. Culling is only ever an optimization: for a closed mesh
    // the image has to be identical with it on and off, which is the check
    // that settles the sign (it was inverted here, and from the front that
    // leaves the inside of the far side of the mesh, which keeps a plausible
    // silhouette while lighting it by normals that point away).
    float facing = (sx[1] - sx[0]) * (sy[2] - sy[0])
                 - (sy[1] - sy[0]) * (sx[2] - sx[0]);
    if (facing <= 0.0f) { atomicAdd(&stats[0], 1); return; }

    int bx0 = max(0,          (int)floorf(fminf(fminf(sx[0], sx[1]), sx[2])));
    int bx1 = min(width  - 1, (int)ceilf (fmaxf(fmaxf(sx[0], sx[1]), sx[2])));
    int by0 = max(0,          (int)floorf(fminf(fminf(sy[0], sy[1]), sy[2])));
    int by1 = min(height - 1, (int)ceilf (fmaxf(fmaxf(sy[0], sy[1]), sy[2])));
    if (bx0 > bx1 || by0 > by1) { atomicAdd(&stats[1], 1); return; }

    int slot = atomicAdd(out_count, 1);
    if (slot >= max_out) { atomicAdd(&stats[2], 1); return; }

    CudaTriangle& t = out[slot];
    t.v[0] = v[0]; t.v[1] = v[1]; t.v[2] = v[2];
    t.color = color;
    t.mat = mat_id;

    // corner attributes travel with the triangle, since shading is
    // per-fragment. normals go to eye space via the inverse-transpose, using
    // only the 3x3 block, so the 4th component is implicitly 0.
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
        t.nx[k] = ex; t.ny[k] = ey; t.nz[k] = ez;
        t.u[k]  = cuvs ? cuvs[(f * 3 + k) * 2 + 0] : 0.0f;
        t.vt[k] = cuvs ? cuvs[(f * 3 + k) * 2 + 1] : 0.0f;
    }
    t.bbox_min_x = bx0; t.bbox_max_x = bx1;
    t.bbox_min_y = by0; t.bbox_max_y = by1;
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
