#pragma once
// =====================================================================================
// Geometry of one LOR for Joseph's method
// =====================================================================================
//
// Shared by all Joseph projectors (non-TOF, TOF sinogram and TOF listmode; forward and back;
// CPU and CUDA), so that they use exactly the same planes, interpolation weights and TOF
// positions (adjointness).
//
// Joseph's method in short
// ------------------------
// The LOR from p0 (lor_start) to p1 (lor_end) is sampled once in every image plane
// perpendicular to its *principal axis* (the coordinate axis along which the LOR changes
// the most, `direction`). Example: a LOR that is closer to the x-axis than to y and z is
// sampled in the planes x = const (index ip of axis 0), one sample per plane:
//
//      plane ip:   |       |       |       |       |
//                  |   *   |       |       |       |      * = intersection of the LOR
//      LOR:  ------+-------*-------+-------+-------+--    with plane ip
//                  |       |      *|       |       |
//                  ip=3    ip=4    ip=5    ip=6
//
// In each plane, the image is interpolated bilinearly at the intersection point (the two
// other axes are called "a" and "b" below). Each sample stands for the piece of the LOR
// between two planes, whose length is cf = voxel_size[direction] / cos(angle between the
// LOR and the principal axis). The (non-TOF) line integral is
//
//      sum over planes ip of  cf * bilinear_interpolation(image, plane ip, fa(ip), fb(ip))
//
// and the back projection adds cf * value, distributed with the same 4 bilinear weights,
// to the 4 voxels around each intersection point (the adjoint).
//
// Image layout: C order, voxel (i0, i1, i2) is image[i0 * n1 * n2 + i1 * n2 + i2]; voxel
// centers are at image_origin + (i0, i1, i2) * voxel_size.
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"

struct JosephLOR
{
  int direction; ///< principal axis (0, 1 or 2)
  float cf;      ///< length of the LOR between two planes: voxel_size[direction] / cos(angle to the principal axis)
  int istart;    ///< first image plane intersected by the LOR (-1: the LOR misses the image)
  int iend;      ///< last intersected image plane
  // continuous voxel coordinates of the intersection point in plane ip along the two other
  // axes ("a" and "b"): fa = ip * aa + ba, fb = ip * ab + bb (linear in ip, see init)
  float aa, ba, ab, bb;
  // memory strides (in voxels) of the plane axis and of the two other axes: voxel
  // (ip, ja, jb) is image[ip * s_dir + ja * s_a + jb * s_b]
  std::size_t s_dir, s_a, s_b;
  int n_a, n_b; ///< image dimensions along the two other axes

  /// Set up the geometry of the LOR from p0 to p1; returns false if the LOR misses the image.
  WORKER_QUALIFIER bool init(const float *p0, const float *p1, const float *image_origin, const float *voxel_size,
                             const int *image_dim)
  {
    // principal axis, cf and the first / last intersected plane (utils.h)
    istart = -1;
    iend = -1;
    ray_cube_intersection_joseph(p0, p1, image_origin, voxel_size, image_dim, direction, cf, istart, iend);
    if (istart == -1)
      return false;

    const int n0 = image_dim[0];
    const int n1 = image_dim[1];
    const int n2 = image_dim[2];
    // the two other axes: (a, b) = (1, 2) for direction 0, (0, 2) for 1, (0, 1) for 2
    const int ax = (direction == 0) ? 1 : 0;
    const int bx = (direction == 2) ? 1 : 2;

    // Derivation of fa(ip) (fb is the same with b instead of a):
    //  - plane ip lies at x_dir = or_dir + ip * vs_dir (position of the voxel centers),
    //  - the LOR reaches it at the parameter s = (x_dir - p0_dir) / dr  (p = p0 + s (p1 - p0)),
    //  - where its a-coordinate is x_a = p0_a + s * da,
    //  - in voxel units: fa = (x_a - or_a) / vs_a.
    // Inserting gives fa = ip * aa + ba with the aa and ba below.
    const float p0_dir = p0[direction], p0_a = p0[ax], p0_b = p0[bx];
    const float dr = p1[direction] - p0_dir;
    const float da = p1[ax] - p0_a;
    const float db = p1[bx] - p0_b;
    const float vs_dir = voxel_size[direction], vs_a = voxel_size[ax], vs_b = voxel_size[bx];
    const float or_dir = image_origin[direction], or_a = image_origin[ax], or_b = image_origin[bx];
    aa = (da * vs_dir) / (vs_a * dr);
    ba = (p0_a - or_a + da * (or_dir - p0_dir) / dr) / vs_a;
    ab = (db * vs_dir) / (vs_b * dr);
    bb = (p0_b - or_b + db * (or_dir - p0_dir) / dr) / vs_b;

    // Memory strides of the three axes in the C-ordered image (stride of axis 0: n1 * n2,
    // axis 1: n2, axis 2: 1), assigned to "plane axis", "a" and "b". Written with
    // conditional expressions instead of a small array indexed with `direction`: on the
    // GPU, such an array would be placed in (slow) local memory instead of registers.
    s_dir = (direction == 0) ? std::size_t(n1) * n2 : ((direction == 1) ? std::size_t(n2) : 1);
    s_a = (ax == 0) ? std::size_t(n1) * n2 : std::size_t(n2); // ax is 0 or 1
    s_b = (bx == 1) ? std::size_t(n2) : 1;                    // bx is 1 or 2
    n_a = (ax == 0) ? n0 : n1;
    n_b = (bx == 1) ? n1 : n2;
    return true;
  }

  /// Bilinear interpolation of the image in plane ip at (fa, fb). Voxels outside the image
  /// count as zero.
  WORKER_QUALIFIER float bilinear(const float *image, int ip) const
  {
    const float fa = ip * aa + ba;
    const float fb = ip * ab + bb;
    // the 4 surrounding voxels are (ia or ia + 1, ib or ib + 1); wa, wb in [0, 1) are the
    // fractional parts (weight of the "+1" neighbor along a and b)
    const int ia = static_cast<int>(floorf(fa));
    const int ib = static_cast<int>(floorf(fb));
    const float wa = fa - ia;
    const float wb = fb - ib;
    float val = 0.0f;
    // q = 0: (ia, ib)      weight (1 - wa) (1 - wb)
    // q = 1: (ia + 1, ib)  weight wa (1 - wb)
    // q = 2: (ia, ib + 1)  weight (1 - wa) wb
    // q = 3: (ia + 1, ib + 1) weight wa wb
    // (q & 1 selects the a-neighbor, q >> 1 the b-neighbor; the loop is unrolled)
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
      const int ja = ia + (q & 1);
      const int jb = ib + (q >> 1);
      if ((ja >= 0) && (ja < n_a) && (jb >= 0) && (jb < n_b))
        val += ((q & 1) ? wa : 1.0f - wa) * ((q >> 1) ? wb : 1.0f - wb) * image[ip * s_dir + ja * s_a + jb * s_b];
    }
    return val;
  }

  /// Adjoint of bilinear: add val, distributed with the same 4 bilinear weights, to the
  /// voxels of plane ip. Atomic additions, because other threads (LORs) can add to the
  /// same voxels at the same time.
  WORKER_QUALIFIER void bilinear_adjoint(float *image, int ip, float val) const
  {
    const float fa = ip * aa + ba;
    const float fb = ip * ab + bb;
    const int ia = static_cast<int>(floorf(fa));
    const int ib = static_cast<int>(floorf(fb));
    const float wa = fa - ia;
    const float wb = fb - ib;
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
      const int ja = ia + (q & 1);
      const int jb = ib + (q >> 1);
      if ((ja >= 0) && (ja < n_a) && (jb >= 0) && (jb < n_b))
        atomic_sum(image + ip * s_dir + ja * s_a + jb * s_b, ((q & 1) ? wa : 1.0f - wa) * ((q >> 1) ? wb : 1.0f - wb) * val);
    }
  }

  /// cf times the bilinear interpolation in plane ip: the contribution of plane ip to the
  /// (non-TOF) line integral (used per plane by the TOF projectors; the non-TOF projectors
  /// sum bilinear() over all planes and multiply by cf once).
  WORKER_QUALIFIER float interp(const float *image, int ip) const { return cf * bilinear(image, ip); }

  /// Adjoint of interp: add cf * val, distributed with the bilinear weights, to plane ip.
  WORKER_QUALIFIER void interp_adjoint(float *image, int ip, float val) const { bilinear_adjoint(image, ip, cf * val); }

  /// TOF position of the planes: the continuous TOF bin coordinate (see tof_kernel.h) of
  /// the intersection point with plane ip is it_f = ip * at + bt
  /// (computed with tof_plane_position(ip, at, bt)).
  ///
  /// Derivation:
  ///  - The TOF bin centers lie on the LOR with spacing tof_bin_width, symmetric around
  ///    the LOR center (the center of bin (num_tof_bins - 1) / 2 is the LOR center for
  ///    zero offset), shifted by tof_center_offset; a positive offset shifts them towards
  ///    p1 (lor_end). Bin numbers increase from p0 towards p1.
  ///  - Projected onto the principal axis, distances along the LOR shrink by
  ///    costheta = cos(angle to the principal axis), so the bin centers are at
  ///    x_dir = tof_origin + t * tof_slope (t = bin number) with
  ///    tof_slope = sign * tof_bin_width * costheta, where sign = +1 if the LOR runs in the
  ///    positive direction of the principal axis (p1 > p0 along it), -1 otherwise.
  ///  - Plane ip is at x_dir = or_dir + ip * vs_dir, so its TOF coordinate is
  ///    it_f = (or_dir + ip * vs_dir - tof_origin) / tof_slope = ip * at + bt with
  ///    at = vs_dir / tof_slope = sign * cf / tof_bin_width (since costheta = vs_dir / cf)
  ///    and bt = (or_dir - tof_origin) / tof_slope.
  /// |at| = cf / tof_bin_width is the number of TOF bins the TOF position moves per plane.
  WORKER_QUALIFIER void tof_line(const float *p0, const float *p1, float tof_bin_width, short num_tof_bins,
                                 float tof_center_offset, const float *image_origin, const float *voxel_size,
                                 float &at, float &bt) const
  {
    const float costheta = voxel_size[direction] / cf;
    // sign: whether the TOF bin numbers increase (+1) or decrease (-1) along the principal axis
    const float sign = (p1[direction] >= p0[direction]) ? 1.0f : -1.0f;
    // center of TOF bin 0 projected onto the principal axis: LOR center, minus half the
    // TOF range towards p0, plus the center offset towards p1
    const float tof_origin = 0.5f * (p0[direction] + p1[direction]) - sign * (0.5f * num_tof_bins - 0.5f) * (tof_bin_width * costheta) + sign * tof_center_offset * costheta;
    const float tof_slope = sign * tof_bin_width * costheta;
    at = sign * cf / tof_bin_width;
    bt = (image_origin[direction] - tof_origin) / tof_slope;
  }
};
