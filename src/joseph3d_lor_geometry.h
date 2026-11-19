#pragma once
// Geometry of one LOR for Joseph's method, shared by the TOF sinogram forward and back
// projection (CPU and CUDA): principal axis, intersected image planes, bilinear
// interpolation in a plane (and its adjoint), and the TOF bin coordinate of a plane.
#include "cuda_compat.h"
#include "utils.h"
#include "tof_kernel.h"

struct JosephLOR
{
  int direction; ///< principal axis
  float cf;      ///< correction factor voxel_size[direction] / cos(angle to the principal axis)
  int istart;    ///< first intersected image plane (-1: no intersection)
  int iend;      ///< last intersected image plane
  // interpolation coordinates of the two other axes in plane ip: fa = ip * aa + ba, fb = ip * ab + bb
  float aa, ba, ab, bb;
  // linear voxel index strides of the plane axis and of the two interpolation axes
  std::size_t s_dir, s_a, s_b;
  int n_a, n_b; // image dimensions of the two interpolation axes

  /// Set up the geometry of the LOR from p0 to p1; returns false if the LOR misses the image.
  WORKER_QUALIFIER bool init(const float *p0, const float *p1, const float *image_origin, const float *voxel_size,
                             const int *image_dim)
  {
    istart = -1;
    iend = -1;
    ray_cube_intersection_joseph(p0, p1, image_origin, voxel_size, image_dim, direction, cf, istart, iend);
    if (istart == -1)
      return false;

    const int n0 = image_dim[0];
    const int n1 = image_dim[1];
    const int n2 = image_dim[2];
    const int ax = (direction == 0) ? 1 : 0;
    const int bx = (direction == 2) ? 1 : 2;
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

    // selects instead of small arrays indexed by direction (which would go to local memory)
    s_dir = (direction == 0) ? std::size_t(n1) * n2 : ((direction == 1) ? std::size_t(n2) : 1);
    s_a = (ax == 0) ? std::size_t(n1) * n2 : std::size_t(n2); // ax is 0 or 1
    s_b = (bx == 1) ? std::size_t(n2) : 1;                    // bx is 1 or 2
    n_a = (ax == 0) ? n0 : n1;
    n_b = (bx == 1) ? n1 : n2;
    return true;
  }

  /// cf times the bilinear interpolation of the image in plane ip (zero outside the image)
  WORKER_QUALIFIER float interp(const float *image, int ip) const
  {
    const float fa = ip * aa + ba;
    const float fb = ip * ab + bb;
    const int ia = static_cast<int>(floorf(fa));
    const int ib = static_cast<int>(floorf(fb));
    const float wa = fa - ia;
    const float wb = fb - ib;
    float val = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
      const int ja = ia + (q & 1);
      const int jb = ib + (q >> 1);
      if ((ja >= 0) && (ja < n_a) && (jb >= 0) && (jb < n_b))
        val += ((q & 1) ? wa : 1.0f - wa) * ((q >> 1) ? wb : 1.0f - wb) * image[ip * s_dir + ja * s_a + jb * s_b];
    }
    return cf * val;
  }

  /// adjoint of interp: add cf * val, distributed with the bilinear weights, to plane ip
  /// (atomic additions)
  WORKER_QUALIFIER void interp_adjoint(float *image, int ip, float val) const
  {
    const float fa = ip * aa + ba;
    const float fb = ip * ab + bb;
    const int ia = static_cast<int>(floorf(fa));
    const int ib = static_cast<int>(floorf(fb));
    const float wa = fa - ia;
    const float wb = fb - ib;
    val *= cf;
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
      const int ja = ia + (q & 1);
      const int jb = ib + (q >> 1);
      if ((ja >= 0) && (ja < n_a) && (jb >= 0) && (jb < n_b))
        atomic_sum(image + ip * s_dir + ja * s_a + jb * s_b, ((q & 1) ? wa : 1.0f - wa) * ((q >> 1) ? wb : 1.0f - wb) * val);
    }
  }

  /// TOF bin coordinate of plane ip is it_f = tof_plane_position(ip, at, bt). A positive
  /// TOF center offset shifts the TOF bins towards p1 (lor_end).
  WORKER_QUALIFIER void tof_line(const float *p0, const float *p1, float tof_bin_width, short num_tof_bins,
                                 float tof_center_offset, const float *image_origin, const float *voxel_size,
                                 float &at, float &bt) const
  {
    const float costheta = voxel_size[direction] / cf;
    // sign: whether the TOF bin numbers increase or decrease along the principal axis
    const float sign = (p1[direction] >= p0[direction]) ? 1.0f : -1.0f;
    // center of TOF bin 0 projected onto the principal axis
    const float tof_origin = 0.5f * (p0[direction] + p1[direction]) - sign * (0.5f * num_tof_bins - 0.5f) * (tof_bin_width * costheta) + sign * tof_center_offset * costheta;
    const float tof_slope = sign * tof_bin_width * costheta;
    at = sign * cf / tof_bin_width;
    bt = (image_origin[direction] - tof_origin) / tof_slope;
  }
};
