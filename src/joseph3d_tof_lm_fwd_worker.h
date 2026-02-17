#pragma once
#include "cuda_compat.h"
#include "utils.h"

WORKER_QUALIFIER inline void joseph3d_tof_lm_fwd_worker(size_t i,
                                                        const float *xstart,
                                                        const float *xend,
                                                        const float *img,
                                                        const float *img_origin,
                                                        const float *voxsize,
                                                        float *p,
                                                        const int *img_dim,
                                                        float tofbin_width,
                                                        const float *sigma_tof,
                                                        const float *tofcenter_offset,
                                                        float n_sigmas,
                                                        const short *tofbin,
                                                        short n_tofbins,
                                                        unsigned char lor_dependent_sigma_tof,
                                                        unsigned char lor_dependent_tofcenter_offset)
{
  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  int direction;
  int i0, i1, i2;
  float i0_f, i1_f, i2_f;
  float cf;

  float a0, a1, a2;
  float b0, b1, b2;

  int istart = -1;
  int iend = -1;
  int it = tofbin[i];

  float d0 = xend[3 * i + 0] - xstart[3 * i + 0];
  float d1 = xend[3 * i + 1] - xstart[3 * i + 1];
  float d2 = xend[3 * i + 2] - xstart[3 * i + 2];

  float dr;

  // test whether the ray intersects the image cube
  // if it does not, istart and iend are set to -1
  // if it does, direction is set to the principal axis
  // and istart and iend are set to the first and last voxel planes
  // that are intersected
  // cf is the correction factor voxsize[dir]/cos[dir]
  ray_cube_intersection_joseph(xstart + 3 * i, xend + 3 * i, img_origin, voxsize, img_dim, direction, cf, istart, iend);

  // if the ray does not intersect the image cube, return
  // istart and iend are set to -1
  if (istart == -1)
  {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// calculate TOF-related parameters
  float toAdd = 0.0;                        // variable LM forward projection (to reduce access to p[i])
  float costheta = voxsize[direction] / cf; // cosine of angle between ray and principal axis

  // get the sigma_tof and tofcenter_offset for this LOR depending on whether they are constant or LOR-dependent
  float sig_tof = lor_dependent_sigma_tof ? sigma_tof[i] : sigma_tof[0];
  float tofcen_offset = lor_dependent_tofcenter_offset ? tofcenter_offset[i] : tofcenter_offset[0];

  // sign variable that indicated whether TOF bin numbers increase or decrease when
  // through the image along the principal axis direction
  float sign = (xend[3 * i + direction] >= xstart[3 * i + direction]) ? 1.0 : -1.0;

  // the center of the first TOF bin (TOF bin 0) projected onto the principal axis
  float tof_origin = 0.5 * (xstart[3 * i + direction] + xend[3 * i + direction]) - sign * (0.5 * n_tofbins - 0.5) * (tofbin_width * costheta) + tofcen_offset * costheta;
  // slope of TOF bin number as a function of distance along the principal axis
  // the position of the TOF bins projects onto the principal axis is: tof_origin + tof_bin_number*tof_slope
  float tof_slope = sign * tofbin_width * costheta;

  // the TOF bin number of intersection point of the ray with a given image plane along the principal axis is it_f = i*at + bt
  float at = sign * cf / tofbin_width;
  float bt = (img_origin[direction] - tof_origin) / tof_slope;

  // recompute istart and iend based on TOF bin and num_sigmas to consider
  // in LM processing, we don't step through all planes, but only those that are within n_sigmas*sig_tof of the TOF bin of the ray
  // NOTE: that istart and iend can be outside the image boundaries
  // NOTE: we need to loop over all planes between istart and iend even if they are outside the image boundaries, to make sure the TOF weights are normalized correctly
  istart = (int)floorf(((it - sign * n_sigmas * sig_tof / tofbin_width) - bt) / at);
  iend = (int)ceilf(((it + sign * n_sigmas * sig_tof / tofbin_width) - bt) / at);

  float it_f = istart * at + bt;

  float tof_plane_weight;
  float tof_plane_weights_sum = 0.0;

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  if (direction == 0)
  {
    dr = d0;

    a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr);
    b1 = (xstart[3 * i + 1] - img_origin[1] + d1 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[1];

    a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr);
    b2 = (xstart[3 * i + 2] - img_origin[2] + d2 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[2];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i1_f = istart * a1 + b1;
    i2_f = istart * a2 + b2;

    for (i0 = istart; i0 <= iend; ++i0)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tofbin_width, sig_tof, tofbin_width);
      tof_plane_weights_sum += tof_plane_weight;

      if (i0 >= 0 && i0 < n0)
      {
        toAdd += tof_plane_weight * bilinear_interp_fixed0(img, n0, n1, n2, i0, i1_f, i2_f);
      }

      i1_f += a1;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 1)
  {
    dr = d1;

    a0 = (d0 * voxsize[direction]) / (voxsize[0] * dr);
    b0 = (xstart[3 * i + 0] - img_origin[0] + d0 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[0];

    a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr);
    b2 = (xstart[3 * i + 2] - img_origin[2] + d2 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[2];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i2_f = istart * a2 + b2;

    for (i1 = istart; i1 <= iend; ++i1)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tofbin_width, sig_tof, tofbin_width);
      tof_plane_weights_sum += tof_plane_weight;

      if (i1 >= 0 && i1 < n1)
      {
        toAdd += tof_plane_weight * bilinear_interp_fixed1(img, n0, n1, n2, i0_f, i1, i2_f);
      }

      i0_f += a0;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 2)
  {
    dr = d2;

    a0 = (d0 * voxsize[direction]) / (voxsize[0] * dr);
    b0 = (xstart[3 * i + 0] - img_origin[0] + d0 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[0];

    a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr);
    b1 = (xstart[3 * i + 1] - img_origin[1] + d1 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[1];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i1_f = istart * a1 + b1;

    for (i2 = istart; i2 <= iend; ++i2)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tofbin_width, sig_tof, tofbin_width);
      tof_plane_weights_sum += tof_plane_weight;

      if (i2 >= 0 && i2 < n2)
      {
        toAdd += tof_plane_weight * bilinear_interp_fixed2(img, n0, n1, n2, i0_f, i1_f, i2);
      }

      i0_f += a0;
      i1_f += a1;
      it_f += at;
    }
  }

  // the tof-weighted interpolated values has to be corrected for the non-TOF correction factor (cf)
  // and also for the the fact that we truncated the plane range based on num sigmas
  // the expected sum of the TOF weights is (tofbin_width / cf)

  // p[i] = toAdd * cf * (tofbin_width / cf) / tof_plane_weights_sum;
  p[i] = toAdd * tofbin_width / tof_plane_weights_sum;
}
