#pragma once
#include "cuda_compat.h"
#include "utils.h"

WORKER_QUALIFIER inline void joseph3d_tof_lm_back_worker(size_t i,
                                                         const float *xstart,
                                                         const float *xend,
                                                         float *img,
                                                         const float *img_origin,
                                                         const float *voxsize,
                                                         const float *p,
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
  if (p[i] == 0)
  {
    return;
  }

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
  istart = static_cast<int>(floorf(((it - sign * n_sigmas * sig_tof / tofbin_width) - bt) / at));
  iend = static_cast<int>(ceilf(((it + sign * n_sigmas * sig_tof / tofbin_width) - bt) / at));

  float it_f = istart * at + bt;

  float tof_plane_weight;
  float tof_plane_weights_sum = 0.0;

  // first loop over all potentially contributing planes to get the sum of the TOF weights for normalization
  // this values depends on the number of sigmas, the tof bin width relative to sigma_tof, and the sampling location of the planes
  // so far I don't see a more direct way to calculate this normalization factor, so I think we have to loop through the planes first
  // in principle, we could buffer all tof_plane_weights here, but this would require quite some memory when executed in parallel
  for (i0 = istart; i0 <= iend; ++i0)
  {
    // TOF contribution
    tof_plane_weights_sum += effective_gaussian_tof_kernel(fabsf(it_f - it) * tofbin_width, sig_tof, tofbin_width);
    it_f += at;
  }

  // this is the "corrected" value up to the tof_plane_weight that we have to inject into the planes using the adjoint of the bilinear interpolation
  toAdd = p[i] * tofbin_width / tof_plane_weights_sum;

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

    // truncate istart and iend to the image boundaries
    istart = (istart < 0) ? 0 : istart;
    istart = (istart > n0 - 1) ? n0 - 1 : istart;

    iend = (iend < 0) ? 0 : iend;
    iend = (iend > n0 - 1) ? n0 - 1 : iend;

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i1_f = istart * a1 + b1;
    i2_f = istart * a2 + b2;
    // reset it_f for 2nd loop
    it_f = istart * at + bt;

    for (i0 = istart; i0 <= iend; ++i0)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tofbin_width, sig_tof, tofbin_width);
      bilinear_interp_adj_fixed0(img, n0, n1, n2, i0, i1_f, i2_f, toAdd * tof_plane_weight);

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

    // truncate istart and iend to the image boundaries
    istart = (istart < 0) ? 0 : istart;
    istart = (istart > n1 - 1) ? n1 - 1 : istart;

    iend = (iend < 0) ? 0 : iend;
    iend = (iend > n1 - 1) ? n1 - 1 : iend;

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i2_f = istart * a2 + b2;
    // reset it_f for 2nd loop
    it_f = istart * at + bt;

    for (i1 = istart; i1 <= iend; ++i1)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tofbin_width, sig_tof, tofbin_width);
      bilinear_interp_adj_fixed1(img, n0, n1, n2, i0_f, i1, i2_f, toAdd * tof_plane_weight);

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

    // truncate istart and iend to the image boundaries
    istart = (istart < 0) ? 0 : istart;
    istart = (istart > n2 - 1) ? n2 - 1 : istart;

    iend = (iend < 0) ? 0 : iend;
    iend = (iend > n2 - 1) ? n2 - 1 : iend;

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i1_f = istart * a1 + b1;
    // reset it_f for 2nd loop
    it_f = istart * at + bt;

    for (i2 = istart; i2 <= iend; ++i2)
    {
      // TOF contribution
      tof_plane_weight = effective_gaussian_tof_kernel(fabsf(it_f - it) * tofbin_width, sig_tof, tofbin_width);
      bilinear_interp_adj_fixed2(img, n0, n1, n2, i0_f, i1_f, i2, toAdd * tof_plane_weight);

      i0_f += a0;
      i1_f += a1;
      it_f += at;
    }
  }
}
