#pragma once
#include "cuda_compat.h"
#include "utils.h"

// Helper: compute TOF weights into caller buffer and scatter normalized contribution.
// No bounds check for MAX_NUM_TOF_WEIGHTS (caller must provide a large enough buffer).
WORKER_QUALIFIER static inline void _apply_fwd_tof_weights(
    float it_f,
    float max_tof_bin_diff,
    float tof_bin_width,
    float tof_sigma,
    float *tof_weights,       // buffer to hold TOF weights of size at least MAX_NUM_TOF_WEIGHTS
    float toAdd,              // normalized contribution
    float *projection_values, // projection output (size: num_lors * num_tof_bins)
    size_t lor_idx,           // LOR index i (use size_t to match caller)
    short num_tof_bins)
{
  int it_min = static_cast<int>(floorf(it_f - max_tof_bin_diff));
  int it_max = static_cast<int>(ceilf(it_f + max_tof_bin_diff));
  int n_tof_weights = it_max + 1 - it_min;

  float sum_weights = 0.0f;
  for (int k = 0; k < n_tof_weights; ++k)
  {
    float dist = fabsf(it_f - it_min - k) * tof_bin_width;
    tof_weights[k] = effective_gaussian_tof_kernel(dist, tof_sigma, tof_bin_width);
    sum_weights += tof_weights[k];
  }

  // normalize and scatter only into valid TOF bins
  toAdd /= sum_weights;
  int k_start = (it_min < 0) ? -it_min : 0;
  int k_end = ((it_min + n_tof_weights) > num_tof_bins) ? (num_tof_bins - it_min) : n_tof_weights;
  for (int k = k_start; k < k_end; ++k)
  {
    projection_values[lor_idx * num_tof_bins + it_min + k] += toAdd * tof_weights[k];
  }
}

WORKER_QUALIFIER inline void joseph3d_tof_sino_fwd_worker(size_t i,
                                                          const float *lor_start,
                                                          const float *lor_end,
                                                          const float *image,
                                                          const float *image_origin,
                                                          const float *voxel_size,
                                                          float *projection_values,
                                                          const int *image_dim,
                                                          float tof_bin_width,
                                                          const float *tof_sigma,
                                                          const float *tof_center_offset,
                                                          float num_sigmas,
                                                          short num_tof_bins,
                                                          unsigned char is_lor_dependent_tof_sigma,
                                                          unsigned char is_lor_dependent_tof_center_offset)
{
  int n0 = image_dim[0];
  int n1 = image_dim[1];
  int n2 = image_dim[2];

  int direction;
  int i0, i1, i2;
  float i0_f, i1_f, i2_f;
  float cf;

  float a0, a1, a2;
  float b0, b1, b2;

  int istart = -1;
  int iend = -1;

  float d0 = lor_end[3 * i + 0] - lor_start[3 * i + 0];
  float d1 = lor_end[3 * i + 1] - lor_start[3 * i + 1];
  float d2 = lor_end[3 * i + 2] - lor_start[3 * i + 2];

  float dr;

  // test whether the ray intersects the image cube
  // if it does not, istart and iend are set to -1
  // if it does, direction is set to the principal axis
  // and istart and iend are set to the first and last voxel planes
  // that are intersected
  // cf is the correction factor voxel_size[dir]/cos[dir]
  ray_cube_intersection_joseph(lor_start + 3 * i, lor_end + 3 * i, image_origin, voxel_size, image_dim, direction, cf, istart, iend);

  // if the ray does not intersect the image cube, return
  // istart and iend are set to -1
  if (istart == -1)
  {
    return;
  }

  for (short it = 0; it < num_tof_bins; ++it)
  {
    projection_values[i * num_tof_bins + it] = 0.0f;
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// calculate TOF-related parameters
  float toAdd;                                 // non-TOF contribution to the projection value for a given image plane
  float tof_weights[MAX_NUM_TOF_WEIGHTS];      // buffer to hold TOF weights for a given image plane, MAX_NUM_TOF_WEIGHTS is defined in utils.h
  float costheta = voxel_size[direction] / cf; // cosine of angle between ray and principal axis

  // get the tof_sigma and tof_center_offset for this LOR depending on whether they are constant or LOR-dependent
  float local_tof_sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
  float local_tof_center_offset = is_lor_dependent_tof_center_offset ? tof_center_offset[i] : tof_center_offset[0];

  // maximum number of TOF bins away from the current TOF bin to consider
  // TOF bins outside this range will have a negligible contribution and will be ignored
  float max_tof_bin_diff = num_sigmas * local_tof_sigma / tof_bin_width;

  // sign variable that indicated whether TOF bin numbers increase or decrease when
  // through the image along the principal axis direction
  float sign = (lor_end[3 * i + direction] >= lor_start[3 * i + direction]) ? 1.0 : -1.0;

  // the center of the first TOF bin (TOF bin 0) projected onto the principal axis
  float tof_origin = 0.5 * (lor_start[3 * i + direction] + lor_end[3 * i + direction]) - sign * (0.5 * num_tof_bins - 0.5) * (tof_bin_width * costheta) + local_tof_center_offset * costheta;
  // slope of TOF bin number as a function of distance along the principal axis
  // the position of the TOF bins projects onto the principal axis is: tof_origin + tof_bin_number*tof_slope
  float tof_slope = sign * tof_bin_width * costheta;

  // the TOF bin number of intersection point of the ray with a given image plane along the principal axis is it_f = i*at + bt
  float at = sign * cf / tof_bin_width;
  float bt = (image_origin[direction] - tof_origin) / tof_slope;
  float it_f = istart * at + bt;

  //////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  if (direction == 0)
  {
    dr = d0;

    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];

    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i1_f = istart * a1 + b1;
    i2_f = istart * a2 + b2;

    for (i0 = istart; i0 <= iend; ++i0)
    {
      // non-TOF contribution
      toAdd = cf * bilinear_interp_fixed0(image, n0, n1, n2, i0, i1_f, i2_f);
      _apply_fwd_tof_weights(it_f, max_tof_bin_diff, tof_bin_width, local_tof_sigma,
                             tof_weights, toAdd, projection_values, i, num_tof_bins);

      i1_f += a1;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 1)
  {
    dr = d1;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a2 = (d2 * voxel_size[direction]) / (voxel_size[2] * dr);
    b2 = (lor_start[3 * i + 2] - image_origin[2] + d2 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[2];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i2_f = istart * a2 + b2;

    for (i1 = istart; i1 <= iend; ++i1)
    {
      // non-TOF contribution
      toAdd = cf * bilinear_interp_fixed1(image, n0, n1, n2, i0_f, i1, i2_f);

      _apply_fwd_tof_weights(it_f, max_tof_bin_diff, tof_bin_width, local_tof_sigma,
                             tof_weights, toAdd, projection_values, i, num_tof_bins);

      i0_f += a0;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 2)
  {
    dr = d2;

    a0 = (d0 * voxel_size[direction]) / (voxel_size[0] * dr);
    b0 = (lor_start[3 * i + 0] - image_origin[0] + d0 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[0];

    a1 = (d1 * voxel_size[direction]) / (voxel_size[1] * dr);
    b1 = (lor_start[3 * i + 1] - image_origin[1] + d1 * (image_origin[direction] - lor_start[3 * i + direction]) / dr) / voxel_size[1];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i1_f = istart * a1 + b1;

    for (i2 = istart; i2 <= iend; ++i2)
    {
      // non-TOF contribution
      toAdd = cf * bilinear_interp_fixed2(image, n0, n1, n2, i0_f, i1_f, i2);

      _apply_fwd_tof_weights(it_f, max_tof_bin_diff, tof_bin_width, local_tof_sigma,
                             tof_weights, toAdd, projection_values, i, num_tof_bins);

      i0_f += a0;
      i1_f += a1;
      it_f += at;
    }
  }
}
