#include "parallelproj.h"
#include "tof_lm_point_back_worker.h"

void tof_lm_point_back(const float *lor_start,
                       const float *lor_end,
                       float *image,
                       const float *image_origin,
                       const float *voxel_size,
                       const float *projection_values,
                       std::size_t num_events,
                       const int *image_dim,
                       float tof_bin_width,
                       const float *tof_center_offset,
                       const short *tof_bin_index,
                       short num_tof_bins,
                       unsigned char is_lor_dependent_tof_center_offset,
                       int device_id,
                       int threads_per_block,
                       void * /* stream: ignored without CUDA */)
{

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_events); ++i)
  {
    tof_lm_point_back_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, tof_bin_width,
                             tof_center_offset, tof_bin_index, num_tof_bins,
                             is_lor_dependent_tof_center_offset);
  }
}
