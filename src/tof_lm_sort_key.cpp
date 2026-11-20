#include "parallelproj.h"
#include "tof_lm_sort_key_worker.h"

// Sort keys of TOF listmode events, CPU / OpenMP: one loop iteration per event (see
// tof_lm_sort_key_worker.h for the meaning of the key).
void tof_lm_sort_key(const float *lor_start, const float *lor_end, const float *image_origin, const float *voxel_size,
                     std::int64_t *sort_key, std::size_t num_events, float tof_bin_width,
                     const float *tof_center_offset, const short *tof_bin_index, short num_tof_bins,
                     unsigned char is_lor_dependent_tof_center_offset, int /* device_id */,
                     int /* threads_per_block */, void * /* stream: ignored without CUDA */)
{
#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_events); ++i)
    tof_lm_sort_key_worker(i, lor_start, lor_end, image_origin, voxel_size, sort_key, tof_bin_width,
                           tof_center_offset, tof_bin_index, num_tof_bins, is_lor_dependent_tof_center_offset);
}
