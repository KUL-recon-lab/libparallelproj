#include "parallelproj.h"
#include "joseph3d_tof_sino_back_worker.h"
#include "joseph3d_tof_sino_fwd_worker.h" // TOF_SINO_FWD_DISPATCH

void joseph3d_tof_sino_back(const float *lor_start,
                            const float *lor_end,
                            float *image,
                            const float *image_origin,
                            const float *voxel_size,
                            const float *projection_values,
                            std::size_t num_lors,
                            const int *image_dim,
                            float tof_bin_width,
                            const float *tof_sigma,
                            const float *tof_center_offset,
                            float num_sigmas,
                            short num_tof_bins,
                            unsigned char is_lor_dependent_tof_sigma,
                            unsigned char is_lor_dependent_tof_center_offset,
                            int /* device_id */,
                            int /* threads_per_block */,
                            void * /* stream: ignored without CUDA */)
{

  // Same per-LOR choice of buffer size W and load width V as the forward projection and
  // as on the GPU (see joseph3d_tof_sino_back.cu); the results do not depend on V.
#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
  {
    const float sigma = is_lor_dependent_tof_sigma ? tof_sigma[i] : tof_sigma[0];
    const float half_width_bins = num_sigmas * sigma / tof_bin_width;
    const int V = tof_store_width(half_width_bins, voxel_size, tof_bin_width);
    const int W = tof_acc_size_class(half_width_bins, V);
#define TOF_SINO_BACK_CALL(W_, V_)                                                                                 \
  joseph3d_tof_sino_back_worker<W_, V_>(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, \
                                        image_dim, tof_bin_width, tof_sigma, tof_center_offset, num_sigmas,       \
                                        num_tof_bins, is_lor_dependent_tof_sigma,                                 \
                                        is_lor_dependent_tof_center_offset)
    TOF_SINO_FWD_DISPATCH(W, V, TOF_SINO_BACK_CALL);
#undef TOF_SINO_BACK_CALL
  }
}
