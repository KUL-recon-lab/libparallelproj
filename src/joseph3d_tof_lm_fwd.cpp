#include "parallelproj.h"
#include "joseph3d_tof_lm_fwd_worker.h"

// Implementation shared by joseph3d_tof_lm_fwd (Gaussian kernel) and joseph3d_tof_lm_fwd_tab (tabulated
// kernel), CPU / OpenMP: one loop iteration per event. KP is TOFGaussParams or TOFTableParams.
template <typename KP>
static void joseph3d_tof_lm_fwd_impl(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                        const float *voxel_size, float *projection_values, std::size_t num_events, const int *image_dim,
                        float tof_bin_width, const float *tof_center_offset, const short *tof_bin_index, short num_tof_bins,
                        unsigned char is_lor_dependent_tof_center_offset, const KP &kp)
{
#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_events); ++i)
    joseph3d_tof_lm_fwd_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim,
                  tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins, is_lor_dependent_tof_center_offset, kp);
}

//////////////////////////////////////////////////////////////////
// public functions (parallelproj.h)
//////////////////////////////////////////////////////////////////

// Gaussian TOF kernel
void joseph3d_tof_lm_fwd(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
            const float *voxel_size, float *projection_values, std::size_t num_events, const int *image_dim,
            float tof_bin_width, const float *tof_sigma, const float *tof_center_offset, float num_sigmas,
            const short *tof_bin_index, short num_tof_bins, unsigned char is_lor_dependent_tof_sigma,
            unsigned char is_lor_dependent_tof_center_offset, int /* device_id */, int /* threads_per_block */,
            void * /* stream: ignored without CUDA */)
{
  const TOFGaussParams kp{tof_sigma, is_lor_dependent_tof_sigma, num_sigmas, tof_bin_width};
  joseph3d_tof_lm_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_events, image_dim,
              tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins, is_lor_dependent_tof_center_offset, kp);
}

// tabulated TOF kernel
void joseph3d_tof_lm_fwd_tab(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                const float *voxel_size, float *projection_values, std::size_t num_events, const int *image_dim,
                float tof_bin_width, const float *tof_center_offset, const short *tof_bin_index, short num_tof_bins,
                const float *tof_kernel_cdf, int tof_kernel_num_samples, float tof_kernel_u_min,
                float tof_kernel_u_max, const float *tof_kernel_scale,
                unsigned char is_lor_dependent_tof_center_offset, unsigned char is_lor_dependent_tof_kernel_scale,
                int /* device_id */, int /* threads_per_block */,
            void * /* stream: ignored without CUDA */)
{
  const TOFTableParams kp{tof_kernel_cdf, tof_kernel_num_samples, tof_kernel_u_min, tof_kernel_u_max,
                         tof_kernel_scale, is_lor_dependent_tof_kernel_scale, tof_bin_width};
  joseph3d_tof_lm_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_events, image_dim,
              tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins, is_lor_dependent_tof_center_offset, kp);
}
