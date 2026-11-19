#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"

// the internal limit (tof_kernel.h) must equal the documented public constant
static_assert(TOF_MAX_WINDOW_BINS == PARALLELPROJ_TOF_MAX_WINDOW_BINS, "TOF window limit mismatch");

// Implementation shared by joseph3d_tof_sino_fwd (Gaussian kernel) and joseph3d_tof_sino_fwd_tab (tabulated
// kernel), CPU / OpenMP: one loop iteration per LOR. KP is TOFGaussParams or TOFTableParams.
template <typename KP>
static void joseph3d_tof_sino_fwd_impl(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                        const float *voxel_size, float *projection_values, std::size_t num_lors, const int *image_dim,
                        float tof_bin_width, const float *tof_center_offset, short num_tof_bins,
                        unsigned char is_lor_dependent_tof_center_offset, const KP &kp)
{
  // For every LOR: choose the accumulator size W (from the width of the TOF kernel
  // support) and the store width V exactly as on the GPU (see joseph3d_tof_sino_fwd.cu and
  // tof_kernel.h), and call the worker template compiled for (W, V). Using the same choice
  // as on the GPU means that the CPU build runs and tests the same code paths. On the CPU,
  // a group of 4 bins is accessed with 4 normal loads / stores; the results do not depend
  // on V.
#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(num_lors); ++i)
  {
    const float support_bins = kp.support_bins(i);
    const int V = tof_store_width(support_bins, voxel_size, tof_bin_width, num_tof_bins);
    const int W = tof_acc_size_class(support_bins, V, num_tof_bins);
#define TOF_CALL(W_, V_)                                                                                    \
  joseph3d_tof_sino_fwd_worker<W_, V_>(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, \
                        tof_bin_width, tof_center_offset, num_tof_bins, is_lor_dependent_tof_center_offset, kp)
    TOF_SINO_FWD_DISPATCH(W, V, TOF_CALL);
#undef TOF_CALL
  }
}

//////////////////////////////////////////////////////////////////
// public functions (parallelproj.h)
//////////////////////////////////////////////////////////////////

// Gaussian TOF kernel
void joseph3d_tof_sino_fwd(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
            const float *voxel_size, float *projection_values, std::size_t num_lors, const int *image_dim,
            float tof_bin_width, const float *tof_sigma, const float *tof_center_offset, float num_sigmas,
            short num_tof_bins, unsigned char is_lor_dependent_tof_sigma,
            unsigned char is_lor_dependent_tof_center_offset, int /* device_id */, int /* threads_per_block */,
            void * /* stream: ignored without CUDA */)
{
  const TOFGaussParams kp{tof_sigma, is_lor_dependent_tof_sigma, num_sigmas, tof_bin_width};
  joseph3d_tof_sino_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim,
              tof_bin_width, tof_center_offset, num_tof_bins, is_lor_dependent_tof_center_offset, kp);
}

// tabulated TOF kernel
void joseph3d_tof_sino_fwd_tab(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                const float *voxel_size, float *projection_values, std::size_t num_lors, const int *image_dim,
                float tof_bin_width, const float *tof_center_offset, short num_tof_bins,
                const float *tof_kernel_cdf, int tof_kernel_num_samples, float tof_kernel_u_min,
                float tof_kernel_u_max, const float *tof_kernel_scale,
                unsigned char is_lor_dependent_tof_center_offset, unsigned char is_lor_dependent_tof_kernel_scale,
                int /* device_id */, int /* threads_per_block */,
            void * /* stream: ignored without CUDA */)
{
  const TOFTableParams kp{tof_kernel_cdf, tof_kernel_num_samples, tof_kernel_u_min, tof_kernel_u_max,
                         tof_kernel_scale, is_lor_dependent_tof_kernel_scale, tof_bin_width};
  joseph3d_tof_sino_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim,
              tof_bin_width, tof_center_offset, num_tof_bins, is_lor_dependent_tof_center_offset, kp);
}
