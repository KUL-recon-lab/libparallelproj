#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"
#include "tof_kernel_cuda.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// How the GPU kernels are organized
// ----------------------------------
// The worker (joseph3d_tof_sino_fwd_worker.h) is a template in the accumulator size W,
// the store width V and the TOF kernel type KP (Gaussian or table), so there is one
// compiled GPU kernel per combination (W, V) (9 sizes x 2 widths = 18 kernels per kernel
// type). Which combination a LOR needs depends on the width of its TOF kernel support
// (window size) and on voxel size / TOF bin width (store width, see tof_kernel.h).
//
// The host cannot simply read sigma (or the kernel scale) and choose one kernel: these
// are GPU arrays (also when they hold a single value), and reading them would force the
// CPU to wait for the GPU (a synchronization, which we avoid so that calls can run
// asynchronously on streams). Moreover, they can be different for every LOR.
//
// Solution: the projector launches all 18 kernels, one after the other on the same
// stream. Every launch has one thread per LOR. A thread first computes the class (W, V)
// of its LOR (cheap: one value of sigma / scale) and returns immediately if it is not
// the class of this launch. So every LOR is projected exactly once, by the launch of its
// class. The 17 launches that return immediately cost very little (measured: < 1 %).
//
// (On the CPU, the loop over the LORs simply calls the right template for each LOR, see
// joseph3d_tof_sino_fwd.cpp.)
template <int W, int V, typename KP>
__global__ void joseph3d_tof_sino_fwd_kernel(const float *lor_start, const float *lor_end, const float *image,
                              const float *image_origin, const float *voxel_size, float *projection_values,
                              std::size_t num_lors, ImageDim image_dim, float tof_bin_width,
                              const float *tof_center_offset, short num_tof_bins,
                              unsigned char is_lor_dependent_tof_center_offset, KP kp)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_lors)
        return;

    // class (W, V) of this LOR; return if it belongs to another launch
    const float support_bins = kp.support_bins(i);
    if (tof_acc_size_class(support_bins, V, num_tof_bins) != W)
        return;
    if (tof_store_width(support_bins, voxel_size, tof_bin_width, num_tof_bins) != V)
        return;

    joseph3d_tof_sino_fwd_worker<W, V>(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim.d,
                        tof_bin_width, tof_center_offset, num_tof_bins, is_lor_dependent_tof_center_offset, kp);
}

// Implementation shared by joseph3d_tof_sino_fwd (Gaussian kernel) and joseph3d_tof_sino_fwd_tab (tabulated
// kernel): KP is TOFGaussParams or TOFTableParams with the caller's (host, device or
// managed) pointers.
template <typename KP>
static void joseph3d_tof_sino_fwd_impl(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                        const float *voxel_size, float *projection_values, std::size_t num_lors, const int *image_dim,
                        float tof_bin_width, const float *tof_center_offset, short num_tof_bins,
                        unsigned char is_lor_dependent_tof_center_offset, const KP &kp, int device_id,
                        int threads_per_block, void *stream)
{
    // Use device_id (if >= 0) during this call and restore the caller's current CUDA device
    // afterwards (also on errors). Declared before the first CUDA call and before all device
    // buffers (RAII: buffers are freed before the device is restored).
    CudaDeviceGuard device_guard(device_id);

    // Stream of this call (nullptr: legacy default stream, synchronous).
    CudaCallStream call_stream(stream);

    // Image dimensions on the host (image_dim can be a host, device or managed pointer);
    // they are passed to the kernel by value.
    ImageDim h_image_dim = cuda_image_dim_to_host(image_dim, call_stream.stream);
    const std::size_t nvoxels = h_image_dim.nvoxels();
    const cudaStream_t s = call_stream.stream;

    // copy the input / output arrays to the device if needed (host arrays: temporary
    // device copies, freed automatically, see cuda_utils.h)
    CudaDevicePtr<float> d_lor_start, d_lor_end, d_image, d_image_origin, d_voxel_size, d_projection_values, d_tof_center_offset;
    handle_cuda_input_array(lor_start, d_lor_start, sizeof(float) * num_lors * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(lor_end, d_lor_end, sizeof(float) * num_lors * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(image, d_image, sizeof(float) * nvoxels, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(image_origin, d_image_origin, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(voxel_size, d_voxel_size, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, s);
    const std::size_t projection_values_size = sizeof(float) * num_lors * num_tof_bins;
    handle_cuda_input_array(projection_values, d_projection_values, projection_values_size, device_id, cudaMemAdviseSetAccessedBy, s);
    handle_cuda_input_array(tof_center_offset, d_tof_center_offset, sizeof(float) * (is_lor_dependent_tof_center_offset ? num_lors : 1), device_id, cudaMemAdviseSetReadMostly, s);
    // TOF kernel parameters (sigma, or table and kernel scale) on the device
    TOFParamsOnDevice<KP> kp_dev(kp, num_lors, device_id, s);

    const int num_blocks = (int)((num_lors + threads_per_block - 1) / threads_per_block);
    // Flush any stale (non-sticky) last-error state of the caller, so that the launch
    // check below does not misattribute it to this kernel launch.
    cudaGetLastError();
    if (num_lors > 0)
    {
        // one launch per (accumulator size, store width), all on the call's stream
#define TOF_LAUNCH(W_, V_)                                                                                           \
    joseph3d_tof_sino_fwd_kernel<W_, V_, KP><<<num_blocks, threads_per_block, 0, s>>>(                                              \
        d_lor_start.ptr, d_lor_end.ptr, d_image.ptr, d_image_origin.ptr, d_voxel_size.ptr, d_projection_values.ptr, \
        num_lors, h_image_dim, tof_bin_width, d_tof_center_offset.ptr, num_tof_bins,                                \
        is_lor_dependent_tof_center_offset, kp_dev.params)
        for (int k = 0; k < TOF_NUM_ACC_SIZES; ++k)
        {
            TOF_SINO_FWD_DISPATCH(tof_acc_size(k), 1, TOF_LAUNCH);
            cuda_check_launch();
            TOF_SINO_FWD_DISPATCH(tof_acc_size(k), 4, TOF_LAUNCH);
            cuda_check_launch();
        }
#undef TOF_LAUNCH
    }

    ////////////////////////////////////////////////////////////////////////////
    // If the output is a temporary device copy of a host array, copy the result back
    // (queued on the call's stream).
    // Then wait for the call's stream if the call is synchronous (stream ==
    // nullptr) or uses any host buffer; otherwise (asynchronous call with
    // device/managed arrays) return immediately. Temporary device buffers are
    // freed by their CudaDevicePtr destructors (RAII, also on exceptions).
    ////////////////////////////////////////////////////////////////////////////
    if (d_projection_values.owns)
        cuda_memcpy_d2h_async(projection_values, d_projection_values.ptr, projection_values_size, s);
    cuda_finish_call(call_stream, d_lor_start.owns || d_lor_end.owns || d_image.owns || d_image_origin.owns ||
                                      d_voxel_size.owns || d_projection_values.owns || d_tof_center_offset.owns ||
                                      kp_dev.owns());
}

//////////////////////////////////////////////////////////////////
// public functions (parallelproj.h)
//////////////////////////////////////////////////////////////////

// Gaussian TOF kernel
void joseph3d_tof_sino_fwd(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
            const float *voxel_size, float *projection_values, std::size_t num_lors, const int *image_dim,
            float tof_bin_width, const float *tof_sigma, const float *tof_center_offset, float num_sigmas,
            short num_tof_bins, unsigned char is_lor_dependent_tof_sigma,
            unsigned char is_lor_dependent_tof_center_offset, int device_id, int threads_per_block, void *stream)
{
    const TOFGaussParams kp{tof_sigma, is_lor_dependent_tof_sigma, num_sigmas, tof_bin_width};
    joseph3d_tof_sino_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim,
                tof_bin_width, tof_center_offset, num_tof_bins, is_lor_dependent_tof_center_offset, kp, device_id, threads_per_block, stream);
}

// tabulated TOF kernel
void joseph3d_tof_sino_fwd_tab(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                const float *voxel_size, float *projection_values, std::size_t num_lors, const int *image_dim,
                float tof_bin_width, const float *tof_center_offset, short num_tof_bins,
                const float *tof_kernel_cdf, int tof_kernel_num_samples, float tof_kernel_u_min,
                float tof_kernel_u_max, const float *tof_kernel_scale,
                unsigned char is_lor_dependent_tof_center_offset, unsigned char is_lor_dependent_tof_kernel_scale,
                int device_id, int threads_per_block, void *stream)
{
    const TOFTableParams kp{tof_kernel_cdf, tof_kernel_num_samples, tof_kernel_u_min, tof_kernel_u_max,
                           tof_kernel_scale, is_lor_dependent_tof_kernel_scale, tof_bin_width};
    joseph3d_tof_sino_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_lors, image_dim,
                tof_bin_width, tof_center_offset, num_tof_bins, is_lor_dependent_tof_center_offset, kp, device_id, threads_per_block, stream);
}
