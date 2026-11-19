#include "parallelproj.h"
#include "joseph3d_tof_lm_fwd_worker.h"
#include "tof_kernel_cuda.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// one thread per event; KP: TOF kernel parameters (Gaussian or table, see tof_kernel.h)
template <typename KP>
__global__ void joseph3d_tof_lm_fwd_kernel(const float *lor_start, const float *lor_end, const float *image,
                              const float *image_origin, const float *voxel_size, float *projection_values,
                              std::size_t num_events, ImageDim image_dim, float tof_bin_width,
                              const float *tof_center_offset, const short *tof_bin_index, short num_tof_bins,
                              unsigned char is_lor_dependent_tof_center_offset, KP kp)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < num_events)
        joseph3d_tof_lm_fwd_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim.d,
                      tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins,
                      is_lor_dependent_tof_center_offset, kp);
}

// Implementation shared by joseph3d_tof_lm_fwd (Gaussian kernel) and joseph3d_tof_lm_fwd_tab (tabulated
// kernel): KP is TOFGaussParams or TOFTableParams with the caller's (host, device or
// managed) pointers.
template <typename KP>
static void joseph3d_tof_lm_fwd_impl(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                        const float *voxel_size, float *projection_values, std::size_t num_events, const int *image_dim,
                        float tof_bin_width, const float *tof_center_offset, const short *tof_bin_index, short num_tof_bins,
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
    handle_cuda_input_array(lor_start, d_lor_start, sizeof(float) * num_events * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(lor_end, d_lor_end, sizeof(float) * num_events * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(image, d_image, sizeof(float) * nvoxels, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(image_origin, d_image_origin, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(voxel_size, d_voxel_size, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, s);
    const std::size_t projection_values_size = sizeof(float) * num_events;
    handle_cuda_input_array(projection_values, d_projection_values, projection_values_size, device_id, cudaMemAdviseSetAccessedBy, s);
    handle_cuda_input_array(tof_center_offset, d_tof_center_offset, sizeof(float) * (is_lor_dependent_tof_center_offset ? num_events : 1), device_id, cudaMemAdviseSetReadMostly, s);
    CudaDevicePtr<short> d_tof_bin_index;
    handle_cuda_input_array(tof_bin_index, d_tof_bin_index, sizeof(short) * num_events, device_id, cudaMemAdviseSetReadMostly, s);
    // TOF kernel parameters (sigma, or table and kernel scale) on the device
    TOFParamsOnDevice<KP> kp_dev(kp, num_events, device_id, s);

    const int num_blocks = (int)((num_events + threads_per_block - 1) / threads_per_block);
    // Flush any stale (non-sticky) last-error state of the caller, so that the launch
    // check below does not misattribute it to this kernel launch.
    cudaGetLastError();
    if (num_events > 0)
    {
        joseph3d_tof_lm_fwd_kernel<KP><<<num_blocks, threads_per_block, 0, s>>>(
            d_lor_start.ptr, d_lor_end.ptr, d_image.ptr, d_image_origin.ptr, d_voxel_size.ptr, d_projection_values.ptr,
            num_events, h_image_dim, tof_bin_width, d_tof_center_offset.ptr, d_tof_bin_index.ptr, num_tof_bins,
            is_lor_dependent_tof_center_offset, kp_dev.params);
        cuda_check_launch();
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
                                      d_tof_bin_index.owns || kp_dev.owns());
}

//////////////////////////////////////////////////////////////////
// public functions (parallelproj.h)
//////////////////////////////////////////////////////////////////

// Gaussian TOF kernel
void joseph3d_tof_lm_fwd(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
            const float *voxel_size, float *projection_values, std::size_t num_events, const int *image_dim,
            float tof_bin_width, const float *tof_sigma, const float *tof_center_offset, float num_sigmas,
            const short *tof_bin_index, short num_tof_bins, unsigned char is_lor_dependent_tof_sigma,
            unsigned char is_lor_dependent_tof_center_offset, int device_id, int threads_per_block, void *stream)
{
    const TOFGaussParams kp{tof_sigma, is_lor_dependent_tof_sigma, num_sigmas, tof_bin_width};
    joseph3d_tof_lm_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_events, image_dim,
                tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins, is_lor_dependent_tof_center_offset, kp, device_id, threads_per_block, stream);
}

// tabulated TOF kernel
void joseph3d_tof_lm_fwd_tab(const float *lor_start, const float *lor_end, const float *image, const float *image_origin,
                const float *voxel_size, float *projection_values, std::size_t num_events, const int *image_dim,
                float tof_bin_width, const float *tof_center_offset, const short *tof_bin_index, short num_tof_bins,
                const float *tof_kernel_cdf, int tof_kernel_num_samples, float tof_kernel_u_min,
                float tof_kernel_u_max, const float *tof_kernel_scale,
                unsigned char is_lor_dependent_tof_center_offset, unsigned char is_lor_dependent_tof_kernel_scale,
                int device_id, int threads_per_block, void *stream)
{
    const TOFTableParams kp{tof_kernel_cdf, tof_kernel_num_samples, tof_kernel_u_min, tof_kernel_u_max,
                           tof_kernel_scale, is_lor_dependent_tof_kernel_scale, tof_bin_width};
    joseph3d_tof_lm_fwd_impl(lor_start, lor_end, image, image_origin, voxel_size, projection_values, num_events, image_dim,
                tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins, is_lor_dependent_tof_center_offset, kp, device_id, threads_per_block, stream);
}
