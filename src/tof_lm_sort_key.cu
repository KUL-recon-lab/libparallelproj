#include "parallelproj.h"
#include "tof_lm_sort_key_worker.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Sort keys of TOF listmode events, CUDA: one thread per event (see
// tof_lm_sort_key_worker.h for the meaning of the key). Array handling as in the
// projectors (cuda_utils.h): device / managed arrays are used directly, host arrays are
// copied to temporary device buffers (and the keys back).
__global__ void tof_lm_sort_key_kernel(const float *lor_start, const float *lor_end, const float *image_origin,
                                       const float *voxel_size, std::int64_t *sort_key, std::size_t num_events,
                                       float tof_bin_width, const float *tof_center_offset,
                                       const short *tof_bin_index, short num_tof_bins,
                                       unsigned char is_lor_dependent_tof_center_offset)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < num_events)
        tof_lm_sort_key_worker(i, lor_start, lor_end, image_origin, voxel_size, sort_key, tof_bin_width,
                               tof_center_offset, tof_bin_index, num_tof_bins, is_lor_dependent_tof_center_offset);
}

void tof_lm_sort_key(const float *lor_start, const float *lor_end, const float *image_origin, const float *voxel_size,
                     std::int64_t *sort_key, std::size_t num_events, float tof_bin_width,
                     const float *tof_center_offset, const short *tof_bin_index, short num_tof_bins,
                     unsigned char is_lor_dependent_tof_center_offset, int device_id, int threads_per_block,
                     void *stream)
{
    // use device_id during this call and restore the caller's current device afterwards;
    // declared before all device buffers (RAII: buffers are freed before the device is restored)
    CudaDeviceGuard device_guard(device_id);
    // stream of this call (nullptr: legacy default stream, synchronous)
    CudaCallStream call_stream(stream);
    const cudaStream_t s = call_stream.stream;

    CudaDevicePtr<float> d_lor_start, d_lor_end, d_image_origin, d_voxel_size, d_tof_center_offset;
    handle_cuda_input_array(lor_start, d_lor_start, sizeof(float) * num_events * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(lor_end, d_lor_end, sizeof(float) * num_events * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(image_origin, d_image_origin, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(voxel_size, d_voxel_size, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, s);
    handle_cuda_input_array(tof_center_offset, d_tof_center_offset,
                            sizeof(float) * (is_lor_dependent_tof_center_offset ? num_events : 1), device_id,
                            cudaMemAdviseSetReadMostly, s);
    CudaDevicePtr<short> d_tof_bin_index;
    handle_cuda_input_array(tof_bin_index, d_tof_bin_index, sizeof(short) * num_events, device_id, cudaMemAdviseSetReadMostly, s);
    // output (write access)
    CudaDevicePtr<std::int64_t> d_sort_key;
    const std::size_t sort_key_size = sizeof(std::int64_t) * num_events;
    handle_cuda_input_array(sort_key, d_sort_key, sort_key_size, device_id, cudaMemAdviseSetAccessedBy, s);

    // flush a stale (non-sticky) last error of the caller, so that the launch check below
    // does not misattribute it to this kernel
    cudaGetLastError();
    if (num_events > 0)
    {
        const int num_blocks = (int)((num_events + threads_per_block - 1) / threads_per_block);
        tof_lm_sort_key_kernel<<<num_blocks, threads_per_block, 0, s>>>(
            d_lor_start.ptr, d_lor_end.ptr, d_image_origin.ptr, d_voxel_size.ptr, d_sort_key.ptr, num_events,
            tof_bin_width, d_tof_center_offset.ptr, d_tof_bin_index.ptr, num_tof_bins,
            is_lor_dependent_tof_center_offset);
        cuda_check_launch();
    }

    // host output: copy the keys back (queued on the call's stream); then wait for the
    // stream if the call is synchronous or uses host buffers
    if (d_sort_key.owns)
        cuda_memcpy_d2h_async(sort_key, d_sort_key.ptr, sort_key_size, s);
    cuda_finish_call(call_stream, d_lor_start.owns || d_lor_end.owns || d_image_origin.owns || d_voxel_size.owns ||
                                      d_tof_center_offset.owns || d_tof_bin_index.owns || d_sort_key.owns);
}
