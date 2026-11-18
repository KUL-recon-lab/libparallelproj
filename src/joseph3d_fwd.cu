#include "parallelproj.h"
#include "joseph3d_fwd_worker.h"
#include "debug.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void joseph3d_fwd_kernel(const float *lor_start,
                                    const float *lor_end,
                                    const float *image,
                                    const float *image_origin,
                                    const float *voxel_size,
                                    float *projection_values,
                                    std::size_t num_lors,
                                    ImageDim image_dim)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_lors)
    {
        joseph3d_fwd_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim.d);
    }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void joseph3d_fwd(const float *lor_start,
                  const float *lor_end,
                  const float *image,
                  const float *image_origin,
                  const float *voxel_size,
                  float *projection_values,
                  std::size_t num_lors,
                  const int *image_dim,
                  int device_id,
                  int threads_per_block,
                  void *stream)
{
    // Use device_id (if >= 0) during this call and restore the caller's current
    // CUDA device afterwards (also on errors). Declared before the first CUDA call
    // and before all device buffers (RAII: buffers are freed before the device is restored).
    CudaDeviceGuard device_guard(device_id);

    // Stream of this call (nullptr: legacy default stream, synchronous).
    CudaCallStream call_stream(stream);

    // Image dimensions on the host (image_dim can be a host, device or managed pointer);
    // they are passed to the kernel by value.
    ImageDim h_image_dim = cuda_image_dim_to_host(image_dim, call_stream.stream);
    std::size_t nvoxels = h_image_dim.nvoxels();

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    // copy arrays to device if needed
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Handle lor_start (read mostly)
    CudaDevicePtr<float> d_lor_start;
    handle_cuda_input_array(lor_start, d_lor_start, sizeof(float) * num_lors * 3, device_id, cudaMemAdviseSetReadMostly, call_stream.stream);

    // Handle lor_end (read mostly)
    CudaDevicePtr<float> d_lor_end;
    handle_cuda_input_array(lor_end, d_lor_end, sizeof(float) * num_lors * 3, device_id, cudaMemAdviseSetReadMostly, call_stream.stream);

    // Handle image (read mostly)
    CudaDevicePtr<float> d_image;
    handle_cuda_input_array(image, d_image, sizeof(float) * nvoxels, device_id, cudaMemAdviseSetReadMostly, call_stream.stream);

    // Handle image_origin (read mostly)
    CudaDevicePtr<float> d_image_origin;
    handle_cuda_input_array(image_origin, d_image_origin, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, call_stream.stream);

    // Handle voxel_size (read mostly)
    CudaDevicePtr<float> d_voxel_size;
    handle_cuda_input_array(voxel_size, d_voxel_size, sizeof(float) * 3, device_id, cudaMemAdviseSetReadMostly, call_stream.stream);

    // Handle projection_values (write access)
    CudaDevicePtr<float> d_projection_values;
    handle_cuda_input_array(projection_values, d_projection_values, sizeof(float) * num_lors, device_id, cudaMemAdviseSetAccessedBy, call_stream.stream);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // launch the kernel
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

#ifdef DEBUG
    // get and print the current cuda device ID
    int current_device_id;
    cudaGetDevice(&current_device_id);
    DEBUG_PRINT("Using CUDA device: %d\n", current_device_id);
#endif

    int num_blocks = (int)((num_lors + threads_per_block - 1) / threads_per_block);
    // Flush any stale (non-sticky) last-error state - e.g. left behind by a
    // previously failed CUDA call of the CALLER (other libraries, earlier
    // failed calls into this library, ...). CUDA only resets the last error
    // when it is read, so without this flush the launch check below would
    // misattribute such a stale error to this kernel launch.
    cudaGetLastError();
    joseph3d_fwd_kernel<<<num_blocks, threads_per_block, 0, call_stream.stream>>>(d_lor_start.ptr, d_lor_end.ptr, d_image.ptr,
                                                           d_image_origin.ptr, d_voxel_size.ptr,
                                                           d_projection_values.ptr, num_lors, h_image_dim);
    cuda_check_launch();

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // If the output is a temporary device copy of a host array (owns == true),
    // copy the result back, queued on the call's stream. If the caller passed
    // device/managed memory, the kernel wrote to it directly.
    //
    // Then wait for the call's stream if the call is synchronous (stream ==
    // nullptr) or uses any host buffer; otherwise (asynchronous call with
    // device/managed arrays) return immediately.
    //
    // No explicit frees are needed here: every temporary device buffer is held
    // by a CudaDevicePtr (RAII, see cuda_utils.h), whose destructor frees it
    // stream-ordered when this function returns OR when an exception is thrown
    // anywhere above.
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    if (d_projection_values.owns)
        cuda_memcpy_d2h_async(projection_values, d_projection_values.ptr, sizeof(float) * num_lors, call_stream.stream);

    cuda_finish_call(call_stream, d_lor_start.owns ||
                     d_lor_end.owns ||
                     d_image.owns ||
                     d_image_origin.owns ||
                     d_voxel_size.owns ||
                     d_projection_values.owns);
}
