#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void joseph3d_tof_sino_fwd_kernel(const float *lor_start,
                                             const float *lor_end,
                                             const float *image,
                                             const float *image_origin,
                                             const float *voxel_size,
                                             float *projection_values,
                                             size_t num_lors,
                                             const int *image_dim,
                                             float tof_bin_width,
                                             const float *tof_sigma,
                                             const float *tof_center_offset,
                                             float num_sigmas,
                                             short num_tof_bins,
                                             unsigned char is_lor_dependent_tof_sigma,
                                             unsigned char is_lor_dependent_tof_center_offset)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_lors)
    {
        joseph3d_tof_sino_fwd_worker(i, lor_start, lor_end, image, image_origin, voxel_size, projection_values, image_dim, tof_bin_width,
                                     tof_sigma, tof_center_offset, num_sigmas, num_tof_bins,
                                     is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
    }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void joseph3d_tof_sino_fwd(const float *lor_start,
                           const float *lor_end,
                           const float *image,
                           const float *image_origin,
                           const float *voxel_size,
                           float *projection_values,
                           size_t num_lors,
                           const int *image_dim,
                           float tof_bin_width,
                           const float *tof_sigma,
                           const float *tof_center_offset,
                           float num_sigmas,
                           short num_tof_bins,
                           unsigned char is_lor_dependent_tof_sigma,
                           unsigned char is_lor_dependent_tof_center_offset,
                           int device_id,
                           int threads_per_block)
{
    // Calculate nvoxels from image_dim - image_dim can be device pointer!
    size_t nvoxels = cuda_nvoxels_from_img_dim(image_dim);

    // Set the CUDA device
    if (device_id >= 0)
    {
        cudaSetDevice(device_id);
    }

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    // copy arrays to device if needed
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Handle lor_start (read mostly)
    float *d_lor_start = nullptr;
    bool free_lor_start = false;
    handle_cuda_input_array(lor_start, &d_lor_start, sizeof(float) * num_lors * 3, free_lor_start, device_id, cudaMemAdviseSetReadMostly);

    // Handle lor_end (read mostly)
    float *d_lor_end = nullptr;
    bool free_lor_end = false;
    handle_cuda_input_array(lor_end, &d_lor_end, sizeof(float) * num_lors * 3, free_lor_end, device_id, cudaMemAdviseSetReadMostly);

    // Handle image (read mostly)
    float *d_image = nullptr;
    bool free_image = false;
    handle_cuda_input_array(image, &d_image, sizeof(float) * nvoxels, free_image, device_id, cudaMemAdviseSetReadMostly);

    // Handle image_origin (read mostly)
    float *d_image_origin = nullptr;
    bool free_image_origin = false;
    handle_cuda_input_array(image_origin, &d_image_origin, sizeof(float) * 3, free_image_origin, device_id, cudaMemAdviseSetReadMostly);

    // Handle voxel_size (read mostly)
    float *d_voxel_size = nullptr;
    bool free_voxel_size = false;
    handle_cuda_input_array(voxel_size, &d_voxel_size, sizeof(float) * 3, free_voxel_size, device_id, cudaMemAdviseSetReadMostly);

    // Handle projection_values (write access) - size depends on TOF bins
    size_t projection_values_size = sizeof(float) * num_lors * num_tof_bins;
    float *d_projection_values = nullptr;
    bool free_projection_values = false;
    handle_cuda_input_array(projection_values, &d_projection_values, projection_values_size, free_projection_values, device_id, cudaMemAdviseSetAccessedBy);

    // Handle image_dim (read mostly)
    int *d_image_dim = nullptr;
    bool free_image_dim = false;
    handle_cuda_input_array(image_dim, &d_image_dim, sizeof(int) * 3, free_image_dim, device_id, cudaMemAdviseSetReadMostly);

    // Handle tof_sigma (read mostly)
    float *d_tof_sigma = nullptr;
    bool free_tof_sigma = false;
    size_t tof_sigma_size = is_lor_dependent_tof_sigma ? sizeof(float) * num_lors : sizeof(float);
    handle_cuda_input_array(tof_sigma, &d_tof_sigma, tof_sigma_size, free_tof_sigma, device_id, cudaMemAdviseSetReadMostly);

    // Handle tof_center_offset (read mostly)
    float *d_tof_center_offset = nullptr;
    bool free_tof_center_offset = false;
    size_t tof_center_offset_size = is_lor_dependent_tof_center_offset ? sizeof(float) * num_lors : sizeof(float);
    handle_cuda_input_array(tof_center_offset, &d_tof_center_offset, tof_center_offset_size, free_tof_center_offset, device_id, cudaMemAdviseSetReadMostly);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // launch the kernel
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    int num_blocks = (int)((num_lors + threads_per_block - 1) / threads_per_block);
    joseph3d_tof_sino_fwd_kernel<<<num_blocks, threads_per_block>>>(
        d_lor_start, d_lor_end, d_image, d_image_origin, d_voxel_size, d_projection_values, num_lors, d_image_dim,
        tof_bin_width, d_tof_sigma, d_tof_center_offset, num_sigmas, num_tof_bins,
        is_lor_dependent_tof_sigma, is_lor_dependent_tof_center_offset);
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // free device memory if needed
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // Free device memory if it was allocated
    if (free_lor_start)
        cudaFree(d_lor_start);
    if (free_lor_end)
        cudaFree(d_lor_end);
    if (free_image)
        cudaFree(d_image);
    if (free_image_origin)
        cudaFree(d_image_origin);
    if (free_voxel_size)
        cudaFree(d_voxel_size);
    if (free_projection_values)
    {
        cudaMemcpy(projection_values, d_projection_values, projection_values_size, cudaMemcpyDeviceToHost);
        cudaFree(d_projection_values);
    }
    if (free_image_dim)
        cudaFree(d_image_dim);
    if (free_tof_sigma)
        cudaFree(d_tof_sigma);
    if (free_tof_center_offset)
        cudaFree(d_tof_center_offset);
}
