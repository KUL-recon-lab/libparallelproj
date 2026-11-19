#pragma once
// Host-side helper of the CUDA TOF projectors (only used in .cu files): make the arrays of
// the per-call TOF kernel parameters (TOFGaussParams, TOFTableParams, see tof_kernel.h)
// available on the GPU.
//
// The caller can pass host, device or managed pointers. Like all other input arrays (see
// handle_cuda_input_array in cuda_utils.h), host arrays are copied to a temporary device
// buffer, which is freed (stream-ordered) when the TOFParamsOnDevice object goes out of
// scope. Construct it after the CudaDeviceGuard of the call, so that the buffers are
// freed before the caller's device is restored.
#include "tof_kernel.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

template <typename KP>
struct TOFParamsOnDevice;

/// Gaussian kernel: copy tof_sigma (1 or num values) to the device if needed
template <>
struct TOFParamsOnDevice<TOFGaussParams>
{
  CudaDevicePtr<float> d_tof_sigma;
  TOFGaussParams params; ///< same parameters, with device pointers

  TOFParamsOnDevice(const TOFGaussParams &p, std::size_t num, int device_id, cudaStream_t stream) : params(p)
  {
    handle_cuda_input_array(p.tof_sigma, d_tof_sigma, sizeof(float) * (p.lor_dependent ? num : 1), device_id,
                            cudaMemAdviseSetReadMostly, stream);
    params.tof_sigma = d_tof_sigma.ptr;
  }
  /// whether a host array was copied (the call then has to wait for the GPU before returning)
  bool owns() const { return d_tof_sigma.owns; }
};

/// Tabulated kernel: copy the table and the kernel scale (1 or num values) to the device if needed
template <>
struct TOFParamsOnDevice<TOFTableParams>
{
  CudaDevicePtr<float> d_table;
  CudaDevicePtr<float> d_scale;
  TOFTableParams params; ///< same parameters, with device pointers

  TOFParamsOnDevice(const TOFTableParams &p, std::size_t num, int device_id, cudaStream_t stream) : params(p)
  {
    handle_cuda_input_array(p.table, d_table, sizeof(float) * p.n, device_id, cudaMemAdviseSetReadMostly, stream);
    handle_cuda_input_array(p.scale, d_scale, sizeof(float) * (p.lor_dependent ? num : 1), device_id,
                            cudaMemAdviseSetReadMostly, stream);
    params.table = d_table.ptr;
    params.scale = d_scale.ptr;
  }
  bool owns() const { return d_table.owns || d_scale.owns; }
};
