#include "cuda_utils.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>

#if CUDART_VERSION >= 13000
static inline cudaMemLocation make_mem_location(int device_id) {
  cudaMemLocation location;
  location.type = cudaMemLocationTypeDevice;
  location.id = device_id;
  return location;
}
#endif

// Best-effort prefetch / memory-advise hints for CUDA managed memory.
// Failures here are non-fatal (e.g. platforms without concurrent managed
// access support), but the CUDA runtime's last-error state must be cleared
// with cudaGetLastError(), otherwise a stale error would later be
// misattributed to the next kernel launch check.
static void apply_managed_memory_hints(void *ptr, std::size_t size,
                                       int device_id,
                                       cudaMemoryAdvise memory_hint,
                                       cudaStream_t stream)
{
  int target_device = device_id;
  if (target_device < 0) {
    if (cudaGetDevice(&target_device) != cudaSuccess) {
      cudaGetLastError(); // clear last-error state
      return;
    }
  }

  // The prefetch is queued on the call's stream (stream-ordered with the kernel).
#if CUDART_VERSION >= 13000
  cudaMemLocation loc = make_mem_location(target_device);
  if (cudaMemPrefetchAsync(ptr, size, loc, 0, stream) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
  if (cudaMemAdvise(ptr, size, memory_hint, loc) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
#else
  if (cudaMemPrefetchAsync(ptr, size, target_device, stream) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
  if (cudaMemAdvise(ptr, size, memory_hint, target_device) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
#endif
}

// Whether the current device supports stream-ordered allocation
// (cudaMallocAsync / cudaFreeAsync). True for all recent GPUs and drivers.
static bool stream_ordered_alloc_supported()
{
  int device = 0;
  int supported = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device) != cudaSuccess)
  {
    cudaGetLastError(); // clear last-error state
    return false;
  }
  return supported != 0;
}

// Shared implementation of both handle_cuda_input_array overloads.
template <typename T>
static void handle_cuda_input_array_impl(T *input_ptr, CudaDevicePtr<T> &out,
                                         std::size_t size, int device_id,
                                         cudaMemoryAdvise memory_hint,
                                         cudaStream_t stream)
{
  out = CudaDevicePtr<T>(); // release a previous allocation, start empty

  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, input_ptr);

  if (err != cudaSuccess) {
    cudaGetLastError();
  }

  if (err == cudaSuccess && attr.type == cudaMemoryTypeManaged) {
    apply_managed_memory_hints(input_ptr, size, device_id, memory_hint, stream);
  }

  if (err == cudaSuccess &&
      (attr.type == cudaMemoryTypeManaged || attr.type == cudaMemoryTypeDevice)) {
    // device / managed memory: use directly, not owned
    out.ptr = input_ptr;
    return;
  }

  // host memory: temporary device copy, allocated and copied stream-ordered
  T *device_ptr = nullptr;
  const bool stream_ordered = stream_ordered_alloc_supported();
  cudaError_t malloc_err = stream_ordered
                               ? cudaMallocAsync(reinterpret_cast<void **>(&device_ptr), size, stream)
                               : cudaMalloc(reinterpret_cast<void **>(&device_ptr), size);
  if (malloc_err != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(
        std::string(stream_ordered ? "cudaMallocAsync failed: " : "cudaMalloc failed: ") +
        cudaGetErrorString(malloc_err));
  }

  // From here on the buffer is owned by `out` and freed by its destructor,
  // also if the copy below fails.
  out.ptr = device_ptr;
  out.owns = true;
  out.stream_ordered = stream_ordered;
  out.stream = stream;

  cudaError_t memcpy_err = cudaMemcpyAsync(device_ptr, input_ptr, size, cudaMemcpyHostToDevice, stream);
  if (memcpy_err != cudaSuccess) {
    cudaGetLastError(); // clear last-error state before throwing
    out = CudaDevicePtr<T>(); // free the buffer now, leave out empty
    throw std::runtime_error(
        std::string("cudaMemcpyAsync (H2D) failed: ") + cudaGetErrorString(memcpy_err));
  }
}

template <typename T>
void handle_cuda_input_array(const T *input_ptr, CudaDevicePtr<T> &out,
                             std::size_t size, int device_id,
                             cudaMemoryAdvise memory_hint, cudaStream_t stream)
{
  // Inputs are only read by the kernels; the const_cast only adapts the type
  // of the CudaDevicePtr that holds the (possibly borrowed) pointer.
  handle_cuda_input_array_impl(const_cast<T *>(input_ptr), out, size, device_id, memory_hint, stream);
}

template <typename T>
void handle_cuda_input_array(T *input_ptr, CudaDevicePtr<T> &out,
                             std::size_t size, int device_id,
                             cudaMemoryAdvise memory_hint, cudaStream_t stream)
{
  handle_cuda_input_array_impl(input_ptr, out, size, device_id, memory_hint, stream);
}

template void handle_cuda_input_array<double>(const double *, CudaDevicePtr<double> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<double>(double *, CudaDevicePtr<double> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<float>(const float *, CudaDevicePtr<float> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<float>(float *, CudaDevicePtr<float> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);

template void handle_cuda_input_array<int>(const int *, CudaDevicePtr<int> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<int>(int *, CudaDevicePtr<int> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<unsigned int>(const unsigned int *, CudaDevicePtr<unsigned int> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<unsigned int>(unsigned int *, CudaDevicePtr<unsigned int> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<std::size_t>(const std::size_t *, CudaDevicePtr<std::size_t> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<std::size_t>(std::size_t *, CudaDevicePtr<std::size_t> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
// 64-bit signed integers (sort keys of tof_lm_sort_key)
template void handle_cuda_input_array<std::int64_t>(const std::int64_t *, CudaDevicePtr<std::int64_t> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<std::int64_t>(std::int64_t *, CudaDevicePtr<std::int64_t> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);

template void handle_cuda_input_array<char>(const char *, CudaDevicePtr<char> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<char>(char *, CudaDevicePtr<char> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<unsigned char>(const unsigned char *, CudaDevicePtr<unsigned char> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<unsigned char>(unsigned char *, CudaDevicePtr<unsigned char> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);

template void handle_cuda_input_array<bool>(const bool *, CudaDevicePtr<bool> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<bool>(bool *, CudaDevicePtr<bool> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);

template void handle_cuda_input_array<short>(const short *, CudaDevicePtr<short> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<short>(short *, CudaDevicePtr<short> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<unsigned short>(const unsigned short *, CudaDevicePtr<unsigned short> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
template void handle_cuda_input_array<unsigned short>(unsigned short *, CudaDevicePtr<unsigned short> &, std::size_t, int, cudaMemoryAdvise, cudaStream_t);
