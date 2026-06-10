#include "cuda_utils.h"
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
                                       cudaMemoryAdvise memory_hint)
{
  int target_device = device_id;
  if (target_device < 0) {
    // No explicit device requested: prefetch/advise for the current device.
    if (cudaGetDevice(&target_device) != cudaSuccess) {
      cudaGetLastError(); // clear last-error state
      return;
    }
  }

#if CUDART_VERSION >= 13000
  cudaMemLocation loc = make_mem_location(target_device);
  if (cudaMemPrefetchAsync(ptr, size, loc, 0, (cudaStream_t)0) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
  if (cudaMemAdvise(ptr, size, memory_hint, loc) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
#else
  if (cudaMemPrefetchAsync(ptr, size, target_device, (cudaStream_t)0) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
  if (cudaMemAdvise(ptr, size, memory_hint, target_device) != cudaSuccess)
    cudaGetLastError(); // clear last-error state
#endif
}

template <typename T>
void handle_cuda_input_array(const T *input_ptr, T **device_ptr,
                             std::size_t size, bool &free_flag,
                             int device_id, cudaMemoryAdvise memory_hint)
{
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, input_ptr);
  free_flag = false;

  if (err != cudaSuccess) {
    // Expected for plain host pointers on older CUDA versions
    // (cudaErrorInvalidValue). Treat as host memory below, but clear the
    // last-error state so it cannot surface in a later kernel launch check.
    cudaGetLastError();
  }

  if (err == cudaSuccess && attr.type == cudaMemoryTypeManaged) {
    apply_managed_memory_hints(
        const_cast<void *>(static_cast<const void *>(input_ptr)),
        size, device_id, memory_hint);
  }

  if (err == cudaSuccess &&
      (attr.type == cudaMemoryTypeManaged || attr.type == cudaMemoryTypeDevice)) {
    *device_ptr = const_cast<T *>(input_ptr);
  } else {
    cudaError_t malloc_err = cudaMalloc(device_ptr, size);
    if (malloc_err != cudaSuccess) {
      // Clear the runtime's last-error state before throwing: CUDA only
      // resets it via cudaGetLastError(), so a stale error would otherwise
      // be misreported by the kernel launch check of a LATER, successful call.
      cudaGetLastError();
      throw std::runtime_error(
          std::string("cudaMalloc failed: ") + cudaGetErrorString(malloc_err));
    }
    cudaError_t memcpy_err = cudaMemcpy(*device_ptr, input_ptr, size, cudaMemcpyHostToDevice);
    if (memcpy_err != cudaSuccess) {
      cudaFree(*device_ptr);
      *device_ptr = nullptr;
      cudaGetLastError(); // clear last-error state before throwing (see above)
      throw std::runtime_error(
          std::string("cudaMemcpy (H2D) failed: ") + cudaGetErrorString(memcpy_err));
    }
    free_flag = true;
  }
}

template <typename T>
void handle_cuda_input_array(T *input_ptr, T **device_ptr,
                             std::size_t size, bool &free_flag,
                             int device_id, cudaMemoryAdvise memory_hint)
{
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, input_ptr);
  free_flag = false;

  if (err != cudaSuccess) {
    // Expected for plain host pointers on older CUDA versions
    // (cudaErrorInvalidValue). Treat as host memory below, but clear the
    // last-error state so it cannot surface in a later kernel launch check.
    cudaGetLastError();
  }

  if (err == cudaSuccess && attr.type == cudaMemoryTypeManaged) {
    apply_managed_memory_hints(input_ptr, size, device_id, memory_hint);
  }

  if (err == cudaSuccess &&
      (attr.type == cudaMemoryTypeManaged || attr.type == cudaMemoryTypeDevice)) {
    *device_ptr = input_ptr;
  } else {
    cudaError_t malloc_err = cudaMalloc(device_ptr, size);
    if (malloc_err != cudaSuccess) {
      // Clear the runtime's last-error state before throwing: CUDA only
      // resets it via cudaGetLastError(), so a stale error would otherwise
      // be misreported by the kernel launch check of a LATER, successful call.
      cudaGetLastError();
      throw std::runtime_error(
          std::string("cudaMalloc failed: ") + cudaGetErrorString(malloc_err));
    }
    cudaError_t memcpy_err = cudaMemcpy(*device_ptr, input_ptr, size, cudaMemcpyHostToDevice);
    if (memcpy_err != cudaSuccess) {
      cudaFree(*device_ptr);
      *device_ptr = nullptr;
      cudaGetLastError(); // clear last-error state before throwing (see above)
      throw std::runtime_error(
          std::string("cudaMemcpy (H2D) failed: ") + cudaGetErrorString(memcpy_err));
    }
    free_flag = true;
  }
}


// Explicit template instantiations
template void handle_cuda_input_array<double>(const double *, double **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<double>(double *, double **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<float>(const float *, float **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<float>(float *, float **, std::size_t, bool &, int, cudaMemoryAdvise);

template void handle_cuda_input_array<int>(const int *, int **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<int>(int *, int **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned int>(const unsigned int *, unsigned int **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned int>(unsigned int *, unsigned int **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<std::size_t>(const std::size_t *, std::size_t **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<std::size_t>(std::size_t *, std::size_t **, std::size_t, bool &, int, cudaMemoryAdvise);

template void handle_cuda_input_array<char>(const char *, char **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<char>(char *, char **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned char>(const unsigned char *, unsigned char **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned char>(unsigned char *, unsigned char **, std::size_t, bool &, int, cudaMemoryAdvise);

template void handle_cuda_input_array<bool>(const bool *, bool **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<bool>(bool *, bool **, std::size_t, bool &, int, cudaMemoryAdvise);

template void handle_cuda_input_array<short>(const short *, short **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<short>(short *, short **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned short>(const unsigned short *, unsigned short **, std::size_t, bool &, int, cudaMemoryAdvise);
template void handle_cuda_input_array<unsigned short>(unsigned short *, unsigned short **, std::size_t, bool &, int, cudaMemoryAdvise);
