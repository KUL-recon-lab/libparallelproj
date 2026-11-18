#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include "parallelproj.h"
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

// ND c-contiguous float arrays
using ConstFloatNDArray = nb::ndarray<const float, nb::c_contig>;
using ConstShortNDArray = nb::ndarray<const short, nb::c_contig>;
using FloatNDArray = nb::ndarray<float, nb::c_contig>;
// 3D c-contiguous float arrays
using ConstFloat3DArray = nb::ndarray<const float, nb::c_contig, nb::ndim<3>>;
using Float3DArray = nb::ndarray<float, nb::c_contig, nb::ndim<3>>;
// 1D c-contiguous float arrays
using ConstFloat1D3ELArray = nb::ndarray<const float, nb::c_contig, nb::shape<3>>;

////////////////////////////////////////////////////////////////////////////////
// Device checks shared by all projector bindings
////////////////////////////////////////////////////////////////////////////////

namespace
{
  struct ArrayDevice
  {
    int32_t type;
    int32_t id;
    std::string name;
  };

  template <typename Array>
  ArrayDevice array_device(const Array &a, const char *name)
  {
    return {a.device_type(), a.device_id(), name};
  }

  bool is_host_device(int32_t type)
  {
    return type == nb::device::cpu::value || type == nb::device::cuda_host::value;
  }

  bool is_cuda_device(int32_t type)
  {
    return type == nb::device::cuda::value || type == nb::device::cuda_managed::value;
  }

  // Checks the devices of all array arguments and returns the CUDA device ID to use.
  //
  // - main: the arrays that define where the computation runs (LOR / event coordinates,
  //   image, projection values, ...). They must all be on the same device (type and ID).
  // - params: small parameter arrays (e.g. tof_sigma, tof_center_offset). They may be
  //   CPU arrays (copied to the GPU in CUDA builds) or on the same device as the main arrays.
  // - device_id: requested CUDA device (-1: automatic).
  //
  // CPU-only builds: all arrays must be CPU arrays.
  // CUDA builds: if the main arrays are on a GPU, the computation runs on that GPU and an
  // explicitly requested different device_id is an error. For host arrays, device_id is
  // passed on unchanged (-1: current CUDA device).
  //
  // check_devices_impl() takes the build type as an argument, so that the logic of
  // both build types can be tested in any build (exposed as _check_devices).
  int check_devices_impl(const std::vector<ArrayDevice> &main,
                         const std::vector<ArrayDevice> &params,
                         int device_id,
                         bool cuda_build)
  {
    if (main.empty())
      throw std::invalid_argument("check_devices: no main arrays");

    const ArrayDevice &ref = main.front();

    for (const ArrayDevice &a : main)
    {
      if (a.type != ref.type)
        throw std::invalid_argument("All input arrays must be on the same device type (" +
                                    ref.name + " and " + a.name + " differ)");
      if (a.id != ref.id)
        throw std::invalid_argument("All input arrays must be on the same device ID (" +
                                    ref.name + " and " + a.name + " differ)");
    }

    if (!cuda_build)
    {
      for (const std::vector<ArrayDevice> *group : {&main, &params})
      {
        for (const ArrayDevice &a : *group)
        {
          if (!is_host_device(a.type))
            throw std::invalid_argument(a.name +
                                        " is not a CPU array, but parallelproj_core was built without CUDA support");
        }
      }
      return device_id;
    }

    if (!is_host_device(ref.type) && !is_cuda_device(ref.type))
      throw std::invalid_argument("unsupported device type of " + ref.name +
                                  " (only CPU and CUDA arrays are supported)");

    for (const ArrayDevice &p : params)
    {
      if (is_host_device(p.type))
        continue;
      if (!is_cuda_device(p.type) || !is_cuda_device(ref.type) || p.id != ref.id)
        throw std::invalid_argument(p.name +
                                    " must be a CPU array or on the same device as the other input arrays");
    }

    if (is_cuda_device(ref.type))
    {
      if (device_id >= 0 && device_id != ref.id)
        throw std::invalid_argument("device_id = " + std::to_string(device_id) +
                                    " does not match the CUDA device of the input arrays (" +
                                    std::to_string(ref.id) + ")");
      return ref.id;
    }

    return device_id;
  }

  int check_devices(std::initializer_list<ArrayDevice> main,
                    std::initializer_list<ArrayDevice> params,
                    int device_id)
  {
    return check_devices_impl(std::vector<ArrayDevice>(main), std::vector<ArrayDevice>(params),
                              device_id, parallelproj_cuda_enabled() != 0);
  }

  // Stream handle (Python int, as returned by torch.cuda.Stream.cuda_stream or
  // cupy.cuda.Stream.ptr) -> stream argument of the C API.
  //   None -> nullptr: legacy default stream, synchronous call
  //   0    -> cudaStreamLegacy: legacy default stream, asynchronous call
  //           (PyTorch and CuPy report their default stream as handle 0)
  //   else -> that stream, asynchronous call
  // The bindings are built without CUDA headers, so cudaStreamLegacy is spelled
  // out here; its value ((cudaStream_t)0x1) is fixed by the CUDA runtime API.
  constexpr std::uintptr_t cuda_stream_legacy_handle = 0x1;

  void *stream_pointer(const std::optional<std::uintptr_t> &stream)
  {
    if (!stream.has_value())
      return nullptr;
    if (*stream == 0)
      return reinterpret_cast<void *>(cuda_stream_legacy_handle);
    return reinterpret_cast<void *>(*stream);
  }
} // namespace

// Wrapper for joseph3d_fwd
void joseph3d_fwd_py(ConstFloatNDArray lor_start,
                     ConstFloatNDArray lor_end,
                     ConstFloat3DArray image,
                     ConstFloat1D3ELArray image_origin,
                     ConstFloat1D3ELArray voxel_size,
                     FloatNDArray projection_values,
                     int device_id = -1,
                     int threads_per_block = 64,
                     std::optional<std::uintptr_t> stream = std::nullopt)
{
  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check that the shape of projection_values matches lor_start.shape[:-1]
  if (projection_values.ndim() != lor_start.ndim() - 1)
    throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  for (std::size_t i = 0; i < projection_values.ndim(); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  }

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(lor_start, "lor_start"), array_device(lor_end, "lor_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values")},
                            {},
                            device_id);
  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    joseph3d_fwd(lor_start.data(), lor_end.data(), image.data(), image_origin.data(), voxel_size.data(), projection_values.data(), num_lors, image_dim, device_id, threads_per_block,
                 stream_pointer(stream));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_back
void joseph3d_back_py(ConstFloatNDArray lor_start,
                      ConstFloatNDArray lor_end,
                      Float3DArray image,
                      ConstFloat1D3ELArray image_origin,
                      ConstFloat1D3ELArray voxel_size,
                      ConstFloatNDArray projection_values,
                      int device_id = -1,
                      int threads_per_block = 64,
                      std::optional<std::uintptr_t> stream = std::nullopt)
{
  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check that the shape of projection_values matches lor_start.shape[:-1]
  if (projection_values.ndim() != lor_start.ndim() - 1)
    throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  for (std::size_t i = 0; i < projection_values.ndim(); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("projection_values must have a shape equal to lor_start.shape[:-1]");
  }

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(lor_start, "lor_start"), array_device(lor_end, "lor_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values")},
                            {},
                            device_id);
  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    joseph3d_back(lor_start.data(), lor_end.data(), image.data(), image_origin.data(), voxel_size.data(), projection_values.data(), num_lors, image_dim, device_id, threads_per_block,
                 stream_pointer(stream));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_fwd
void joseph3d_tof_sino_fwd_py(ConstFloatNDArray lor_start,
                              ConstFloatNDArray lor_end,
                              ConstFloat3DArray image,
                              ConstFloat1D3ELArray image_origin,
                              ConstFloat1D3ELArray voxel_size,
                              FloatNDArray projection_values,
                              float tof_bin_width,
                              ConstFloatNDArray tof_sigma,
                              ConstFloatNDArray tof_center_offset,
                              short num_tof_bins,
                              float num_sigmas = 3.0f,
                              int device_id = -1,
                              int threads_per_block = 64,
                              std::optional<std::uintptr_t> stream = std::nullopt)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check projection_values has same ndim as lor_start
  if (projection_values.ndim() != lor_start.ndim())
    throw std::invalid_argument("projection_values must have same number of dimensions as lor_start");
  for (std::size_t i = 0; i < (projection_values.ndim() - 1); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("shape of projection_values[:-1] must match shape of lor_start[:-1]");
  }
  // check that projection_values.shape[-1] == num_tof_bins
  if (projection_values.shape(projection_values.ndim() - 1) != static_cast<std::size_t>(num_tof_bins))
    throw std::invalid_argument("last dimension of projection_values must equal num_tof_bins");

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(lor_start, "lor_start"), array_device(lor_end, "lor_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values")},
                            {array_device(tof_sigma, "tof_sigma"), array_device(tof_center_offset, "tof_center_offset")},
                            device_id);
  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or lor_start.shape[:-1]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_sigma.ndim()); ++i)
    {
      if (tof_sigma.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
  }

  // check that the shape of tof_center_offset is either [1,] or lor_start.shape[:-1]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_center_offset.ndim()); ++i)
    {
      if (tof_center_offset.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
  }

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    joseph3d_tof_sino_fwd(lor_start.data(),
                            lor_end.data(),
                            image.data(),
                            image_origin.data(),
                            voxel_size.data(),
                            projection_values.data(),
                            num_lors,
                            image_dim,
                            tof_bin_width,
                            tof_sigma.data(),
                            tof_center_offset.data(),
                            num_sigmas,
                            num_tof_bins,
                            static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                            static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                            device_id,
                            threads_per_block,
                            stream_pointer(stream));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_back
void joseph3d_tof_back_fwd_py(ConstFloatNDArray lor_start,
                              ConstFloatNDArray lor_end,
                              Float3DArray image,
                              ConstFloat1D3ELArray image_origin,
                              ConstFloat1D3ELArray voxel_size,
                              ConstFloatNDArray projection_values,
                              float tof_bin_width,
                              ConstFloatNDArray tof_sigma,
                              ConstFloatNDArray tof_center_offset,
                              short num_tof_bins,
                              float num_sigmas = 3.0f,
                              int device_id = -1,
                              int threads_per_block = 64,
                              std::optional<std::uintptr_t> stream = std::nullopt)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of lor_start and lor_end >=2 and last dim ==3
  if (lor_start.ndim() < 2 || lor_start.shape(lor_start.ndim() - 1) != 3)
    throw std::invalid_argument("lor_start must have at least 2 dims and shape (..., 3)");

  // 2 check that lor_start and lor_end have same ndim and shape
  if (lor_start.ndim() != lor_end.ndim())
    throw std::invalid_argument("lor_start and lor_end must have the same number of dimensions");
  for (std::size_t i = 0; i < lor_start.ndim(); ++i)
  {
    if (lor_start.shape(i) != lor_end.shape(i))
      throw std::invalid_argument("lor_start and lor_end must have the same shape");
  }

  // 3 check projection_values has same ndim as lor_start
  if (projection_values.ndim() != lor_start.ndim())
    throw std::invalid_argument("projection_values must have same number of dimensions as lor_start");
  for (std::size_t i = 0; i < (projection_values.ndim() - 1); ++i)
  {
    if (projection_values.shape(i) != lor_start.shape(i))
      throw std::invalid_argument("shape of projection_values[:-1] must match shape of lor_start[:-1]");
  }
  // check that projection_values.shape[-1] == num_tof_bins
  if (projection_values.shape(projection_values.ndim() - 1) != static_cast<std::size_t>(num_tof_bins))
    throw std::invalid_argument("last dimension of projection_values must equal num_tof_bins");

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(lor_start, "lor_start"), array_device(lor_end, "lor_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values")},
                            {array_device(tof_sigma, "tof_sigma"), array_device(tof_center_offset, "tof_center_offset")},
                            device_id);
  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  std::size_t num_lors = 1;
  for (std::size_t i = 0; i < lor_start.ndim() - 1; ++i)
  {
    num_lors *= lor_start.shape(i);
  }

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or lor_start.shape[:-1]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_sigma.ndim()); ++i)
    {
      if (tof_sigma.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must match shape of lor_start[:-1] or be scalar");
  }

  // check that the shape of tof_center_offset is either [1,] or lor_start.shape[:-1]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == lor_start.ndim() - 1)
  {
    for (std::size_t i = 0; i < (tof_center_offset.ndim()); ++i)
    {
      if (tof_center_offset.shape(i) != lor_start.shape(i))
        throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
    }
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must match shape of lor_start[:-1] or be scalar");
  }

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    joseph3d_tof_sino_back(lor_start.data(),
                             lor_end.data(),
                             image.data(),
                             image_origin.data(),
                             voxel_size.data(),
                             projection_values.data(),
                             num_lors,
                             image_dim,
                             tof_bin_width,
                             tof_sigma.data(),
                             tof_center_offset.data(),
                             num_sigmas,
                             num_tof_bins,
                             static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                             static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                             device_id,
                             threads_per_block,
                             stream_pointer(stream));
  }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_lm_fwd
void joseph3d_tof_lm_fwd_py(ConstFloatNDArray event_start,
                            ConstFloatNDArray event_end,
                            ConstFloat3DArray image,
                            ConstFloat1D3ELArray image_origin,
                            ConstFloat1D3ELArray voxel_size,
                            FloatNDArray projection_values,
                            float tof_bin_width,
                            ConstFloatNDArray tof_sigma,
                            ConstFloatNDArray tof_center_offset,
                            ConstShortNDArray tof_bin_index,
                            short num_tof_bins,
                            float num_sigmas = 3.0f,
                            int device_id = -1,
                            int threads_per_block = 64,
                            std::optional<std::uintptr_t> stream = std::nullopt)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of event_start and event_end are equal to 2
  if (event_start.ndim() != 2 || event_end.ndim() != 2)
    throw std::invalid_argument("event_start and event_end must have 2 dimensions");
  if (event_start.shape(1) != 3 || event_end.shape(1) != 3)
    throw std::invalid_argument("event_start and event_end must have shape (..., 3)");
  if (event_start.shape(0) != event_end.shape(0))
    throw std::invalid_argument("event_start and event_end must have the same number of events (shape[0])");

  std::size_t num_events = event_start.shape(0);

  // 2 check dims and shapes of projection_values and tof_bin_index
  if (projection_values.ndim() != 1)
    throw std::invalid_argument("projection_values.ndim must be 1");
  if (projection_values.shape(0) != num_events)
    throw std::invalid_argument("projection_values.shape[0] must match event_start.shape[0]");

  if (tof_bin_index.ndim() != 1)
    throw std::invalid_argument("tof_bin_index.ndim must be 1");
  if (tof_bin_index.shape(0) != num_events)
    throw std::invalid_argument("tof_bin_index.shape[0] must match event_start.shape[0]");

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(event_start, "event_start"), array_device(event_end, "event_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values"), array_device(tof_bin_index, "tof_bin_index")},
                            {array_device(tof_sigma, "tof_sigma"), array_device(tof_center_offset, "tof_center_offset")},
                            device_id);
  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or [num_events,]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == num_events)
  {
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must be [1,] or [num_events,]");
  }

  // check that the shape of tof_center_offset is either [1,] or [num_events,]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == num_events)
  {
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must be [1,] or [num_events,]");
  }

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    joseph3d_tof_lm_fwd(event_start.data(),
                          event_end.data(),
                          image.data(),
                          image_origin.data(),
                          voxel_size.data(),
                          projection_values.data(),
                          num_events,
                          image_dim,
                          tof_bin_width,
                          tof_sigma.data(),
                          tof_center_offset.data(),
                          num_sigmas,
                          tof_bin_index.data(),
                          num_tof_bins,
                          static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                          static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                          device_id,
                          threads_per_block,
                          stream_pointer(stream));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_lm_back
void joseph3d_tof_lm_back_py(ConstFloatNDArray event_start,
                             ConstFloatNDArray event_end,
                             Float3DArray image,
                             ConstFloat1D3ELArray image_origin,
                             ConstFloat1D3ELArray voxel_size,
                             ConstFloatNDArray projection_values,
                             float tof_bin_width,
                             ConstFloatNDArray tof_sigma,
                             ConstFloatNDArray tof_center_offset,
                             ConstShortNDArray tof_bin_index,
                             short num_tof_bins,
                             float num_sigmas = 3.0f,
                             int device_id = -1,
                             int threads_per_block = 64,
                             std::optional<std::uintptr_t> stream = std::nullopt)
{
  bool is_lor_dependent_tof_sigma;
  bool is_lor_dependent_tof_center_offset;

  // 1 check that ndim of event_start and event_end are equal to 2
  if (event_start.ndim() != 2 || event_end.ndim() != 2)
    throw std::invalid_argument("event_start and event_end must have 2 dimensions");
  if (event_start.shape(1) != 3 || event_end.shape(1) != 3)
    throw std::invalid_argument("event_start and event_end must have shape (..., 3)");
  if (event_start.shape(0) != event_end.shape(0))
    throw std::invalid_argument("event_start and event_end must have the same number of events (shape[0])");

  std::size_t num_events = event_start.shape(0);

  // 2 check dims and shapes of projection_values and tof_bin_index
  if (projection_values.ndim() != 1)
    throw std::invalid_argument("projection_values.ndim must be 1");
  if (projection_values.shape(0) != num_events)
    throw std::invalid_argument("projection_values.shape[0] must match event_start.shape[0]");

  if (tof_bin_index.ndim() != 1)
    throw std::invalid_argument("tof_bin_index.ndim must be 1");
  if (tof_bin_index.shape(0) != num_events)
    throw std::invalid_argument("tof_bin_index.shape[0] must match event_start.shape[0]");

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(event_start, "event_start"), array_device(event_end, "event_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values"), array_device(tof_bin_index, "tof_bin_index")},
                            {array_device(tof_sigma, "tof_sigma"), array_device(tof_center_offset, "tof_center_offset")},
                            device_id);
  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check that the shape of tof_sigma is either [1,] or [num_events,]
  if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == 1)
  {
    is_lor_dependent_tof_sigma = false;
  }
  else if (tof_sigma.ndim() == 1 && tof_sigma.shape(0) == num_events)
  {
    is_lor_dependent_tof_sigma = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_sigma must be [1,] or [num_events,]");
  }

  // check that the shape of tof_center_offset is either [1,] or [num_events,]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
  {
    is_lor_dependent_tof_center_offset = false;
  }
  else if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == num_events)
  {
    is_lor_dependent_tof_center_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tof_center_offset must be [1,] or [num_events,]");
  }

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    joseph3d_tof_lm_back(event_start.data(),
                           event_end.data(),
                           image.data(),
                           image_origin.data(),
                           voxel_size.data(),
                           projection_values.data(),
                           num_events,
                           image_dim,
                           tof_bin_width,
                           tof_sigma.data(),
                           tof_center_offset.data(),
                           num_sigmas,
                           tof_bin_index.data(),
                           num_tof_bins,
                           static_cast<unsigned char>(is_lor_dependent_tof_sigma ? 1 : 0),
                           static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                           device_id,
                           threads_per_block,
                           stream_pointer(stream));
  }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Shared input validation for tof_lm_point_fwd / tof_lm_point_back.
// Returns whether the TOF center offset is event dependent.
template <typename ImageArray, typename ValueArray>
static bool check_tof_lm_point_args(const ConstFloatNDArray &event_start,
                                    const ConstFloatNDArray &event_end,
                                    const ImageArray &image,
                                    const ConstFloat1D3ELArray &image_origin,
                                    const ConstFloat1D3ELArray &voxel_size,
                                    const ValueArray &projection_values,
                                    float tof_bin_width,
                                    const ConstFloatNDArray &tof_center_offset,
                                    const ConstShortNDArray &tof_bin_index,
                                    short num_tof_bins)
{
  if (event_start.ndim() != 2 || event_end.ndim() != 2)
    throw std::invalid_argument("event_start and event_end must have 2 dimensions");
  if (event_start.shape(1) != 3 || event_end.shape(1) != 3)
    throw std::invalid_argument("event_start and event_end must have shape (..., 3)");
  if (event_start.shape(0) != event_end.shape(0))
    throw std::invalid_argument("event_start and event_end must have the same number of events (shape[0])");

  std::size_t num_events = event_start.shape(0);

  if (projection_values.ndim() != 1)
    throw std::invalid_argument("projection_values.ndim must be 1");
  if (projection_values.shape(0) != num_events)
    throw std::invalid_argument("projection_values.shape[0] must match event_start.shape[0]");

  if (tof_bin_index.ndim() != 1)
    throw std::invalid_argument("tof_bin_index.ndim must be 1");
  if (tof_bin_index.shape(0) != num_events)
    throw std::invalid_argument("tof_bin_index.shape[0] must match event_start.shape[0]");

  if (!(tof_bin_width > 0.0f) || !std::isfinite(tof_bin_width))
    throw std::invalid_argument("tof_bin_width must be finite and > 0");
  if (num_tof_bins < 1)
    throw std::invalid_argument("num_tof_bins must be >= 1");

  // the shape of tof_center_offset must be either [1,] or [num_events,]
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == 1)
    return false;
  if (tof_center_offset.ndim() == 1 && tof_center_offset.shape(0) == num_events)
    return true;
  throw std::invalid_argument("shape of tof_center_offset must be [1,] or [num_events,]");
}

// Wrapper for tof_lm_point_fwd
void tof_lm_point_fwd_py(ConstFloatNDArray event_start,
                         ConstFloatNDArray event_end,
                         ConstFloat3DArray image,
                         ConstFloat1D3ELArray image_origin,
                         ConstFloat1D3ELArray voxel_size,
                         FloatNDArray projection_values,
                         float tof_bin_width,
                         ConstFloatNDArray tof_center_offset,
                         ConstShortNDArray tof_bin_index,
                         short num_tof_bins,
                         int device_id = -1,
                         int threads_per_block = 64,
                         std::optional<std::uintptr_t> stream = std::nullopt)
{
  bool is_lor_dependent_tof_center_offset = check_tof_lm_point_args(
      event_start, event_end, image, image_origin, voxel_size, projection_values,
      tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins);

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(event_start, "event_start"), array_device(event_end, "event_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values"), array_device(tof_bin_index, "tof_bin_index")},
                            {array_device(tof_center_offset, "tof_center_offset")},
                            device_id);

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    tof_lm_point_fwd(event_start.data(),
                       event_end.data(),
                       image.data(),
                       image_origin.data(),
                       voxel_size.data(),
                       projection_values.data(),
                       event_start.shape(0),
                       image_dim,
                       tof_bin_width,
                       tof_center_offset.data(),
                       tof_bin_index.data(),
                       num_tof_bins,
                       static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                       device_id,
                       threads_per_block,
                       stream_pointer(stream));
  }
}

// Wrapper for tof_lm_point_back
void tof_lm_point_back_py(ConstFloatNDArray event_start,
                          ConstFloatNDArray event_end,
                          Float3DArray image,
                          ConstFloat1D3ELArray image_origin,
                          ConstFloat1D3ELArray voxel_size,
                          ConstFloatNDArray projection_values,
                          float tof_bin_width,
                          ConstFloatNDArray tof_center_offset,
                          ConstShortNDArray tof_bin_index,
                          short num_tof_bins,
                          int device_id = -1,
                          int threads_per_block = 64,
                          std::optional<std::uintptr_t> stream = std::nullopt)
{
  bool is_lor_dependent_tof_center_offset = check_tof_lm_point_args(
      event_start, event_end, image, image_origin, voxel_size, projection_values,
      tof_bin_width, tof_center_offset, tof_bin_index, num_tof_bins);

  int image_dim[3] = {static_cast<int>(image.shape(0)),
                      static_cast<int>(image.shape(1)),
                      static_cast<int>(image.shape(2))};

  // check the devices of all arrays and get the CUDA device to use
  device_id = check_devices({array_device(event_start, "event_start"), array_device(event_end, "event_end"), array_device(image, "image"), array_device(image_origin, "image_origin"), array_device(voxel_size, "voxel_size"), array_device(projection_values, "projection_values"), array_device(tof_bin_index, "tof_bin_index")},
                            {array_device(tof_center_offset, "tof_center_offset")},
                            device_id);

  // release the GIL while the projector runs (other Python threads, e.g.
  // one thread per GPU, can run in the meantime)
  {
    nb::gil_scoped_release release;
    tof_lm_point_back(event_start.data(),
                        event_end.data(),
                        image.data(),
                        image_origin.data(),
                        voxel_size.data(),
                        projection_values.data(),
                        event_start.shape(0),
                        image_dim,
                        tof_bin_width,
                        tof_center_offset.data(),
                        tof_bin_index.data(),
                        num_tof_bins,
                        static_cast<unsigned char>(is_lor_dependent_tof_center_offset ? 1 : 0),
                        device_id,
                        threads_per_block,
                        stream_pointer(stream));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

NB_MODULE(parallelproj_core, m)
{
  m.doc() = R"pbdoc(
    Python bindings for parallelproj backend.

    This module provides efficient C++/CUDA implementations of 3D Joseph
    forward and backward projectors with optional Time-of-Flight (TOF) support.

    Attributes
    ----------
    __version__ : str
        The version of the parallelproj backend library.
    version_major : int
      Major version number of the linked parallelproj library.
    version_minor : int
      Minor version number of the linked parallelproj library.
    version_patch : int
      Patch version number of the linked parallelproj library.
    cuda_enabled : int
        Flag indicating if the library was compiled with CUDA support.
        Returns 1 if CUDA is enabled, 0 otherwise.
  )pbdoc";

  // Expose the linked library version as __version__
  m.attr("__version__") = parallelproj_version();
  m.attr("version_major") = parallelproj_version_major();
  m.attr("version_minor") = parallelproj_version_minor();
  m.attr("version_patch") = parallelproj_version_patch();

  m.attr("cuda_enabled") = parallelproj_cuda_enabled();

  // internal: the stream pointer passed to the C API for a given stream argument
  // (exposed for testing)
  m.def(
      "_stream_pointer",
      [](std::optional<std::uintptr_t> stream)
      { return reinterpret_cast<std::uintptr_t>(stream_pointer(stream)); },
      "stream"_a = nb::none(),
      "Internal: stream pointer passed to the C API for a given stream argument (for testing).");

  // internal: the device check logic of the projector bindings, exposed for testing
  // (arrays given as (name, DLPack device type, device id) tuples)
  m.def(
      "_check_devices",
      [](const std::vector<std::tuple<std::string, int, int>> &main,
         const std::vector<std::tuple<std::string, int, int>> &params,
         int device_id, bool cuda_build)
      {
        auto convert = [](const std::vector<std::tuple<std::string, int, int>> &in)
        {
          std::vector<ArrayDevice> out;
          for (const auto &t : in)
            out.push_back({std::get<1>(t), std::get<2>(t), std::get<0>(t)});
          return out;
        };
        return check_devices_impl(convert(main), convert(params), device_id, cuda_build);
      },
      "main"_a, "params"_a, "device_id"_a, "cuda_build"_a,
      "Internal: device check logic of the projector bindings (for testing).");

  m.def("joseph3d_fwd", &joseph3d_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        Forward projection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) for forward projection.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Output array for projection results, shape must match lor_start.shape[:-1].
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.

        Notes
        -----
        All arrays must be on the same device (CPU or GPU) with matching device IDs.
        )pbdoc");

  m.def("joseph3d_back", &joseph3d_back_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        Backprojection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate backprojection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Input values to backproject, shape must match lor_start.shape[:-1].
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        All arrays must be on the same device (CPU or GPU) with matching device IDs.
        )pbdoc");

  m.def("joseph3d_tof_sino_fwd", &joseph3d_tof_sino_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        TOF sinogram forward projection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) for forward projection.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Output array for TOF sinogram, shape (..., num_tof_bins).
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (...,) for LOR-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (...,).
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.
        )pbdoc");

  m.def("joseph3d_tof_sino_back", &joseph3d_tof_back_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        TOF sinogram backprojection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate backprojection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Input TOF sinogram to backproject, shape (..., num_tof_bins).
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (...,) for LOR-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (...,).
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        )pbdoc");

  m.def("joseph3d_tof_lm_fwd", &joseph3d_tof_lm_fwd_py,
        "event_start"_a.noconvert(), "event_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        TOF listmode forward projection using the Joseph 3D algorithm.

        Parameters
        ----------
        event_start : ndarray
          Array of shape (..., 3) with coordinates of event LOR start points.
        event_end : ndarray
          Array of shape (..., 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) for forward projection.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Output array for event projections, shape matches event_start.shape[:-1].
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (...,) for event-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (...,).
        tof_bin_index : ndarray
          TOF bin indices for each event, shape matches event_start.shape[:-1].
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.
        )pbdoc");

  m.def("joseph3d_tof_lm_back", &joseph3d_tof_lm_back_py,
        "event_start"_a.noconvert(), "event_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        TOF listmode backprojection using the Joseph 3D algorithm.

        Parameters
        ----------
        event_start : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR start points.
        event_end : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate backprojection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
            Input array to be backprojected, shape (num_events,).
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (num_events,) for event-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (num_events,).
        tof_bin_index : ndarray
            TOF bin indices for each event, shape (num_events,).
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        )pbdoc");

  m.def("tof_lm_point_fwd", &tof_lm_point_fwd_py,
        "event_start"_a.noconvert(), "event_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        TOF listmode "point" forward projection.

        Fast, approximate TOF listmode projector: every event is reduced to
        a single point, the center of its TOF bin, and the image is
        trilinearly interpolated there (zero outside of the image).

        TOF bin ``i`` of an event is centered at a distance
        ``(i - (num_tof_bins - 1) / 2) * tof_bin_width + tof_center_offset``
        from the LOR midpoint, measured along the LOR towards ``event_end``
        (same bin placement as :func:`joseph3d_tof_lm_fwd`).

        Parameters
        ----------
        event_start : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR start points.
        event_end : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to be projected.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
            Output array of shape (num_events,). Overwritten with the projections.
        tof_bin_width : float
            Width of TOF bins in spatial units (finite, > 0).
        tof_center_offset : ndarray
            Offset of the central TOF bin from the LOR midpoint towards the LOR
            end point, in spatial units. Shape (1,) or (num_events,).
        tof_bin_index : ndarray
            TOF bin indices for each event (int16), shape (num_events,).
        num_tof_bins : int
            Number of TOF bins.
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.
        )pbdoc");

  m.def("tof_lm_point_back", &tof_lm_point_back_py,
        "event_start"_a.noconvert(), "event_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "device_id"_a = -1, "threads_per_block"_a = 64, "stream"_a = nb::none(),
        R"pbdoc(
        TOF listmode "point" back projection (adjoint of :func:`tof_lm_point_fwd`).

        Adds the value of every event, distributed with trilinear
        interpolation weights, to the (up to 8) voxels around the center of
        its TOF bin. Back projecting events with value 1 creates a TOF
        "histo-image".

        TOF bin ``i`` of an event is centered at a distance
        ``(i - (num_tof_bins - 1) / 2) * tof_bin_width + tof_center_offset``
        from the LOR midpoint, measured along the LOR towards ``event_end``
        (same bin placement as :func:`joseph3d_tof_lm_fwd`).

        Parameters
        ----------
        event_start : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR start points.
        event_end : ndarray
          Array of shape (num_events, 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate the back projection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
            Event values to be back projected, shape (num_events,).
        tof_bin_width : float
            Width of TOF bins in spatial units (finite, > 0).
        tof_center_offset : ndarray
            Offset of the central TOF bin from the LOR midpoint towards the LOR
            end point, in spatial units. Shape (1,) or (num_events,).
        tof_bin_index : ndarray
            TOF bin indices for each event (int16), shape (num_events,).
        num_tof_bins : int
            Number of TOF bins.
        device_id : int, optional
            CUDA device to use (CUDA builds only). Default -1 (automatic): the GPU of the
            input arrays for GPU arrays, the current CUDA device for CPU arrays (which are
            copied to the GPU). Must match the GPU of the input arrays if given for GPU arrays.
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        stream : int or None, optional
            CUDA stream handle for this call (CUDA builds only, ignored otherwise), e.g.
            ``torch.cuda.current_stream(dev).cuda_stream`` or
            ``cupy.cuda.get_current_stream().ptr``. ``None`` (default): the call is
            synchronous, i.e. results are complete on return (it waits for its own
            stream, not for the whole GPU). Otherwise the call is queued on that
            stream and, for GPU arrays, returns immediately, like a PyTorch / CuPy
            operation on that stream; ``0`` is the default stream of PyTorch / CuPy.
            Calls with CPU arrays always return complete results.

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        )pbdoc");
}
