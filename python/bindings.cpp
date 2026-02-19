#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "parallelproj.h"

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

// Wrapper for joseph3d_fwd
void joseph3d_fwd_py(ConstFloatNDArray xstart,
                     ConstFloatNDArray xend,
                     ConstFloat3DArray img,
                     ConstFloat1D3ELArray img_origin,
                     ConstFloat1D3ELArray voxsize,
                     FloatNDArray p,
                     int device_id = 0,
                     int threadsperblock = 64)
{
  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check that the shape of p matches xstart.shape[:-1]
  if (p.ndim() != xstart.ndim() - 1)
    throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  for (size_t i = 0; i < p.ndim(); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  }

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  joseph3d_fwd(xstart.data(), xend.data(), img.data(), img_origin.data(), voxsize.data(), p.data(), nlors, img_dim, device_id, threadsperblock);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_back
void joseph3d_back_py(ConstFloatNDArray xstart,
                      ConstFloatNDArray xend,
                      Float3DArray img,
                      ConstFloat1D3ELArray img_origin,
                      ConstFloat1D3ELArray voxsize,
                      ConstFloatNDArray p,
                      int device_id = 0,
                      int threadsperblock = 64)
{
  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check that the shape of p matches xstart.shape[:-1]
  if (p.ndim() != xstart.ndim() - 1)
    throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  for (size_t i = 0; i < p.ndim(); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  }

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  joseph3d_back(xstart.data(), xend.data(), img.data(), img_origin.data(), voxsize.data(), p.data(), nlors, img_dim, device_id, threadsperblock);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_fwd
void joseph3d_tof_sino_fwd_py(ConstFloatNDArray xstart,
                              ConstFloatNDArray xend,
                              ConstFloat3DArray img,
                              ConstFloat1D3ELArray img_origin,
                              ConstFloat1D3ELArray voxsize,
                              FloatNDArray p,
                              float tofbin_width,
                              ConstFloatNDArray sigma_tof,
                              ConstFloatNDArray tofcenter_offset,
                              short n_tofbins,
                              float n_sigmas = 3.0f,
                              int device_id = 0,
                              int threadsperblock = 64)
{
  bool lor_dependent_sigma_tof;
  bool lor_dependent_tofcenter_offset;

  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check p has same ndim as xstart
  if (p.ndim() != xstart.ndim())
    throw std::invalid_argument("p must have same number of dimensions as xstart");
  for (size_t i = 0; i < (p.ndim() - 1); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("shape of p[:-1] must match shape of xstart[:-1]");
  }
  // check that p.shape[-1] == n_tofbins
  if (p.shape(p.ndim() - 1) != static_cast<size_t>(n_tofbins))
    throw std::invalid_argument("last dimension of p must equal n_tofbins");

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  // check that the shape of sigma_tof is either [1,] or xstart.shape[:-1]
  if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == 1)
  {
    lor_dependent_sigma_tof = false;
  }
  else if (sigma_tof.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (sigma_tof.ndim()); ++i)
    {
      if (sigma_tof.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_sigma_tof = true;
  }
  else
  {
    throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
  }

  // check that the shape of tofcenter_offset is either [1,] or xstart.shape[:-1]
  if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == 1)
  {
    lor_dependent_tofcenter_offset = false;
  }
  else if (tofcenter_offset.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (tofcenter_offset.ndim()); ++i)
    {
      if (tofcenter_offset.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_tofcenter_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
  }

  joseph3d_tof_sino_fwd(xstart.data(),
                        xend.data(),
                        img.data(),
                        img_origin.data(),
                        voxsize.data(),
                        p.data(),
                        nlors,
                        img_dim,
                        tofbin_width,
                        sigma_tof.data(),
                        tofcenter_offset.data(),
                        n_sigmas,
                        n_tofbins,
                        static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
                        static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
                        device_id,
                        threadsperblock);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_back
void joseph3d_tof_back_fwd_py(ConstFloatNDArray xstart,
                              ConstFloatNDArray xend,
                              Float3DArray img,
                              ConstFloat1D3ELArray img_origin,
                              ConstFloat1D3ELArray voxsize,
                              ConstFloatNDArray p,
                              float tofbin_width,
                              ConstFloatNDArray sigma_tof,
                              ConstFloatNDArray tofcenter_offset,
                              short n_tofbins,
                              float n_sigmas = 3.0f,
                              int device_id = 0,
                              int threadsperblock = 64)
{
  bool lor_dependent_sigma_tof;
  bool lor_dependent_tofcenter_offset;

  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check p has same ndim as xstart
  if (p.ndim() != xstart.ndim())
    throw std::invalid_argument("p must have same number of dimensions as xstart");
  for (size_t i = 0; i < (p.ndim() - 1); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("shape of p[:-1] must match shape of xstart[:-1]");
  }
  // check that p.shape[-1] == n_tofbins
  if (p.shape(p.ndim() - 1) != static_cast<size_t>(n_tofbins))
    throw std::invalid_argument("last dimension of p must equal n_tofbins");

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  // check that the shape of sigma_tof is either [1,] or xstart.shape[:-1]
  if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == 1)
  {
    lor_dependent_sigma_tof = false;
  }
  else if (sigma_tof.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (sigma_tof.ndim()); ++i)
    {
      if (sigma_tof.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_sigma_tof = true;
  }
  else
  {
    throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
  }

  // check that the shape of tofcenter_offset is either [1,] or xstart.shape[:-1]
  if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == 1)
  {
    lor_dependent_tofcenter_offset = false;
  }
  else if (tofcenter_offset.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (tofcenter_offset.ndim()); ++i)
    {
      if (tofcenter_offset.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_tofcenter_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
  }

  joseph3d_tof_sino_back(xstart.data(),
                         xend.data(),
                         img.data(),
                         img_origin.data(),
                         voxsize.data(),
                         p.data(),
                         nlors,
                         img_dim,
                         tofbin_width,
                         sigma_tof.data(),
                         tofcenter_offset.data(),
                         n_sigmas,
                         n_tofbins,
                         static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
                         static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
                         device_id,
                         threadsperblock);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_lm_fwd
void joseph3d_tof_lm_fwd_py(ConstFloatNDArray xstart,
                            ConstFloatNDArray xend,
                            ConstFloat3DArray img,
                            ConstFloat1D3ELArray img_origin,
                            ConstFloat1D3ELArray voxsize,
                            FloatNDArray p,
                            float tofbin_width,
                            ConstFloatNDArray sigma_tof,
                            ConstFloatNDArray tofcenter_offset,
                            ConstShortNDArray tofbin,
                            short n_tofbins,
                            float n_sigmas = 3.0f,
                            int device_id = 0,
                            int threadsperblock = 64)
{
  bool lor_dependent_sigma_tof;
  bool lor_dependent_tofcenter_offset;

  // 1 check that ndim of xstart and xend are equal to 2
  if (xstart.ndim() != 2 || xend.ndim() != 2)
    throw std::invalid_argument("xstart and xend must have 2 dimensions");
  if (xstart.shape(1) != 3 || xend.shape(1) != 3)
    throw std::invalid_argument("xstart and xend must have shape (..., 3)");
  if (xstart.shape(0) != xend.shape(0))
    throw std::invalid_argument("xstart and xend must have the same number of LORs (shape[0])");

  size_t nlors = xstart.shape(0);

  // 3 check dims and shapes p amd tof_bin
  if (p.ndim() != 1)
    throw std::invalid_argument("p.ndim must be 1");
  if (p.shape(0) != nlors)
    throw std::invalid_argument("p.shape[0] must match xstart.shape[0]");

  if (tofbin.ndim() != 1)
    throw std::invalid_argument("tofbin.ndim must be 1");
  if (tofbin.shape(0) != nlors)
    throw std::invalid_argument("tofbin.shape[0] must match xstart.shape[0]");

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type() ||
      xstart.device_type() != tofbin.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id() ||
      xstart.device_id() != tofbin.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  // check that the shape of sigma_tof is either [1,] or xstart.shape[:-1]
  if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == 1)
  {
    lor_dependent_sigma_tof = false;
  }
  else if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == nlors)
  {
    lor_dependent_sigma_tof = true;
  }
  else
  {
    throw std::invalid_argument("shape of sigma_tof must be [1,] or [nlors,]");
  }

  // check that the shape of tofcenter_offset is either [1,] or xstart.shape[:-1]
  if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == 1)
  {
    lor_dependent_tofcenter_offset = false;
  }
  else if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == nlors)
  {
    lor_dependent_tofcenter_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tofcenter_offset must be [1,] or [nlors,]");
  }

  joseph3d_tof_lm_fwd(xstart.data(),
                      xend.data(),
                      img.data(),
                      img_origin.data(),
                      voxsize.data(),
                      p.data(),
                      nlors,
                      img_dim,
                      tofbin_width,
                      sigma_tof.data(),
                      tofcenter_offset.data(),
                      n_sigmas,
                      tofbin.data(),
                      n_tofbins,
                      static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
                      static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
                      device_id,
                      threadsperblock);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_lm_back
void joseph3d_tof_lm_back_py(ConstFloatNDArray xstart,
                             ConstFloatNDArray xend,
                             Float3DArray img,
                             ConstFloat1D3ELArray img_origin,
                             ConstFloat1D3ELArray voxsize,
                             ConstFloatNDArray p,
                             float tofbin_width,
                             ConstFloatNDArray sigma_tof,
                             ConstFloatNDArray tofcenter_offset,
                             ConstShortNDArray tofbin,
                             short n_tofbins,
                             float n_sigmas = 3.0f,
                             int device_id = 0,
                             int threadsperblock = 64)
{
  bool lor_dependent_sigma_tof;
  bool lor_dependent_tofcenter_offset;

  // 1 check that ndim of xstart and xend are equal to 2
  if (xstart.ndim() != 2 || xend.ndim() != 2)
    throw std::invalid_argument("xstart and xend must have 2 dimensions");
  if (xstart.shape(1) != 3 || xend.shape(1) != 3)
    throw std::invalid_argument("xstart and xend must have shape (..., 3)");
  if (xstart.shape(0) != xend.shape(0))
    throw std::invalid_argument("xstart and xend must have the same number of LORs (shape[0])");

  size_t nlors = xstart.shape(0);

  // 3 check dims and shapes p amd tof_bin
  if (p.ndim() != 1)
    throw std::invalid_argument("p.ndim must be 1");
  if (p.shape(0) != nlors)
    throw std::invalid_argument("p.shape[0] must match xstart.shape[0]");

  if (tofbin.ndim() != 1)
    throw std::invalid_argument("tofbin.ndim must be 1");
  if (tofbin.shape(0) != nlors)
    throw std::invalid_argument("tofbin.shape[0] must match xstart.shape[0]");

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type() ||
      xstart.device_type() != tofbin.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id() ||
      xstart.device_id() != tofbin.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  // check that the shape of sigma_tof is either [1,] or xstart.shape[:-1]
  if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == 1)
  {
    lor_dependent_sigma_tof = false;
  }
  else if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == nlors)
  {
    lor_dependent_sigma_tof = true;
  }
  else
  {
    throw std::invalid_argument("shape of sigma_tof must be [1,] or [nlors,]");
  }

  // check that the shape of tofcenter_offset is either [1,] or xstart.shape[:-1]
  if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == 1)
  {
    lor_dependent_tofcenter_offset = false;
  }
  else if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == nlors)
  {
    lor_dependent_tofcenter_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tofcenter_offset must be [1,] or [nlors,]");
  }

  joseph3d_tof_lm_back(xstart.data(),
                       xend.data(),
                       img.data(),
                       img_origin.data(),
                       voxsize.data(),
                       p.data(),
                       nlors,
                       img_dim,
                       tofbin_width,
                       sigma_tof.data(),
                       tofcenter_offset.data(),
                       n_sigmas,
                       tofbin.data(),
                       n_tofbins,
                       static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
                       static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
                       device_id,
                       threadsperblock);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

NB_MODULE(parallelproj_backend, m)
{
  m.doc() = R"pbdoc(
    Python bindings for parallelproj backend.

    This module provides efficient C++/CUDA implementations of 3D Joseph
    forward and backward projectors with optional Time-of-Flight (TOF) support.

    Attributes
    ----------
    __version__ : str
        The version of the parallelproj backend library.
    PARALLELPROJ_CUDA : int
        Flag indicating if the library was compiled with CUDA support.
        Returns 1 if CUDA is enabled, 0 otherwise.
  )pbdoc";

  // Expose the project version as __version__
#ifdef PROJECT_VERSION
  m.attr("__version__") = PROJECT_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif

  // Expose the PARALLELPROJ_CUDA definition as a Python constant
#ifdef PARALLELPROJ_CUDA
  m.attr("PARALLELPROJ_CUDA") = PARALLELPROJ_CUDA;
#else
  m.attr("PARALLELPROJ_CUDA") = 0; // Default to 0 if not defined
#endif

  m.def("joseph3d_fwd", &joseph3d_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = 0, "threads_per_block"_a = 64,
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
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        All arrays must be on the same device (CPU or GPU) with matching device IDs.
        )pbdoc");

  m.def("joseph3d_back", &joseph3d_back_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "device_id"_a = 0, "threads_per_block"_a = 64,
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
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

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
        "device_id"_a = 0, "threads_per_block"_a = 64,
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
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        )pbdoc");

  m.def("joseph3d_tof_sino_back", &joseph3d_tof_back_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threads_per_block"_a = 64,
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
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        )pbdoc");

  m.def("joseph3d_tof_lm_fwd", &joseph3d_tof_lm_fwd_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        TOF listmode forward projection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (..., 3) with coordinates of event LOR start points.
        lor_end : ndarray
            Array of shape (..., 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) for forward projection.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
          Output array for event projections, shape matches lor_start.shape[:-1].
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (...,) for event-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (...,).
        tof_bin_index : ndarray
            TOF bin centers for each event in spatial units, shape matches xstart.shape[:-1].
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).
        )pbdoc");

  m.def("joseph3d_tof_lm_back", &joseph3d_tof_lm_back_py,
        "lor_start"_a.noconvert(), "lor_end"_a.noconvert(), "image"_a.noconvert(),
        "image_origin"_a.noconvert(), "voxel_size"_a.noconvert(), "projection_values"_a.noconvert(),
        "tof_bin_width"_a,
        "tof_sigma"_a.noconvert(),
        "tof_center_offset"_a.noconvert(),
        "tof_bin_index"_a.noconvert(),
        "num_tof_bins"_a,
        "num_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threads_per_block"_a = 64,
        R"pbdoc(
        TOF listmode backprojection using the Joseph 3D algorithm.

        Parameters
        ----------
        lor_start : ndarray
            Array of shape (nlors, 3) with coordinates of event LOR start points.
        lor_end : ndarray
            Array of shape (nlors, 3) with coordinates of event LOR end points.
        image : ndarray
            3D image array of shape (n0, n1, n2) to accumulate backprojection into.
        image_origin : ndarray
            Array of shape (3,) with coordinates of [0,0,0] voxel center.
        voxel_size : ndarray
            Array of shape (3,) with voxel sizes.
        projection_values : ndarray
            Input array to be back projected, shape (nlors,)
        tof_bin_width : float
            Width of TOF bins in spatial units.
        tof_sigma : ndarray
            TOF resolution (sigma) in spatial units. Shape (1,) or (nlors,) for event-dependent.
        tof_center_offset : ndarray
            Offset of central TOF bin from LOR midpoint. Shape (1,) or (nlors,).
        tof_bin_index : ndarray
            TOF bin indices for each event, shape (nlors,).
        num_tof_bins : int
            Number of TOF bins.
        num_sigmas : float, optional
            Number of sigmas for TOF kernel calculation (default: 3.0).
        device_id : int, optional
            Device ID for computation (default: 0).
        threads_per_block : int, optional
            Number of threads per block for GPU (default: 64).

        Notes
        -----
        Values are accumulated into the existing image array (not overwritten).
        )pbdoc");
}
