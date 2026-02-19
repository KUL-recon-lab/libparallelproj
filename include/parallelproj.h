/*!
@file parallelproj.h
*/

#pragma once
#include <cstddef>
// import parallelproj_export.h to get the PARALLELPROJ_API macro
// needed for __declspec(dllexport) and __declspec(dllimport)
// This is needed for Windows to export the functions in the DLL
#include "parallelproj_export.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief Forward projection using the Joseph 3D algorithm.

  @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.

  @param lor_start Pointer to array of shape [3*num_lors] with the coordinates of the start points of the LORs.
                   The start coordinates of the n-th LOR are at lor_start[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param lor_end   Pointer to array of shape [3*num_lors] with the coordinates of the end points of the LORs.
                   The end coordinates of the n-th LOR are at lor_end[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param image    Pointer to array of shape [n0*n1*n2] containing the 3D image used for forward projection.
                The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
  @param image_origin Pointer to array [x0_0, x0_1, x0_2] of coordinates of the center of the [0,0,0] voxel.
  @param voxel_size Pointer to array [vs0, vs1, vs2] of the voxel sizes.
  @param projection_values Pointer to array of length num_lors where the forward projection results will be stored.
  @param num_lors  Number of geometrical LORs.
  @param image_dim Pointer to array with dimensions of the image [n0, n1, n2].
  @param device_id ID of the device to use for computation (default: 0).
  @param threads_per_block Number of threads per block for GPU computation (default: 64).
  */
  PARALLELPROJ_API void joseph3d_fwd(const float *lor_start,
                                     const float *lor_end,
                                     const float *image,
                                     const float *image_origin,
                                     const float *voxel_size,
                                     float *projection_values,
                                     size_t num_lors,
                                     const int *image_dim,
                                     int device_id = 0,
                                     int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief Backprojection using the Joseph 3D algorithm.

  @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.

  @param lor_start Pointer to array of shape [3*num_lors] with the coordinates of the start points of the LORs.
                   The start coordinates of the n-th LOR are at lor_start[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param lor_end   Pointer to array of shape [3*num_lors] with the coordinates of the end points of the LORs.
                   The end coordinates of the n-th LOR are at lor_end[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param image    Pointer to array of shape [n0*n1*n2] containing the 3D image used for backprojection (output).
                The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
                !! Values are added to the existing array !!
  @param image_origin Pointer to array [x0_0, x0_1, x0_2] of coordinates of the center of the [0,0,0] voxel.
  @param voxel_size Pointer to array [vs0, vs1, vs2] of the voxel sizes.
  @param projection_values Pointer to array of length num_lors with the values to be backprojected.
  @param num_lors  Number of geometrical LORs.
  @param image_dim Pointer to array with dimensions of the image [n0, n1, n2].
  @param device_id ID of the device to use for computation (default: 0).
  @param threads_per_block Number of threads per block for GPU computation (default: 64).
   */
  PARALLELPROJ_API void joseph3d_back(const float *lor_start,
                                      const float *lor_end,
                                      float *image,
                                      const float *image_origin,
                                      const float *voxel_size,
                                      const float *projection_values,
                                      size_t num_lors,
                                      const int *image_dim,
                                      int device_id = 0,
                                      int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D sinogram TOF Joseph forward projector

  @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.

  @param lor_start pointer to array of shape [3*num_lors] with the coordinates of the start points of the LORs.
                   The start coordinates of the n-th LOR are at lor_start[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param lor_end   pointer array of shape [3*num_lors] with the coordinates of the end points of the LORs.
                   The end coordinates of the n-th LOR are at lor_end[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param image    pointer array of shape [n0*n1*n2] containing the 3D image to be projected.
                The voxel [i,j,k] is stored at index n1*n2*i + n2*j + k.
  @param image_origin  pointer array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel.
  @param voxel_size     pointer array [vs0, vs1, vs2] of the voxel sizes.
  @param projection_values pointer to array of length num_lors*num_tof_bins (output) used to store the projections.
                     The ordering is row-major per LOR:
                     [LOR0-TOFBIN-0, LOR0-TOFBIN-1, ... LOR0-TOFBIN-(n-1),
                      LOR1-TOFBIN-0, LOR1-TOFBIN-1, ... LOR1-TOFBIN-(n-1),
                      ...
                      LOR(N-1)-TOFBIN-0, LOR(N-1)-TOFBIN-1, ... LOR(N-1)-TOFBIN-(n-1)]
  @param num_lors       number of geometrical LORs
  @param image_dim     array with dimensions of image [n0,n1,n2]
  @param tof_bin_width     width of the TOF bins in spatial units (units of lor_start and lor_end)
  @param tof_sigma        pointer to array of length 1 or num_lors (depending on is_lor_dependent_tof_sigma)
                          with the TOF resolution (sigma) for each LOR in spatial units
                          (units of lor_start and lor_end)
  @param tof_center_offset pointer to array of length 1 or num_lors (depending on is_lor_dependent_tof_center_offset)
                          with the offset of the central TOF bin from the midpoint of each LOR in spatial units.
                          A positive value means a shift towards the end point of the LOR.
  @param num_sigmas         number of sigmas to consider for calculation of TOF kernel
  @param num_tof_bins        number of TOF bins
  @param is_lor_dependent_tof_sigma unsigned char 0 or 1
                                 0 means that the first value in the tof_sigma array is used for all LORs
                                 1 (non-zero) means that the TOF resolutions are LOR dependent
  @param is_lor_dependent_tof_center_offset unsigned char 0 or 1
                                        0 means that the first value in the tof_center_offset array is used for all LORs
                                        1 (non-zero) means that the TOF center offsets are LOR dependent
  @param device_id ID of the device to use for computation (default: 0).
  @param threads_per_block Number of threads per block for GPU computation (default: 64).
  */

  PARALLELPROJ_API void joseph3d_tof_sino_fwd(const float *lor_start,
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
                                              int device_id = 0,
                                              int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief TOF sinogram backprojection using the Joseph 3D algorithm.

  @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.

  @details The function backprojects a TOF sinogram into a 3D image volume using the Joseph
           ray-driven algorithm. The projection data @p projection_values is organized row-major per LOR:
           [LOR0-TOFBIN-0, LOR0-TOFBIN-1, ..., LOR0-TOFBIN-(num_tof_bins-1),
            LOR1-TOFBIN-0, LOR1-TOFBIN-1, ..., LOR1-TOFBIN-(num_tof_bins-1), ...].
           Values from @p projection_values are distributed into @p image (accumulated, not overwritten).

  @param lor_start Pointer to array of shape [3*num_lors] with start coordinates for each LOR
                (lor_start[n*3 + i], i=0..2). Units are those of @p voxel_size.
  @param lor_end   Pointer to array of shape [3*num_lors] with end coordinates for each LOR
                (lor_end[n*3 + i], i=0..2). Units are those of @p voxel_size.
  @param image    Pointer to array of shape [n0*n1*n2] containing the 3D image to add backprojected
                contributions into. The element (i,j,k) is stored at index n1*n2*i + n2*j + k.
                Values are added to the existing contents of this array.
  @param image_origin Pointer to array [x0_0, x0_1, x0_2] giving the coordinates of the center of the
                    voxel at index [0,0,0].
  @param voxel_size Pointer to array [vs0, vs1, vs2] specifying voxel sizes in the same units as LOR coords.
  @param projection_values Pointer to TOF sinogram data of length num_lors * num_tof_bins (see details).
  @param num_lors  Number of geometric LORs.
  @param image_dim Pointer to array [n0, n1, n2] with image dimensions. Can be host/device/managed.
  @param tof_bin_width Width of each TOF bin in spatial units (same units as LOR coordinates).
  @param tof_sigma Pointer to array of length 1 or num_lors (depending on
                   is_lor_dependent_tof_sigma) specifying TOF sigma(s) in spatial units.
  @param tof_center_offset Pointer to array of length 1 or num_lors (depending on
                          is_lor_dependent_tof_center_offset) specifying per-LOR offset of the
                          central TOF bin from the geometric midpoint (positive towards lor_end).
  @param num_sigmas Number of sigmas to consider when evaluating the TOF kernel (controls kernel radius).
  @param num_tof_bins Number of TOF bins per LOR.
  @param is_lor_dependent_tof_sigma If non-zero, @p tof_sigma contains one sigma per LOR; otherwise the first
                                 element is used for all LORs.
  @param is_lor_dependent_tof_center_offset If non-zero, @p tof_center_offset contains one offset per LOR;
                                        otherwise the first element is used for all LORs.
  @param device_id CUDA device to use (default: 0). If negative, CPU path is used when available.
  @param threads_per_block Number of CUDA threads per block for GPU execution (default: 64).

  @return void
  */
  PARALLELPROJ_API void joseph3d_tof_sino_back(const float *lor_start,
                                               const float *lor_end,
                                               float *image,
                                               const float *image_origin,
                                               const float *voxel_size,
                                               const float *projection_values,
                                               size_t num_lors,
                                               const int *image_dim,
                                               float tof_bin_width,
                                               const float *tof_sigma,
                                               const float *tof_center_offset,
                                               float num_sigmas,
                                               short num_tof_bins,
                                               unsigned char is_lor_dependent_tof_sigma,
                                               unsigned char is_lor_dependent_tof_center_offset,
                                               int device_id = 0,
                                               int threads_per_block = 64);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D listmode TOF Joseph forward projector

  @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.

  @param lor_start pointer to array of shape [3*num_lors] with the coordinates of the start points of the event LORs.
                   The start coordinates of the n-th event are at lor_start[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param lor_end   pointer array of shape [3*num_lors] with the coordinates of the end points of the event LORs.
                   The end coordinates of the n-th event are at lor_end[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param image    pointer array of shape [n0*n1*n2] containing the 3D image to be projected.
                The voxel [i,j,k] is stored at index n1*n2*i + n2*j + k.
  @param image_origin  pointer array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel.
  @param voxel_size     pointer array [vs0, vs1, vs2] of the voxel sizes.
  @param projection_values pointer to array of length number of events (output) used to store the projections.
  @param num_lors       number of events
  @param image_dim     array with dimensions of image [n0,n1,n2]
  @param tof_bin_width     width of the TOF bins in spatial units (units of lor_start and lor_end)
  @param tof_sigma        pointer to array of length 1 or number of events (depending on is_lor_dependent_tof_sigma)
                          with the TOF resolution (sigma) for each event in spatial units
                          (units of lor_start and lor_end)
  @param tof_center_offset pointer to array of length 1 or number of events (depending on is_lor_dependent_tof_center_offset)
                          with the offset of the central TOF bin from the midpoint of each LOR in spatial units.
                          A positive value means a shift towards the end point of the LOR.
  @param num_sigmas         number of sigmas to consider for calculation of TOF kernel
  @param tof_bin_index           pointer to array of length number of events with the TOF bin numbers
  @param num_tof_bins        number of TOF bins
  @param is_lor_dependent_tof_sigma unsigned char 0 or 1
                                 0 means that the first value in the tof_sigma array is used for all LORs
                                 1 (non-zero) means that the TOF resolutions are LOR dependent
  @param is_lor_dependent_tof_center_offset unsigned char 0 or 1
                                        0 means that the first value in the tof_center_offset array is used for all LORs
                                        1 (non-zero) means that the TOF center offsets are LOR dependent
  @param device_id ID of the device to use for computation (default: 0).
  @param threads_per_block Number of threads per block for GPU computation (default: 64).
  */

  PARALLELPROJ_API void joseph3d_tof_lm_fwd(const float *lor_start,
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
                                            const short *tof_bin_index,
                                            short num_tof_bins,
                                            unsigned char is_lor_dependent_tof_sigma,
                                            unsigned char is_lor_dependent_tof_center_offset,
                                            int device_id = 0,
                                            int threads_per_block = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D listmode TOF Joseph back projector

  @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.

  @param lor_start pointer to array of shape [3*num_lors] with the coordinates of the start points of the event LORs.
                   The start coordinates of the n-th event are at lor_start[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param lor_end   pointer array of shape [3*num_lors] with the coordinates of the end points of the event LORs.
                   The end coordinates of the n-th event are at lor_end[n*3 + i] with i = 0,1,2.
                   Units are the ones of voxel_size.
  @param image    Pointer to array of shape [n0*n1*n2] containing the 3D image to add backprojected
                contributions into. The element (i,j,k) is stored at index n1*n2*i + n2*j + k.
                Values are added to the existing contents of this array.
  @param image_origin  pointer array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel.
  @param voxel_size     pointer array [vs0, vs1, vs2] of the voxel sizes.
  @param projection_values pointer to array of values to be backprojected (length number of events / num_lors)
  @param num_lors       number of events
  @param image_dim     array with dimensions of image [n0,n1,n2]
  @param tof_bin_width     width of the TOF bins in spatial units (units of lor_start and lor_end)
  @param tof_sigma        pointer to array of length 1 or number of events (depending on is_lor_dependent_tof_sigma)
                          with the TOF resolution (sigma) for each event in spatial units
                          (units of lor_start and lor_end)
  @param tof_center_offset pointer to array of length 1 or number of events (depending on is_lor_dependent_tof_center_offset)
                          with the offset of the central TOF bin from the midpoint of each LOR in spatial units.
                          A positive value means a shift towards the end point of the LOR.
  @param num_sigmas         number of sigmas to consider for calculation of TOF kernel
  @param tof_bin_index           pointer to array of length number of events with the TOF bin numbers
  @param num_tof_bins        number of TOF bins
  @param is_lor_dependent_tof_sigma unsigned char 0 or 1
                                 0 means that the first value in the tof_sigma array is used for all LORs
                                 1 (non-zero) means that the TOF resolutions are LOR dependent
  @param is_lor_dependent_tof_center_offset unsigned char 0 or 1
                                        0 means that the first value in the tof_center_offset array is used for all LORs
                                        1 (non-zero) means that the TOF center offsets are LOR dependent
  @param device_id ID of the device to use for computation (default: 0).
  @param threads_per_block Number of threads per block for GPU computation (default: 64).
  */

  PARALLELPROJ_API void joseph3d_tof_lm_back(const float *lor_start,
                                             const float *lor_end,
                                             float *image,
                                             const float *image_origin,
                                             const float *voxel_size,
                                             const float *projection_values,
                                             size_t num_lors,
                                             const int *image_dim,
                                             float tof_bin_width,
                                             const float *tof_sigma,
                                             const float *tof_center_offset,
                                             float num_sigmas,
                                             const short *tof_bin_index,
                                             short num_tof_bins,
                                             unsigned char is_lor_dependent_tof_sigma,
                                             unsigned char is_lor_dependent_tof_center_offset,
                                             int device_id = 0,
                                             int threads_per_block = 64);

#ifdef __cplusplus
}
#endif
