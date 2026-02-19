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

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_fwd
  \pp_param_image_origin
  \pp_param_voxel_size
  @param projection_values Pointer to array of length @p num_lors where the forward projection results will be stored.
  @param num_lors  Number of geometrical LORs.
  \pp_param_image_dim
  \pp_param_device_id
  \pp_param_threads
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

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_back
  \pp_param_image_origin
  \pp_param_voxel_size
  @param projection_values Pointer to array of length @p num_lors with the values to be backprojected.
  @param num_lors  Number of geometrical LORs.
  \pp_param_image_dim
  \pp_param_device_id
  \pp_param_threads
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

  \pp_note_pointer_types

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_fwd
  \pp_param_image_origin
  \pp_param_voxel_size
  @param projection_values Pointer to array of length @p num_lors * @p num_tof_bins (output) used to store the projections.
                     The ordering is row-major per LOR:
                     [LOR0-TOFBIN-0, LOR0-TOFBIN-1, ... LOR0-TOFBIN-(n-1),
                      LOR1-TOFBIN-0, LOR1-TOFBIN-1, ... LOR1-TOFBIN-(n-1),
                      ...
                      LOR(N-1)-TOFBIN-0, LOR(N-1)-TOFBIN-1, ... LOR(N-1)-TOFBIN-(n-1)]
  @param num_lors Number of geometrical LORs.
  @param image_dim Pointer to array with image dimensions [n0, n1, n2].
  @param tof_bin_width Width of the TOF bins in spatial units (units of @p lor_start and @p lor_end).
  @param tof_sigma Pointer to array of length 1 or @p num_lors (depending on @p is_lor_dependent_tof_sigma)
                          with the TOF resolution (sigma) for each LOR in spatial units
                          (units of @p lor_start and @p lor_end).
  @param tof_center_offset Pointer to array of length 1 or @p num_lors (depending on @p is_lor_dependent_tof_center_offset)
                          with the offset of the central TOF bin from the midpoint of each LOR in spatial units.
                          A positive value means a shift towards the end point of the LOR.
  @param num_sigmas Number of sigmas to consider for calculation of the TOF kernel.
  @param num_tof_bins Number of TOF bins.
  @param is_lor_dependent_tof_sigma Unsigned char 0 or 1.
                                 0 means that the first value in @p tof_sigma is used for all LORs.
                                 1 (non-zero) means that the TOF resolutions are LOR dependent
  @param is_lor_dependent_tof_center_offset Unsigned char 0 or 1.
                                        0 means that the first value in @p tof_center_offset is used for all LORs.
                                        1 (non-zero) means that the TOF center offsets are LOR dependent
  \pp_param_device_id
  \pp_param_threads
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

  \pp_note_pointer_types

  @details The function backprojects a TOF sinogram into a 3D image volume using the Joseph
           ray-driven algorithm. The projection data @p projection_values is organized row-major per LOR:
           [LOR0-TOFBIN-0, LOR0-TOFBIN-1, ..., LOR0-TOFBIN-(num_tof_bins-1),
            LOR1-TOFBIN-0, LOR1-TOFBIN-1, ..., LOR1-TOFBIN-(num_tof_bins-1), ...].
           Values from @p projection_values are distributed into @p image (accumulated, not overwritten).

  @param lor_start Pointer to array of shape [3 * @p num_lors] with start coordinates for each LOR
                (lor_start[n*3 + i], i=0..2). Units are those of @p voxel_size.
  @param lor_end   Pointer to array of shape [3 * @p num_lors] with end coordinates for each LOR
                (lor_end[n*3 + i], i=0..2). Units are those of @p voxel_size.
  @param image    Pointer to array of shape [n0*n1*n2] containing the 3D image to add backprojected
                contributions into. The element (i,j,k) is stored at index n1*n2*i + n2*j + k.
                Values are added to the existing contents of this array.
  @param image_origin Pointer to array [x0_0, x0_1, x0_2] giving the coordinates of the center of the
                    voxel at index [0,0,0].
  @param voxel_size Pointer to array [vs0, vs1, vs2] specifying voxel sizes in the same units as LOR coordinates.
  @param projection_values Pointer to TOF sinogram data of length @p num_lors * @p num_tof_bins (see details).
  @param num_lors  Number of geometric LORs.
  @param image_dim Pointer to array [n0, n1, n2] with image dimensions. Can be host pointers, CUDA device pointers, or CUDA managed pointers.
  @param tof_bin_width Width of each TOF bin in spatial units (same units as LOR coordinates).
  @param tof_sigma Pointer to array of length 1 or @p num_lors (depending on
                   @p is_lor_dependent_tof_sigma) specifying TOF sigma(s) in spatial units.
  @param tof_center_offset Pointer to array of length 1 or @p num_lors (depending on
                          @p is_lor_dependent_tof_center_offset) specifying per-LOR offset of the
                          central TOF bin from the geometric midpoint (positive towards @p lor_end).
  @param num_sigmas Number of sigmas to consider when evaluating the TOF kernel (controls kernel radius).
  @param num_tof_bins Number of TOF bins per LOR.
  @param is_lor_dependent_tof_sigma If non-zero, @p tof_sigma contains one sigma per LOR; otherwise the first
                                 element is used for all LORs.
  @param is_lor_dependent_tof_center_offset If non-zero, @p tof_center_offset contains one offset per LOR;
                                        otherwise the first element is used for all LORs.
  @param device_id CUDA device to use (default: 0). If negative, CPU path is used when available.
  @param threads_per_block Number of CUDA threads per block for GPU execution (default: 64).

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

  \pp_note_pointer_types

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_fwd
  \pp_param_image_origin
  \pp_param_voxel_size
  @param projection_values Pointer to array of length @p num_events (output) used to store the projections.
  @param num_events Number of events.
  @param image_dim Pointer to array with image dimensions [n0, n1, n2].
  @param tof_bin_width Width of the TOF bins in spatial units (units of @p lor_start and @p lor_end).
  @param tof_sigma Pointer to array of length 1 or @p num_events (depending on @p is_lor_dependent_tof_sigma)
                          with the TOF resolution (sigma) for each event in spatial units
                          (units of @p lor_start and @p lor_end).
  @param tof_center_offset Pointer to array of length 1 or @p num_events (depending on @p is_lor_dependent_tof_center_offset)
                          with the offset of the central TOF bin from the midpoint of each LOR in spatial units.
                          A positive value means a shift towards the end point of the LOR.
  @param num_sigmas Number of sigmas to consider for calculation of the TOF kernel.
  @param tof_bin_index Pointer to array of length @p num_events with the TOF bin numbers.
  @param num_tof_bins Number of TOF bins.
  @param is_lor_dependent_tof_sigma Unsigned char 0 or 1.
                                 0 means that the first value in @p tof_sigma is used for all events.
                                 1 (non-zero) means that the TOF resolutions are LOR dependent
  @param is_lor_dependent_tof_center_offset Unsigned char 0 or 1.
                                        0 means that the first value in @p tof_center_offset is used for all events.
                                        1 (non-zero) means that the TOF center offsets are LOR dependent
  \pp_param_device_id
  \pp_param_threads
  */

  PARALLELPROJ_API void joseph3d_tof_lm_fwd(const float *lor_start,
                                            const float *lor_end,
                                            const float *image,
                                            const float *image_origin,
                                            const float *voxel_size,
                                            float *projection_values,
                                            size_t num_events,
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

  \pp_note_pointer_types

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_back
  \pp_param_image_origin
  \pp_param_voxel_size
  @param projection_values Pointer to array of values to be backprojected (length @p num_events).
  @param num_events Number of events.
  @param image_dim Pointer to array with image dimensions [n0, n1, n2].
  @param tof_bin_width Width of the TOF bins in spatial units (units of @p lor_start and @p lor_end).
  @param tof_sigma Pointer to array of length 1 or @p num_events (depending on @p is_lor_dependent_tof_sigma)
                          with the TOF resolution (sigma) for each event in spatial units
                          (units of @p lor_start and @p lor_end).
  @param tof_center_offset Pointer to array of length 1 or @p num_events (depending on @p is_lor_dependent_tof_center_offset)
                          with the offset of the central TOF bin from the midpoint of each LOR in spatial units.
                          A positive value means a shift towards the end point of the LOR.
  @param num_sigmas Number of sigmas to consider for calculation of the TOF kernel.
  @param tof_bin_index Pointer to array of length @p num_events with the TOF bin numbers.
  @param num_tof_bins Number of TOF bins.
  @param is_lor_dependent_tof_sigma Unsigned char 0 or 1.
                                 0 means that the first value in @p tof_sigma is used for all events.
                                 1 (non-zero) means that the TOF resolutions are LOR dependent
  @param is_lor_dependent_tof_center_offset Unsigned char 0 or 1.
                                        0 means that the first value in @p tof_center_offset is used for all events.
                                        1 (non-zero) means that the TOF center offsets are LOR dependent
  \pp_param_device_id
  \pp_param_threads
  */

  PARALLELPROJ_API void joseph3d_tof_lm_back(const float *lor_start,
                                             const float *lor_end,
                                             float *image,
                                             const float *image_origin,
                                             const float *voxel_size,
                                             const float *projection_values,
                                             size_t num_events,
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
