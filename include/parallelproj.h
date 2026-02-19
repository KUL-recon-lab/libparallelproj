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
  \pp_param_count{num_lors,geometrical LORs}
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
  \pp_param_count{num_lors,geometrical LORs}
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
  \pp_param_count{num_lors,geometrical LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_lors,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_num_bins
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

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_back
  \pp_param_image_origin
  \pp_param_voxel_size
  @param projection_values Pointer to TOF sinogram data of length @p num_lors * @p num_tof_bins (see details).
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_lors,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_num_bins
  @param is_lor_dependent_tof_sigma If non-zero, @p tof_sigma contains one sigma per LOR; otherwise the first
                                 element is used for all LORs.
  @param is_lor_dependent_tof_center_offset If non-zero, @p tof_center_offset contains one offset per LOR;
                                        otherwise the first element is used for all LORs.
  \pp_param_device_id
  \pp_param_threads

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
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_events,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  @param tof_bin_index Pointer to array of length @p num_events with the TOF bin numbers.
  \pp_param_tof_num_bins
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
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_events,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  @param tof_bin_index Pointer to array of length @p num_events with the TOF bin numbers.
  \pp_param_tof_num_bins
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
