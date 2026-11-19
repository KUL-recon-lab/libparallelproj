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

  /*! @brief Query whether this libparallelproj library was compiled with CUDA support.

  @return 1 if compiled with CUDA support, 0 otherwise.
  */
  PARALLELPROJ_API int parallelproj_cuda_enabled(void);

  /*! @brief Query the version string of this libparallelproj library. Potentially includes extra suffix for "dirty" versions.

  @return Null-terminated version string.
  */
  PARALLELPROJ_API const char *parallelproj_version(void);

  /*! @brief Query the major version number of this libparallelproj library.

  @return Major version number.
  */
  PARALLELPROJ_API int parallelproj_version_major(void);

  /*! @brief Query the minor version number of this libparallelproj library.

  @return Minor version number.
  */
  PARALLELPROJ_API int parallelproj_version_minor(void);

  /*! @brief Query the patch version number of this libparallelproj library.

  @return Patch version number.
  */
  PARALLELPROJ_API int parallelproj_version_patch(void);

#ifdef __cplusplus
}
#endif

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
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_fwd{num_lors}
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_image_dim
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */
  PARALLELPROJ_API void joseph3d_fwd(const float *lor_start,
                                     const float *lor_end,
                                     const float *image,
                                     const float *image_origin,
                                     const float *voxel_size,
                                     float *projection_values,
                                     std::size_t num_lors,
                                     const int *image_dim,
                                     int device_id = 0,
                                     int threads_per_block = 64,
                                     void *stream = nullptr);

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
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_back{num_lors}
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_image_dim
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
   */
  PARALLELPROJ_API void joseph3d_back(const float *lor_start,
                                      const float *lor_end,
                                      float *image,
                                      const float *image_origin,
                                      const float *voxel_size,
                                      const float *projection_values,
                                      std::size_t num_lors,
                                      const int *image_dim,
                                      int device_id = 0,
                                      int threads_per_block = 64,
                                      void *stream = nullptr);

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
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_sino_out{num_lors,num_tof_bins}
  \pp_note_tof_sino_layout
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_lors,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{LORs}
  \pp_param_tof_dep_center{LORs}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */

  PARALLELPROJ_API void joseph3d_tof_sino_fwd(const float *lor_start,
                                              const float *lor_end,
                                              const float *image,
                                              const float *image_origin,
                                              const float *voxel_size,
                                              float *projection_values,
                                              std::size_t num_lors,
                                              const int *image_dim,
                                              float tof_bin_width,
                                              const float *tof_sigma,
                                              const float *tof_center_offset,
                                              float num_sigmas,
                                              short num_tof_bins,
                                              unsigned char is_lor_dependent_tof_sigma,
                                              unsigned char is_lor_dependent_tof_center_offset,
                                              int device_id = 0,
                                              int threads_per_block = 64,
                                              void *stream = nullptr);

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
           ray-driven algorithm. Values from @p projection_values are distributed into @p image
           (accumulated, not overwritten).

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_back
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_sino_in{num_lors,num_tof_bins}
  \pp_note_tof_sino_layout
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_lors,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{LORs}
  \pp_param_tof_dep_center{LORs}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream

  */
  PARALLELPROJ_API void joseph3d_tof_sino_back(const float *lor_start,
                                               const float *lor_end,
                                               float *image,
                                               const float *image_origin,
                                               const float *voxel_size,
                                               const float *projection_values,
                                               std::size_t num_lors,
                                               const int *image_dim,
                                               float tof_bin_width,
                                               const float *tof_sigma,
                                               const float *tof_center_offset,
                                               float num_sigmas,
                                               short num_tof_bins,
                                               unsigned char is_lor_dependent_tof_sigma,
                                               unsigned char is_lor_dependent_tof_center_offset,
                                               int device_id = 0,
                                               int threads_per_block = 64,
                                               void *stream = nullptr);
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
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_out{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_events,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{events}
  \pp_param_tof_dep_center{events}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */

  PARALLELPROJ_API void joseph3d_tof_lm_fwd(const float *lor_start,
                                            const float *lor_end,
                                            const float *image,
                                            const float *image_origin,
                                            const float *voxel_size,
                                            float *projection_values,
                                            std::size_t num_events,
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
                                            int threads_per_block = 64,
                                            void *stream = nullptr);

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
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_back{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_sigma{num_events,is_lor_dependent_tof_sigma}
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_sigmas
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_dep_sigma{events}
  \pp_param_tof_dep_center{events}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */

  PARALLELPROJ_API void joseph3d_tof_lm_back(const float *lor_start,
                                             const float *lor_end,
                                             float *image,
                                             const float *image_origin,
                                             const float *voxel_size,
                                             const float *projection_values,
                                             std::size_t num_events,
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
                                             int threads_per_block = 64,
                                             void *stream = nullptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief TOF sinogram forward projection (Joseph 3D) with a tabulated TOF kernel.

  Same as joseph3d_tof_sino_fwd(), but with a tabulated (arbitrary, possibly asymmetric) TOF kernel
  instead of the Gaussian kernel (parameters tof_sigma, num_sigmas and
  is_lor_dependent_tof_sigma are replaced by the table and the kernel scale).

  \pp_note_pointer_types
  \pp_note_tof_kernel_table

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_fwd
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_sino_out{num_lors,num_tof_bins}
  \pp_note_tof_sino_layout
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_bins
  \pp_param_tof_kernel_cdf
  \pp_param_tof_kernel_num_samples
  \pp_param_tof_kernel_u_min
  \pp_param_tof_kernel_u_max
  \pp_param_tof_kernel_scale{num_lors,is_lor_dependent_tof_kernel_scale}
  \pp_param_tof_dep_center{LORs}
  \pp_param_tof_dep_kernel_scale{LORs}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */
  PARALLELPROJ_API void joseph3d_tof_sino_fwd_tab(const float *lor_start,
                                                  const float *lor_end,
                                                  const float *image,
                                                  const float *image_origin,
                                                  const float *voxel_size,
                                                  float *projection_values,
                                                  std::size_t num_lors,
                                                  const int *image_dim,
                                                  float tof_bin_width,
                                                  const float *tof_center_offset,
                                                  short num_tof_bins,
                                                  const float *tof_kernel_cdf,
                                                  int tof_kernel_num_samples,
                                                  float tof_kernel_u_min,
                                                  float tof_kernel_u_max,
                                                  const float *tof_kernel_scale,
                                                  unsigned char is_lor_dependent_tof_center_offset,
                                                  unsigned char is_lor_dependent_tof_kernel_scale,
                                                  int device_id = 0,
                                                  int threads_per_block = 64,
                                                  void *stream = nullptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief TOF sinogram back projection (Joseph 3D) with a tabulated TOF kernel.

  Same as joseph3d_tof_sino_back(), but with a tabulated (arbitrary, possibly asymmetric) TOF kernel
  instead of the Gaussian kernel (parameters tof_sigma, num_sigmas and
  is_lor_dependent_tof_sigma are replaced by the table and the kernel scale).

  \pp_note_pointer_types
  \pp_note_tof_kernel_table

  \pp_param_lor_start{num_lors}
  \pp_param_lor_end{num_lors}
  \pp_param_image_back
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_sino_in{num_lors,num_tof_bins}
  \pp_note_tof_sino_layout
  \pp_param_count{num_lors,geometric LORs}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_center_offset{num_lors,is_lor_dependent_tof_center_offset}
  \pp_param_tof_num_bins
  \pp_param_tof_kernel_cdf
  \pp_param_tof_kernel_num_samples
  \pp_param_tof_kernel_u_min
  \pp_param_tof_kernel_u_max
  \pp_param_tof_kernel_scale{num_lors,is_lor_dependent_tof_kernel_scale}
  \pp_param_tof_dep_center{LORs}
  \pp_param_tof_dep_kernel_scale{LORs}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */
  PARALLELPROJ_API void joseph3d_tof_sino_back_tab(const float *lor_start,
                                                   const float *lor_end,
                                                   float *image,
                                                   const float *image_origin,
                                                   const float *voxel_size,
                                                   const float *projection_values,
                                                   std::size_t num_lors,
                                                   const int *image_dim,
                                                   float tof_bin_width,
                                                   const float *tof_center_offset,
                                                   short num_tof_bins,
                                                   const float *tof_kernel_cdf,
                                                   int tof_kernel_num_samples,
                                                   float tof_kernel_u_min,
                                                   float tof_kernel_u_max,
                                                   const float *tof_kernel_scale,
                                                   unsigned char is_lor_dependent_tof_center_offset,
                                                   unsigned char is_lor_dependent_tof_kernel_scale,
                                                   int device_id = 0,
                                                   int threads_per_block = 64,
                                                   void *stream = nullptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief TOF listmode forward projection (Joseph 3D) with a tabulated TOF kernel.

  Same as joseph3d_tof_lm_fwd(), but with a tabulated (arbitrary, possibly asymmetric) TOF kernel
  instead of the Gaussian kernel (parameters tof_sigma, num_sigmas and
  is_lor_dependent_tof_sigma are replaced by the table and the kernel scale).

  \pp_note_pointer_types
  \pp_note_tof_kernel_table

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_fwd
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_out{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_kernel_cdf
  \pp_param_tof_kernel_num_samples
  \pp_param_tof_kernel_u_min
  \pp_param_tof_kernel_u_max
  \pp_param_tof_kernel_scale{num_events,is_lor_dependent_tof_kernel_scale}
  \pp_param_tof_dep_center{events}
  \pp_param_tof_dep_kernel_scale{events}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */
  PARALLELPROJ_API void joseph3d_tof_lm_fwd_tab(const float *lor_start,
                                                const float *lor_end,
                                                const float *image,
                                                const float *image_origin,
                                                const float *voxel_size,
                                                float *projection_values,
                                                std::size_t num_events,
                                                const int *image_dim,
                                                float tof_bin_width,
                                                const float *tof_center_offset,
                                                const short *tof_bin_index,
                                                short num_tof_bins,
                                                const float *tof_kernel_cdf,
                                                int tof_kernel_num_samples,
                                                float tof_kernel_u_min,
                                                float tof_kernel_u_max,
                                                const float *tof_kernel_scale,
                                                unsigned char is_lor_dependent_tof_center_offset,
                                                unsigned char is_lor_dependent_tof_kernel_scale,
                                                int device_id = 0,
                                                int threads_per_block = 64,
                                                void *stream = nullptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*!
  @brief TOF listmode back projection (Joseph 3D) with a tabulated TOF kernel.

  Same as joseph3d_tof_lm_back(), but with a tabulated (arbitrary, possibly asymmetric) TOF kernel
  instead of the Gaussian kernel (parameters tof_sigma, num_sigmas and
  is_lor_dependent_tof_sigma are replaced by the table and the kernel scale).

  \pp_note_pointer_types
  \pp_note_tof_kernel_table

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_back
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_back{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_kernel_cdf
  \pp_param_tof_kernel_num_samples
  \pp_param_tof_kernel_u_min
  \pp_param_tof_kernel_u_max
  \pp_param_tof_kernel_scale{num_events,is_lor_dependent_tof_kernel_scale}
  \pp_param_tof_dep_center{events}
  \pp_param_tof_dep_kernel_scale{events}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */
  PARALLELPROJ_API void joseph3d_tof_lm_back_tab(const float *lor_start,
                                                 const float *lor_end,
                                                 float *image,
                                                 const float *image_origin,
                                                 const float *voxel_size,
                                                 const float *projection_values,
                                                 std::size_t num_events,
                                                 const int *image_dim,
                                                 float tof_bin_width,
                                                 const float *tof_center_offset,
                                                 const short *tof_bin_index,
                                                 short num_tof_bins,
                                                 const float *tof_kernel_cdf,
                                                 int tof_kernel_num_samples,
                                                 float tof_kernel_u_min,
                                                 float tof_kernel_u_max,
                                                 const float *tof_kernel_scale,
                                                 unsigned char is_lor_dependent_tof_center_offset,
                                                 unsigned char is_lor_dependent_tof_kernel_scale,
                                                 int device_id = 0,
                                                 int threads_per_block = 64,
                                                 void *stream = nullptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D listmode TOF "point" forward projector

  Fast, approximate TOF listmode forward projector: for every event, the
  image is trilinearly interpolated at a single point, the center of the
  event's TOF bin. No ray tracing and no TOF kernel are involved.

  TOF bin @p i of an event is centered at a distance
  `(i - (num_tof_bins - 1) / 2) * tof_bin_width + tof_center_offset`
  from the LOR midpoint, measured along the LOR towards @p lor_end
  (same bin placement as joseph3d_tof_lm_fwd()).
  Image values outside of the image are treated as zero.

  \pp_note_pointer_types

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_fwd
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_out{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_dep_center{events}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */

  PARALLELPROJ_API void tof_lm_point_fwd(const float *lor_start,
                                         const float *lor_end,
                                         const float *image,
                                         const float *image_origin,
                                         const float *voxel_size,
                                         float *projection_values,
                                         std::size_t num_events,
                                         const int *image_dim,
                                         float tof_bin_width,
                                         const float *tof_center_offset,
                                         const short *tof_bin_index,
                                         short num_tof_bins,
                                         unsigned char is_lor_dependent_tof_center_offset,
                                         int device_id = 0,
                                         int threads_per_block = 64,
                                         void *stream = nullptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /*! @brief 3D listmode TOF "point" back projector (adjoint of tof_lm_point_fwd())

  Adds the value of every event, distributed with trilinear interpolation
  weights, to the (up to 8) voxels around the center of the event's TOF bin.
  Back projecting events with value 1 creates a TOF "histo-image": every
  event whose TOF bin center lies within the box spanned by the centers of
  the outermost voxels adds exactly 1 to the image sum (events closer to the
  image border add less, events further outside add nothing).

  \pp_note_pointer_types

  \pp_param_lor_start{num_events}
  \pp_param_lor_end{num_events}
  \pp_param_image_back
  \pp_note_image_layout
  \pp_param_image_origin
  \pp_param_voxel_size
  \pp_param_projection_values_tof_lm_back{num_events}
  \pp_param_count{num_events,events}
  \pp_param_tof_image_dim
  \pp_param_tof_bin_width
  \pp_param_tof_center_offset{num_events,is_lor_dependent_tof_center_offset}
  \pp_param_tof_bin_index{num_events}
  \pp_param_tof_num_bins
  \pp_param_tof_dep_center{events}
  \pp_param_device_id
  \pp_param_threads
  \pp_param_stream
  */

  PARALLELPROJ_API void tof_lm_point_back(const float *lor_start,
                                          const float *lor_end,
                                          float *image,
                                          const float *image_origin,
                                          const float *voxel_size,
                                          const float *projection_values,
                                          std::size_t num_events,
                                          const int *image_dim,
                                          float tof_bin_width,
                                          const float *tof_center_offset,
                                          const short *tof_bin_index,
                                          short num_tof_bins,
                                          unsigned char is_lor_dependent_tof_center_offset,
                                          int device_id = 0,
                                          int threads_per_block = 64,
                                          void *stream = nullptr);

#ifdef __cplusplus
}
#endif
