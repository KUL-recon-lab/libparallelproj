import array_api_compat.numpy as np
import parallelproj_core as pp


def test_joseph3d_fwd_invalid_arguments():
    """Test that joseph3d_fwd raises ValueError for all invalid argument cases"""

    dev = "cpu"

    # Setup valid inputs that we'll modify for each test case
    img_dim = (2, 3, 4)
    voxsize = np.asarray([4.0, 3.0, 2.0], dtype=np.float32, device=dev)
    img_origin = np.asarray([-4.0, -4.5, -4.0], dtype=np.float32, device=dev)
    img = np.ones(img_dim, dtype=np.float32, device=dev)

    xstart = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend = np.asarray([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32, device=dev)
    img_fwd = np.zeros(xstart.shape[:-1], dtype=np.float32, device=dev)

    # Test 1: lor_start must have at least 2 dims and shape (..., 3)
    # Test with 1D array
    xstart_1d = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart_1d, xend, img, img_origin, voxsize, img_fwd)
        assert False, "Expected ValueError for 1D lor_start"
    except ValueError:
        pass

    # Test with wrong last dimension
    xstart_wrong_last_dim = np.asarray([[1.0, 2.0]], dtype=np.float32, device=dev)
    xend_wrong = np.asarray([[7.0, 8.0]], dtype=np.float32, device=dev)
    img_fwd_scalar = np.zeros((1,), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart_wrong_last_dim, xend_wrong, img, img_origin, voxsize, img_fwd_scalar)
        assert False, "Expected ValueError for lor_start with last dim != 3"
    except ValueError:
        pass

    # Test 2: lor_start and lor_end must have the same number of dimensions
    xstart_2d = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32, device=dev)
    xend_3d = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    img_fwd_1d = np.zeros((1,), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart_2d, xend_3d, img, img_origin, voxsize, img_fwd_1d)
        assert False, "Expected ValueError for different ndim"
    except ValueError:
        pass

    # Test 3: lor_start and lor_end must have the same shape
    xstart_shape1 = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend_shape2 = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart_shape1, xend_shape2, img, img_origin, voxsize, img_fwd)
        assert False, "Expected ValueError for different shapes"
    except ValueError:
        pass

    # Test 4: projection_values must have shape equal to lor_start.shape[:-1] (ndim check)
    img_fwd_wrong_ndim = np.zeros((1, 2, 3), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd_wrong_ndim)
        assert False, "Expected ValueError for wrong projection_values ndim"
    except ValueError:
        pass

    # Test 5: projection_values must have shape equal to lor_start.shape[:-1] (shape values check)
    img_fwd_wrong_shape = np.zeros((1, 3), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd_wrong_shape)
        assert False, "Expected ValueError for wrong projection_values shape"
    except ValueError:
        pass

    # Test 6 & 7: Device type/ID mismatches - only test on CPU with numpy
    # (mixing device types is typically not possible in a single test with array_api)
    # These tests are implicitly covered by the array_api requirements

    # Test 8: image_origin must be a 1D array with 3 elements
    img_origin_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart, xend, img, img_origin_wrong_len, voxsize, img_fwd)
        assert False, "Expected TypeError for image_origin with wrong length"
    except TypeError:
        pass

    ## Test 9: voxel_size must be a 1D array with 3 elements
    voxsize_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_fwd(xstart, xend, img, img_origin, voxsize_wrong_len, img_fwd)
        assert False, "Expected TypeError for voxel_size with wrong length"
    except TypeError:
        pass


def test_joseph3d_back_invalid_arguments():
    """Test that joseph3d_back raises for invalid argument cases"""

    dev = "cpu"

    # Setup valid inputs that we'll modify for each test case
    img_dim = (2, 3, 4)
    voxsize = np.asarray([4.0, 3.0, 2.0], dtype=np.float32, device=dev)
    img_origin = np.asarray([-4.0, -4.5, -4.0], dtype=np.float32, device=dev)
    img = np.zeros(img_dim, dtype=np.float32, device=dev)

    xstart = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend = np.asarray([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32, device=dev)
    proj = np.zeros(xstart.shape[:-1], dtype=np.float32, device=dev)

    # Test 1: lor_start must have at least 2 dims and shape (..., 3)
    xstart_1d = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart_1d, xend, img, img_origin, voxsize, proj)
        assert False, "Expected ValueError for 1D lor_start"
    except ValueError:
        pass

    # Test with wrong last dimension
    xstart_wrong_last_dim = np.asarray([[1.0, 2.0]], dtype=np.float32, device=dev)
    xend_wrong = np.asarray([[7.0, 8.0]], dtype=np.float32, device=dev)
    proj_scalar = np.zeros((1,), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart_wrong_last_dim, xend_wrong, img, img_origin, voxsize, proj_scalar)
        assert False, "Expected ValueError for lor_start with last dim != 3"
    except ValueError:
        pass

    # Test 2: lor_start and lor_end must have the same number of dimensions
    xstart_2d = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32, device=dev)
    xend_3d = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    proj_1d = np.zeros((1,), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart_2d, xend_3d, img, img_origin, voxsize, proj_1d)
        assert False, "Expected ValueError for different ndim"
    except ValueError:
        pass

    # Test 3: lor_start and lor_end must have the same shape
    xstart_shape1 = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend_shape2 = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart_shape1, xend_shape2, img, img_origin, voxsize, proj)
        assert False, "Expected ValueError for different shapes"
    except ValueError:
        pass

    # Test 4: projection_values must have shape equal to lor_start.shape[:-1] (ndim check)
    proj_wrong_ndim = np.zeros((1, 2, 3), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart, xend, img, img_origin, voxsize, proj_wrong_ndim)
        assert False, "Expected ValueError for wrong projection_values ndim"
    except ValueError:
        pass

    # Test 5: projection_values must have shape equal to lor_start.shape[:-1] (shape values check)
    proj_wrong_shape = np.zeros((1, 3), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart, xend, img, img_origin, voxsize, proj_wrong_shape)
        assert False, "Expected ValueError for wrong projection_values shape"
    except ValueError:
        pass

    # Test 6 & 7: Device type/ID mismatches are not tested here (numpy CPU only)

    # Test 8: image_origin must match ConstFloat1D3ELArray shape<3>
    img_origin_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart, xend, img, img_origin_wrong_len, voxsize, proj)
        assert False, "Expected TypeError for image_origin with wrong length"
    except TypeError:
        pass

    # Test 9: voxel_size must match ConstFloat1D3ELArray shape<3>
    voxsize_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_back(xstart, xend, img, img_origin, voxsize_wrong_len, proj)
        assert False, "Expected TypeError for voxel_size with wrong length"
    except TypeError:
        pass


def test_joseph3d_tof_sino_fwd_invalid_arguments():
    """Test that joseph3d_tof_sino_fwd raises for invalid argument cases"""

    dev = "cpu"

    img_dim = (2, 3, 4)
    voxsize = np.asarray([4.0, 3.0, 2.0], dtype=np.float32, device=dev)
    img_origin = np.asarray([-4.0, -4.5, -4.0], dtype=np.float32, device=dev)
    img = np.ones(img_dim, dtype=np.float32, device=dev)

    xstart = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend = np.asarray([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32, device=dev)

    num_tof_bins = 5
    tof_bin_width = 2.0
    tof_sigma = np.asarray([30.0], dtype=np.float32, device=dev)
    tof_center_offset = np.asarray([0.0], dtype=np.float32, device=dev)
    proj = np.zeros(xstart.shape[:-1] + (num_tof_bins,), dtype=np.float32, device=dev)

    # Test 1: lor_start must have at least 2 dims and shape (..., 3)
    xstart_1d = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart_1d, xend, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for 1D lor_start"
    except ValueError:
        pass

    xstart_wrong_last_dim = np.asarray([[1.0, 2.0]], dtype=np.float32, device=dev)
    xend_wrong = np.asarray([[7.0, 8.0]], dtype=np.float32, device=dev)
    proj_wrong_lor = np.zeros((1, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart_wrong_last_dim, xend_wrong, img, img_origin, voxsize, proj_wrong_lor, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for lor_start with last dim != 3"
    except ValueError:
        pass

    # Test 2: lor_start and lor_end must have same ndim
    xstart_2d = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32, device=dev)
    xend_3d = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    proj_2d = np.zeros((1, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart_2d, xend_3d, img, img_origin, voxsize, proj_2d, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for different ndim"
    except ValueError:
        pass

    # Test 3: lor_start and lor_end must have same shape
    xstart_shape1 = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend_shape2 = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart_shape1, xend_shape2, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for different shapes"
    except ValueError:
        pass

    # Test 4: projection_values must have same ndim as lor_start
    proj_wrong_ndim = np.zeros((1, 2, 3, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart, xend, img, img_origin, voxsize, proj_wrong_ndim, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for wrong projection_values ndim"
    except ValueError:
        pass

    # Test 5: projection_values[:-1] must match lor_start[:-1]
    proj_wrong_shape = np.zeros((1, 3, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart, xend, img, img_origin, voxsize, proj_wrong_shape, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for wrong projection_values shape"
    except ValueError:
        pass

    # Test 6: projection_values last dim must equal num_tof_bins
    proj_wrong_bins = np.zeros((1, 2, num_tof_bins + 1), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart, xend, img, img_origin, voxsize, proj_wrong_bins, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for wrong num_tof_bins in projection_values"
    except ValueError:
        pass

    # Test 7 & 8: Device type/ID mismatches are not tested here (numpy CPU only)

    # Test 9: tof_sigma shape must be scalar [1] or lor-dependent shape lor_start[:-1]
    tof_sigma_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart, xend, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma_wrong, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_sigma shape"
    except ValueError:
        pass

    # Test 10: tof_center_offset shape must be scalar [1] or lor-dependent shape lor_start[:-1]
    tof_center_offset_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart, xend, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset_wrong, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_center_offset shape"
    except ValueError:
        pass

    # Test 11: image_origin must match ConstFloat1D3ELArray shape<3>
    img_origin_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart, xend, img, img_origin_wrong_len, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected TypeError for image_origin with wrong length"
    except TypeError:
        pass

    # Test 12: voxel_size must match ConstFloat1D3ELArray shape<3>
    voxsize_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_fwd(xstart, xend, img, img_origin, voxsize_wrong_len, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected TypeError for voxel_size with wrong length"
    except TypeError:
        pass


def test_joseph3d_tof_sino_back_invalid_arguments():
    """Test that joseph3d_tof_sino_back raises for invalid argument cases"""

    dev = "cpu"

    img_dim = (2, 3, 4)
    voxsize = np.asarray([4.0, 3.0, 2.0], dtype=np.float32, device=dev)
    img_origin = np.asarray([-4.0, -4.5, -4.0], dtype=np.float32, device=dev)
    img = np.zeros(img_dim, dtype=np.float32, device=dev)

    xstart = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend = np.asarray([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32, device=dev)

    num_tof_bins = 5
    tof_bin_width = 2.0
    tof_sigma = np.asarray([30.0], dtype=np.float32, device=dev)
    tof_center_offset = np.asarray([0.0], dtype=np.float32, device=dev)
    proj = np.zeros(xstart.shape[:-1] + (num_tof_bins,), dtype=np.float32, device=dev)

    # Test 1: lor_start must have at least 2 dims and shape (..., 3)
    xstart_1d = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart_1d, xend, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for 1D lor_start"
    except ValueError:
        pass

    xstart_wrong_last_dim = np.asarray([[1.0, 2.0]], dtype=np.float32, device=dev)
    xend_wrong = np.asarray([[7.0, 8.0]], dtype=np.float32, device=dev)
    proj_wrong_lor = np.zeros((1, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart_wrong_last_dim, xend_wrong, img, img_origin, voxsize, proj_wrong_lor, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for lor_start with last dim != 3"
    except ValueError:
        pass

    # Test 2: lor_start and lor_end must have same ndim
    xstart_2d = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32, device=dev)
    xend_3d = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    proj_2d = np.zeros((1, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart_2d, xend_3d, img, img_origin, voxsize, proj_2d, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for different ndim"
    except ValueError:
        pass

    # Test 3: lor_start and lor_end must have same shape
    xstart_shape1 = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32, device=dev)
    xend_shape2 = np.asarray([[[7.0, 8.0, 9.0]]], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart_shape1, xend_shape2, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for different shapes"
    except ValueError:
        pass

    # Test 4: projection_values must have same ndim as lor_start
    proj_wrong_ndim = np.zeros((1, 2, 3, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart, xend, img, img_origin, voxsize, proj_wrong_ndim, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for wrong projection_values ndim"
    except ValueError:
        pass

    # Test 5: projection_values[:-1] must match lor_start[:-1]
    proj_wrong_shape = np.zeros((1, 3, num_tof_bins), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart, xend, img, img_origin, voxsize, proj_wrong_shape, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for wrong projection_values shape"
    except ValueError:
        pass

    # Test 6: projection_values last dim must equal num_tof_bins
    proj_wrong_bins = np.zeros((1, 2, num_tof_bins + 1), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart, xend, img, img_origin, voxsize, proj_wrong_bins, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for wrong num_tof_bins in projection_values"
    except ValueError:
        pass

    # Test 7 & 8: Device type/ID mismatches are not tested here (numpy CPU only)

    # Test 9: tof_sigma shape must be scalar [1] or lor-dependent shape lor_start[:-1]
    tof_sigma_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart, xend, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma_wrong, tof_center_offset, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_sigma shape"
    except ValueError:
        pass

    # Test 10: tof_center_offset shape must be scalar [1] or lor-dependent shape lor_start[:-1]
    tof_center_offset_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart, xend, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset_wrong, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_center_offset shape"
    except ValueError:
        pass

    # Test 11: image_origin must match ConstFloat1D3ELArray shape<3>
    img_origin_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart, xend, img, img_origin_wrong_len, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected TypeError for image_origin with wrong length"
    except TypeError:
        pass

    # Test 12: voxel_size must match ConstFloat1D3ELArray shape<3>
    voxsize_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_sino_back(xstart, xend, img, img_origin, voxsize_wrong_len, proj, tof_bin_width, tof_sigma, tof_center_offset, num_tof_bins)
        assert False, "Expected TypeError for voxel_size with wrong length"
    except TypeError:
        pass


def test_joseph3d_tof_lm_fwd_invalid_arguments():
    """Test that joseph3d_tof_lm_fwd raises for invalid argument cases"""

    dev = "cpu"

    img_dim = (2, 3, 4)
    voxsize = np.asarray([4.0, 3.0, 2.0], dtype=np.float32, device=dev)
    img_origin = np.asarray([-4.0, -4.5, -4.0], dtype=np.float32, device=dev)
    img = np.ones(img_dim, dtype=np.float32, device=dev)

    event_start = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32, device=dev)
    event_end = np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32, device=dev)
    num_events = event_start.shape[0]

    num_tof_bins = 5
    tof_bin_width = 2.0
    tof_sigma = np.asarray([30.0], dtype=np.float32, device=dev)
    tof_center_offset = np.asarray([0.0], dtype=np.float32, device=dev)
    proj = np.zeros((num_events,), dtype=np.float32, device=dev)
    tof_bin_index = np.zeros((num_events,), dtype=np.int16, device=dev)

    # Test 1: event_start/event_end must be 2D and shape (..., 3) with same number of events
    event_start_1d = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start_1d, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for event_start ndim != 2"
    except ValueError:
        pass

    event_start_wrong_shape = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, device=dev)
    event_end_wrong_shape = np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start_wrong_shape, event_end_wrong_shape, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for event_start/event_end last dim != 3"
    except ValueError:
        pass

    event_end_other_nevents = np.asarray([[7.0, 8.0, 9.0]], dtype=np.float32, device=dev)
    proj_1event = np.zeros((1,), dtype=np.float32, device=dev)
    tof_bin_index_1event = np.zeros((1,), dtype=np.int16, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end_other_nevents, img, img_origin, voxsize, proj_1event, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index_1event, num_tof_bins)
        assert False, "Expected ValueError for different number of events"
    except ValueError:
        pass

    # Test 2: projection_values and tof_bin_index must be 1D and length == num_events
    proj_wrong_ndim = np.zeros((num_events, 1), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin, voxsize, proj_wrong_ndim, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for projection_values.ndim != 1"
    except ValueError:
        pass

    proj_wrong_len = np.zeros((num_events + 1,), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin, voxsize, proj_wrong_len, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for projection_values length mismatch"
    except ValueError:
        pass

    tof_bin_index_wrong_ndim = np.zeros((num_events, 1), dtype=np.int16, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index_wrong_ndim, num_tof_bins)
        assert False, "Expected ValueError for tof_bin_index.ndim != 1"
    except ValueError:
        pass

    tof_bin_index_wrong_len = np.zeros((num_events + 1,), dtype=np.int16, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index_wrong_len, num_tof_bins)
        assert False, "Expected ValueError for tof_bin_index length mismatch"
    except ValueError:
        pass

    # Test 3 & 4: Device type/ID mismatches are not tested here (numpy CPU only)

    # Test 5: tof_sigma shape must be [1] or [num_events]
    tof_sigma_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma_wrong, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_sigma shape"
    except ValueError:
        pass

    # Test 6: tof_center_offset shape must be [1] or [num_events]
    tof_center_offset_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset_wrong, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_center_offset shape"
    except ValueError:
        pass

    # Test 7: image_origin must match ConstFloat1D3ELArray shape<3>
    img_origin_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin_wrong_len, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected TypeError for image_origin with wrong length"
    except TypeError:
        pass

    # Test 8: voxel_size must match ConstFloat1D3ELArray shape<3>
    voxsize_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_fwd(event_start, event_end, img, img_origin, voxsize_wrong_len, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected TypeError for voxel_size with wrong length"
    except TypeError:
        pass


def test_joseph3d_tof_lm_back_invalid_arguments():
    """Test that joseph3d_tof_lm_back raises for invalid argument cases"""

    dev = "cpu"

    img_dim = (2, 3, 4)
    voxsize = np.asarray([4.0, 3.0, 2.0], dtype=np.float32, device=dev)
    img_origin = np.asarray([-4.0, -4.5, -4.0], dtype=np.float32, device=dev)
    img = np.zeros(img_dim, dtype=np.float32, device=dev)

    event_start = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32, device=dev)
    event_end = np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32, device=dev)
    num_events = event_start.shape[0]

    num_tof_bins = 5
    tof_bin_width = 2.0
    tof_sigma = np.asarray([30.0], dtype=np.float32, device=dev)
    tof_center_offset = np.asarray([0.0], dtype=np.float32, device=dev)
    proj = np.zeros((num_events,), dtype=np.float32, device=dev)
    tof_bin_index = np.zeros((num_events,), dtype=np.int16, device=dev)

    # Test 1: event_start/event_end must be 2D and shape (..., 3) with same number of events
    event_start_1d = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start_1d, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for event_start ndim != 2"
    except ValueError:
        pass

    event_start_wrong_shape = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, device=dev)
    event_end_wrong_shape = np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start_wrong_shape, event_end_wrong_shape, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for event_start/event_end last dim != 3"
    except ValueError:
        pass

    event_end_other_nevents = np.asarray([[7.0, 8.0, 9.0]], dtype=np.float32, device=dev)
    proj_1event = np.zeros((1,), dtype=np.float32, device=dev)
    tof_bin_index_1event = np.zeros((1,), dtype=np.int16, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end_other_nevents, img, img_origin, voxsize, proj_1event, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index_1event, num_tof_bins)
        assert False, "Expected ValueError for different number of events"
    except ValueError:
        pass

    # Test 2: projection_values and tof_bin_index must be 1D and length == num_events
    proj_wrong_ndim = np.zeros((num_events, 1), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin, voxsize, proj_wrong_ndim, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for projection_values.ndim != 1"
    except ValueError:
        pass

    proj_wrong_len = np.zeros((num_events + 1,), dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin, voxsize, proj_wrong_len, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for projection_values length mismatch"
    except ValueError:
        pass

    tof_bin_index_wrong_ndim = np.zeros((num_events, 1), dtype=np.int16, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index_wrong_ndim, num_tof_bins)
        assert False, "Expected ValueError for tof_bin_index.ndim != 1"
    except ValueError:
        pass

    tof_bin_index_wrong_len = np.zeros((num_events + 1,), dtype=np.int16, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index_wrong_len, num_tof_bins)
        assert False, "Expected ValueError for tof_bin_index length mismatch"
    except ValueError:
        pass

    # Test 3 & 4: Device type/ID mismatches are not tested here (numpy CPU only)

    # Test 5: tof_sigma shape must be [1] or [num_events]
    tof_sigma_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma_wrong, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_sigma shape"
    except ValueError:
        pass

    # Test 6: tof_center_offset shape must be [1] or [num_events]
    tof_center_offset_wrong = np.asarray([1.0, 2.0, 3.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset_wrong, tof_bin_index, num_tof_bins)
        assert False, "Expected ValueError for invalid tof_center_offset shape"
    except ValueError:
        pass

    # Test 7: image_origin must match ConstFloat1D3ELArray shape<3>
    img_origin_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin_wrong_len, voxsize, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected TypeError for image_origin with wrong length"
    except TypeError:
        pass

    # Test 8: voxel_size must match ConstFloat1D3ELArray shape<3>
    voxsize_wrong_len = np.asarray([1.0, 2.0], dtype=np.float32, device=dev)
    try:
        pp.joseph3d_tof_lm_back(event_start, event_end, img, img_origin, voxsize_wrong_len, proj, tof_bin_width, tof_sigma, tof_center_offset, tof_bin_index, num_tof_bins)
        assert False, "Expected TypeError for voxel_size with wrong length"
    except TypeError:
        pass
