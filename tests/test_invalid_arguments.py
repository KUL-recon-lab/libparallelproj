import array_api_compat.numpy as np
import parallelproj_core as pp


def test_joseph3d_invalid_arguments():
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
