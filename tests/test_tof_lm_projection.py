import pytest
import math
import random
import array_api_compat
import parallelproj_core as ppb

from types import ModuleType
from .config import pytestmark


def allclose(x, y, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two arrays are close to each other, given absolute and relative error
    inspired by numpy.allclose
    """
    xp = array_api_compat.array_namespace(x)
    return bool(xp.all(xp.less_equal(xp.abs(x - y), atol + rtol * xp.abs(y))))


@pytest.mark.parametrize(
    "direc, sigma_tof, num_tofbins, tof_center_offset",
    [
        (0, 12.0, 29, 0.0),
        (1, 12.0, 29, 0.0),
        (2, 12.0, 29, 0.0),
        (0, 12.0, 29, 0.0),
        (0, 12.0, 28, 0.0),
        (0, 10.0, 29, 0.0),
        (0, 12.0, 29, -2.5),
        (0, 12.0, 29, 2.5),
    ],
)
def test_tof_lm_fwd(
    xp: ModuleType,
    dev: str,
    direc: int,
    sigma_tof: float,
    num_tofbins: int,
    tof_center_offset: float,
    voxsize: tuple[float, float, float] = (1.5, 1.7, 1.6),
    vox_num: int = 29,
    tofbin_width: float = 4.0,
    num_sigmas: float = 4.0,  # we use pm 4 sigma here, since the truncation is slightly different between sino and LM projectors
    nvox: int = 59,
    verbose: bool = False,
):

    img_dim = tuple(nvox if i == direc else 1 for i in range(3))
    xstart = tuple(60.0 if i == direc else 0 for i in range(3))
    xend = tuple(-60.0 if i == direc else 0 for i in range(3))

    ###########################################

    n0, n1, n2 = img_dim

    img_origin = (
        (-(n0 / 2) + 0.5) * voxsize[0],
        (-(n1 / 2) + 0.5) * voxsize[1],
        (-(n2 / 2) + 0.5) * voxsize[2],
    )

    d0 = xend[0] - xstart[0]
    d1 = xend[1] - xstart[1]
    d2 = xend[2] - xstart[2]
    dr_sq = (d0**2, d1**2, d2**2)
    direction = max(range(3), key=lambda i: dr_sq[i])

    img = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    if direction == 0:
        img[vox_num, 0, 0] = 1.0
    elif direction == 1:
        img[0, vox_num, 0] = 1.0
    elif direction == 2:
        img[0, 0, vox_num] = 1.0
    else:
        raise ValueError("direction must be 0, 1, or 2")

    ############################################################################
    # parallelproj-backend based sinogram TOF forward projection

    p_tof_sino = xp.zeros((1, num_tofbins), dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_fwd(
        xp.asarray([xstart], dtype=xp.float32, device=dev),
        xp.asarray([xend], dtype=xp.float32, device=dev),
        img,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        p_tof_sino,
        tofbin_width,
        xp.asarray([sigma_tof], dtype=xp.float32, device=dev),
        xp.asarray([tof_center_offset], dtype=xp.float32, device=dev),
        num_tofbins,
        num_sigmas=num_sigmas,
    )

    p_tof_lm = xp.zeros(num_tofbins, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_fwd(
        xp.tile(xp.asarray([xstart], dtype=xp.float32, device=dev), (num_tofbins, 1)),
        xp.tile(xp.asarray([xend], dtype=xp.float32, device=dev), (num_tofbins, 1)),
        img,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        p_tof_lm,
        tofbin_width,
        xp.tile(xp.asarray([sigma_tof], dtype=xp.float32, device=dev), (num_tofbins,)),
        xp.tile(
            xp.asarray([tof_center_offset], dtype=xp.float32, device=dev),
            (num_tofbins,),
        ),
        xp.arange(num_tofbins, dtype=xp.int16, device=dev),
        num_tofbins,
        num_sigmas=num_sigmas,
    )

    for i in range(num_tofbins):
        # check whether the projection is equal to the expected one
        sino_val = float(p_tof_sino[0, i])
        lm_val = float(p_tof_lm[i])

        assert math.isclose(sino_val, lm_val, abs_tol=1e-4)


@pytest.mark.parametrize(
    "sigma_tof, num_tofbins", [(4.5, 41), (4.5, 40), (8.5, 41), (2.5, 41)]
)
def test_tof_lm_adjointness(
    xp: ModuleType,
    dev: str,
    sigma_tof: float,
    num_tofbins: int,
    voxsize: tuple[float, float, float] = (2.2, 2.5, 2.7),
    tofbin_width: float = 1.5,
    num_sigmas: float = 3.0,
    tof_center_offset: float = 0.0,
    nvox: int = 19,
    verbose: bool = False,
    nlors=200,
):

    # ------
    random.seed(42)
    img_dim = (nvox, nvox, nvox)

    n0, n1, n2 = img_dim
    img_origin = (
        (-(n0 / 2) + 0.5) * voxsize[0],
        (-(n1 / 2) + 0.5) * voxsize[1],
        (-(n2 / 2) + 0.5) * voxsize[2],
    )

    img = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    # fill the image with uniform random values using python's random module
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                img[i, j, k] = random.uniform(0.0, 1.0)

    xstart = xp.zeros((nlors, 3), dtype=xp.float32, device=dev)
    xend = xp.zeros((nlors, 3), dtype=xp.float32, device=dev)

    # fill xstart and xend with random points on a sphere with radius 45
    r = 45.0
    for i in range(nlors):
        theta = random.uniform(0.0, math.pi)
        phi = random.uniform(0.0, 2.0 * math.pi)
        xstart[i, 0] = r * math.sin(theta) * math.cos(phi)
        xstart[i, 1] = r * math.sin(theta) * math.sin(phi)
        xstart[i, 2] = r * math.cos(theta)

        theta = random.uniform(0.0, math.pi)
        phi = random.uniform(0.0, 2.0 * math.pi)
        xend[i, 0] = r * math.sin(theta) * math.cos(phi)
        xend[i, 1] = r * math.sin(theta) * math.sin(phi)
        xend[i, 2] = r * math.cos(theta)

    # simulate LOR-dependent TOF resolution and center offsets
    sigma_tof_array = xp.zeros(nlors, dtype=xp.float32, device=dev)
    for i in range(nlors):
        sigma_tof_array[i] = sigma_tof * random.uniform(0.9, 1.1)

    tof_center_offset_array = xp.zeros(nlors, dtype=xp.float32, device=dev)
    for i in range(nlors):
        tof_center_offset_array[i] = tof_center_offset + random.uniform(-2.0, 2.0)

    tofbin = xp.zeros(nlors, dtype=xp.int16, device=dev)
    for i in range(nlors):
        tofbin[i] = random.randint(0, num_tofbins - 1)

    img_fwd = xp.zeros(nlors, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_fwd(
        xstart,
        xend,
        img,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        img_fwd,
        tofbin_width,
        sigma_tof_array,
        tof_center_offset_array,
        tofbin,
        num_tofbins,
        num_sigmas=num_sigmas,
    )

    # back project a random TOF sinogram
    y = xp.zeros(nlors, dtype=xp.float32, device=dev)
    for i in range(nlors):
        y[i] = random.uniform(0.0, 1.0)

    y_back = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_back(
        xstart,
        xend,
        y_back,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        y,
        tofbin_width,
        sigma_tof_array,
        tof_center_offset_array,
        tofbin,
        num_tofbins,
        num_sigmas=num_sigmas,
    )

    # test the adjointness property
    innerprod1 = float(xp.sum(img_fwd * y))
    innerprod2 = float(xp.sum(img * y_back))

    if verbose:
        print(f"Inner product 1: {innerprod1:.5E}")
        print(f"Inner product 2: {innerprod2:.5E}")

    assert math.isclose(innerprod1, innerprod2, abs_tol=3e-4)


@pytest.mark.parametrize(
    "direction",
    [0, 1, 2],
)
def test_tof_lm_vs_sino_fwd(xp: ModuleType, dev: str, direction: int):

    vx = (2.0, 2.0, 2.0)
    img_shape = (40, 40, 40)

    num_tofbins = 44
    tofbin_width = 3.0
    sigma_tof = 12.0
    num_sigmas = 3.0
    tofcenter_offset = 0

    img_origin = xp.asarray([-39.0, -39.0, -39.0], dtype=xp.float32, device=dev)
    voxel_size = xp.asarray([2.0, 2.0, 2.0], dtype=xp.float32, device=dev)

    if direction == 0:
        iy = 0
        iz = img_shape[2] // 2

        xstart = xp.asarray(
            [
                [
                    -0.6 * vx[0] * img_shape[0],
                    float(img_origin[1] + voxel_size[1] * iy),
                    float(img_origin[2] + voxel_size[2] * iz),
                ]
            ],
            dtype=xp.float32,
            device=dev,
        )
        xend = xp.asarray(
            [
                [
                    0.6 * vx[0] * img_shape[0],
                    float(img_origin[1] + voxel_size[1] * (iy - 1)),
                    float(img_origin[2] + voxel_size[2] * iz),
                ]
            ],
            dtype=xp.float32,
            device=dev,
        )

        x_true = xp.zeros(img_shape, dtype=xp.float32, device=dev)
        x_true[10:-10, iy, iz] = 1.0
    elif direction == 1:
        ix = 0
        iz = img_shape[2] // 2

        xstart = xp.asarray(
            [
                [
                    float(img_origin[0] + voxel_size[0] * ix),
                    -0.6 * vx[1] * img_shape[1],
                    float(img_origin[2] + voxel_size[2] * iz),
                ]
            ],
            dtype=xp.float32,
            device=dev,
        )
        xend = xp.asarray(
            [
                [
                    float(img_origin[0] + voxel_size[0] * (ix - 1)),
                    0.6 * vx[1] * img_shape[1],
                    float(img_origin[2] + voxel_size[2] * iz),
                ]
            ],
            dtype=xp.float32,
            device=dev,
        )

        x_true = xp.zeros(img_shape, dtype=xp.float32, device=dev)
        x_true[ix, 10:-10, iz] = 1.0
    elif direction == 2:
        ix = 0
        iy = img_shape[1] // 2

        xstart = xp.asarray(
            [
                [
                    float(img_origin[0] + voxel_size[0] * ix),
                    float(img_origin[1] + voxel_size[1] * iy),
                    -0.6 * vx[2] * img_shape[2],
                ]
            ],
            dtype=xp.float32,
            device=dev,
        )
        xend = xp.asarray(
            [
                [
                    float(img_origin[0] + voxel_size[0] * (ix - 1)),
                    float(img_origin[1] + voxel_size[1] * iy),
                    0.6 * vx[2] * img_shape[2],
                ]
            ],
            dtype=xp.float32,
            device=dev,
        )

        x_true = xp.zeros(img_shape, dtype=xp.float32, device=dev)
        x_true[ix, iy, 10:-10] = 1.0
    else:
        raise ValueError("direction must be 0, 1, or 2")

    # %%
    # TOF fwd projection in listmode
    x_fwd_lm_tof = xp.zeros(num_tofbins, dtype=xp.float32, device=dev)

    for tb in range(num_tofbins):
        tof_bin = xp.asarray([tb], dtype=xp.int16, device=dev)

        ppb.joseph3d_tof_lm_fwd(
            xstart,
            xend,
            x_true,
            img_origin,
            voxel_size,
            x_fwd_lm_tof[tb : tb + 1],
            tofbin_width,
            xp.asarray(
                [sigma_tof],
                dtype=xp.float32,
                device=dev,
            ),
            xp.asarray(
                [tofcenter_offset],
                dtype=xp.float32,
                device=dev,
            ),
            tof_bin,
            num_tofbins,
            num_sigmas,
        )

    # %%
    # TOF fwd projection in sinogram mode
    x_fwd_sino_tof = xp.zeros(
        (xstart.shape[0], num_tofbins), dtype=xp.float32, device=dev
    )

    ppb.joseph3d_tof_sino_fwd(
        xstart,
        xend,
        x_true,
        img_origin,
        voxel_size,
        x_fwd_sino_tof,
        tofbin_width,
        xp.asarray(
            [sigma_tof],
            dtype=xp.float32,
            device=dev,
        ),
        xp.asarray(
            [tofcenter_offset],
            dtype=xp.float32,
            device=dev,
        ),
        num_tofbins,
        num_sigmas,
    )

    # %%
    # non-TOF fwd projection
    x_fwd_sino = xp.zeros((xstart.shape[0],), dtype=xp.float32, device=dev)

    ppb.joseph3d_fwd(
        xstart,
        xend,
        x_true,
        img_origin,
        voxel_size,
        x_fwd_sino,
    )

    assert allclose(x_fwd_lm_tof, x_fwd_sino_tof, atol=1e-3)
    assert math.isclose(float(x_fwd_sino[0]), float(xp.sum(x_fwd_lm_tof)), abs_tol=1e-3)
    assert math.isclose(
        float(x_fwd_sino[0]), float(xp.sum(x_fwd_sino_tof)), abs_tol=1e-3
    )


def test_tof_lm_degenerate_params(
    xp: ModuleType,
    dev: str,
    voxsize: tuple[float, float, float] = (2.0, 2.0, 2.0),
    nvox: int = 19,
    num_tofbins: int = 27,
    nlors: int = 5,
):
    """Degenerate TOF parameters (non-positive tof_bin_width) must be rejected
    per event: the LM forward writes a defined 0 and the back deposits nothing,
    instead of dividing by zero. Exercises the parameter guard in the LM workers.
    """
    direc = 0
    img_dim = tuple(nvox if i == direc else 1 for i in range(3))
    n0, n1, n2 = img_dim
    img_origin = (
        (-(n0 / 2) + 0.5) * voxsize[0],
        (-(n1 / 2) + 0.5) * voxsize[1],
        (-(n2 / 2) + 0.5) * voxsize[2],
    )

    # LORs crossing the image volume, so the geometry check passes and we reach
    # the TOF-parameter guard
    one_start = [60.0 if i == direc else 0.0 for i in range(3)]
    one_end = [-60.0 if i == direc else 0.0 for i in range(3)]
    xstart = xp.asarray([one_start] * nlors, dtype=xp.float32, device=dev)
    xend = xp.asarray([one_end] * nlors, dtype=xp.float32, device=dev)

    img = xp.ones(img_dim, dtype=xp.float32, device=dev)
    sigma = xp.asarray([5.0], dtype=xp.float32, device=dev)
    offset = xp.asarray([0.0], dtype=xp.float32, device=dev)
    tofbin = xp.zeros(nlors, dtype=xp.int16, device=dev)

    bad_tofbin_width = 0.0  # degenerate -> guard must fire (no divide-by-zero)

    # forward: the -1 sentinel must be overwritten with a defined 0
    img_fwd = xp.asarray([-1.0] * nlors, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_fwd(
        xstart,
        xend,
        img,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        img_fwd,
        bad_tofbin_width,
        sigma,
        offset,
        tofbin,
        num_tofbins,
        num_sigmas=3.0,
    )
    assert bool(xp.all(img_fwd == 0.0))

    # back: no deposit, no NaN
    y = xp.ones((nlors,), dtype=xp.float32, device=dev)
    y_back = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_back(
        xstart,
        xend,
        y_back,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        y,
        bad_tofbin_width,
        sigma,
        offset,
        tofbin,
        num_tofbins,
        num_sigmas=3.0,
    )
    assert bool(xp.all(y_back == 0.0))


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_tof_lm_no_plane_in_tof_window(xp: ModuleType, dev: str, direction: int):
    """Voxels much larger than the TOF bins and a very narrow TOF kernel: many TOF
    bins contain no voxel-center plane, so the sum of the TOF plane weights is 0.
    Such events must not contribute (forward: 0, back: skipped) instead of
    producing NaN, and the projectors must stay adjoint."""
    nvox = 21
    vsize = 4.0
    num_tofbins = 41
    tofbin_width = 1.0
    sigma_tof = 0.05

    img_dim = tuple(nvox if k == direction else 1 for k in range(3))
    img_origin = xp.asarray(
        [-(nvox // 2) * vsize if k == direction else 0.0 for k in range(3)], dtype=xp.float32, device=dev
    )
    voxel_size = xp.asarray([vsize, vsize, vsize], dtype=xp.float32, device=dev)

    start = [[-100.0 if k == direction else 0.0 for k in range(3)] for _ in range(num_tofbins)]
    end = [[100.0 if k == direction else 0.0 for k in range(3)] for _ in range(num_tofbins)]
    xstart = xp.asarray(start, dtype=xp.float32, device=dev)
    xend = xp.asarray(end, dtype=xp.float32, device=dev)
    tofbin = xp.asarray(list(range(num_tofbins)), dtype=xp.int16, device=dev)
    sigma = xp.asarray([sigma_tof], dtype=xp.float32, device=dev)
    offset = xp.asarray([0.0], dtype=xp.float32, device=dev)

    random.seed(3)
    img = xp.reshape(
        xp.asarray([random.uniform(0.5, 1.5) for _ in range(nvox)], dtype=xp.float32, device=dev), img_dim
    )
    y = xp.asarray([random.uniform(0.5, 1.5) for _ in range(num_tofbins)], dtype=xp.float32, device=dev)

    p = xp.zeros(num_tofbins, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_fwd(
        xstart, xend, img, img_origin, voxel_size, p, tofbin_width, sigma, offset, tofbin, num_tofbins, num_sigmas=3.0
    )

    y_back = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_back(
        xstart, xend, y_back, img_origin, voxel_size, y, tofbin_width, sigma, offset, tofbin, num_tofbins, num_sigmas=3.0
    )

    assert bool(xp.all(xp.isfinite(p)))
    assert bool(xp.all(xp.isfinite(y_back)))
    # some TOF bins contain a voxel-center plane and do contribute, others do not
    assert int(xp.sum(xp.astype(p > 0, xp.int32))) > 0
    assert int(xp.sum(xp.astype(p == 0, xp.int32))) > 0

    innerprod1 = float(xp.sum(p * y))
    innerprod2 = float(xp.sum(img * y_back))
    assert math.isclose(innerprod1, innerprod2, rel_tol=1e-5)


@pytest.mark.parametrize("sigma_tof", [12.0, 4.0, 1.0, 0.5, 0.1])
@pytest.mark.parametrize("oblique", [False, True])
def test_tof_lm_vs_sino_narrow_kernels(xp: ModuleType, dev: str, sigma_tof: float, oblique: bool):
    """Listmode and sinogram TOF projections of all TOF bins of a LOR must agree,
    also for TOF kernels that are narrow compared to the TOF bin width
    (the listmode TOF window has to include the bin width)."""
    nvox = 41
    num_tofbins = 21
    tofbin_width = 4.0
    num_sigmas = 3.0
    img_dim = (nvox, 9, 9)
    vox = (1.0, 1.1, 0.9)
    img_origin = xp.asarray(
        [(-(n / 2) + 0.5) * v for n, v in zip(img_dim, vox)], dtype=xp.float32, device=dev
    )
    voxel_size = xp.asarray(vox, dtype=xp.float32, device=dev)

    random.seed(7)
    img = xp.asarray(
        [[[random.uniform(0.0, 1.0) for _ in range(img_dim[2])] for _ in range(img_dim[1])] for _ in range(img_dim[0])],
        dtype=xp.float32,
        device=dev,
    )

    if oblique:
        start, end = [-100.0, -1.3, 0.7], [100.0, 1.9, -1.1]
    else:
        start, end = [-100.0, 0.2, -0.3], [100.0, 0.2, -0.3]
    sigma = xp.asarray([sigma_tof], dtype=xp.float32, device=dev)
    offset = xp.asarray([0.0], dtype=xp.float32, device=dev)

    p_sino = xp.zeros((1, num_tofbins), dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_fwd(
        xp.asarray([start], dtype=xp.float32, device=dev),
        xp.asarray([end], dtype=xp.float32, device=dev),
        img, img_origin, voxel_size, p_sino, tofbin_width, sigma, offset, num_tofbins, num_sigmas=num_sigmas,
    )

    p_lm = xp.zeros(num_tofbins, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_lm_fwd(
        xp.asarray([start] * num_tofbins, dtype=xp.float32, device=dev),
        xp.asarray([end] * num_tofbins, dtype=xp.float32, device=dev),
        img, img_origin, voxel_size, p_lm, tofbin_width, sigma, offset,
        xp.asarray(list(range(num_tofbins)), dtype=xp.int16, device=dev), num_tofbins, num_sigmas=num_sigmas,
    )

    # Both projectors use the same (plane, TOF bin) pairs. A small difference remains
    # for TOF kernels much narrower than the voxels: listmode normalizes the TOF weights
    # of an event over the planes, the sinogram projector those of a plane over the TOF
    # bins, which differ slightly when the kernel is sampled by only a few planes.
    # (Before the listmode window included the TOF bin width, the difference was >= 1e-3
    # for all sigmas here and ~1e-1 for sigma = 0.1.)
    rel_tol = 1e-3 if sigma_tof < min(vox) else 3e-4
    max_sino = float(xp.max(p_sino))
    assert max_sino > 0
    assert float(xp.max(xp.abs(p_lm - p_sino[0, :]))) < rel_tol * max_sino
