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
