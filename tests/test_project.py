"""render-water-caustics 的契约测试：15 条，初始全部为红。

所有断言容差 1e-9；测试只依赖标准库与 pytest。
"""

import math

import pytest

from caustics.project import (
    accumulate,
    blur_radius,
    caustic_intensity,
    caustics_batch,
    project_to_floor,
    refract_dir,
    validate_params,
    water_normal,
)

TOL = 1e-9
UP = (0.0, 1.0, 0.0)
DOWN = (0.0, -1.0, 0.0)


def _length(vec):
    return math.sqrt(sum(c * c for c in vec))


def _normalize(vec):
    n = _length(vec)
    return tuple(c / n for c in vec)


def _assert_grid_close(got, expected, tol=TOL):
    assert len(got) == len(expected)
    for row_got, row_expected in zip(got, expected):
        assert len(row_got) == len(row_expected)
        for g, e in zip(row_got, row_expected):
            assert abs(g - e) <= tol


def _wave_params(amplitude=0.05, waves=None):
    if waves is None:
        waves = [
            {"dir": (1.0, 0.0), "freq": 1.0, "amp": 0.5, "speed": 0.0, "phase": 0.0},
            {"dir": (0.0, 1.0), "freq": 2.0, "amp": 0.25, "speed": 0.0, "phase": 0.0},
        ]
    return {"amplitude": amplitude, "waves": waves}


def _valid_params():
    return {"eta": 1.33, "amplitude": 0.02, "depth": 2.0, "roughness": 0.1}


# ---------------------------------------------------------------------------
# 10 条对应缺陷的测试
# ---------------------------------------------------------------------------


def test_water_normal_uses_height_gradient_not_wave_mean():
    """缺陷 1：波叠加必须取高度场梯度，不能取算术平均（否则法线偏平）。"""
    params = _wave_params()
    u, v = 0.1, 0.2
    height_u = 0.05 * (0.5 * 2.0 * math.pi * 1.0) * math.cos(2.0 * math.pi * u)
    height_v = 0.05 * (0.25 * 2.0 * math.pi * 2.0) * math.cos(2.0 * math.pi * 2.0 * v)
    expected = _normalize((-height_u, 1.0, -height_v))

    got = water_normal((u, v), 0.0, params)

    assert got == pytest.approx(expected, abs=TOL, rel=0)
    assert abs(got[0]) > 1e-3 and abs(got[2]) > 1e-3


def test_water_normal_is_unit_length():
    """缺陷 2：法线必须归一化，否则折射角会算错。"""
    params = _wave_params(
        amplitude=0.18,
        waves=[
            {"dir": (1.0, 0.0), "freq": 3.0, "amp": 0.4, "speed": 1.3, "phase": 0.2},
            {"dir": (0.0, 1.0), "freq": 5.0, "amp": 0.3, "speed": 0.7, "phase": 1.1},
            {"dir": (0.6, 0.8), "freq": 2.0, "amp": 0.3, "speed": 2.1, "phase": 2.0},
        ],
    )

    for uv in ((0.0, 0.0), (0.13, 0.71), (0.5, 0.5), (0.99, 0.01)):
        for t in (0.0, 0.37, 2.5):
            got = water_normal(uv, t, params)
            assert len(got) == 3
            assert _length(got) == pytest.approx(1.0, abs=TOL, rel=0)
            assert got[1] > 0.0


def test_refract_eta_air_to_water_bends_toward_normal():
    """缺陷 3：eta 必须用传入值（空气 -> 水为 1/1.33），不能写反。"""
    assert validate_params(_valid_params()) == []
    bad = dict(_valid_params(), eta=-1.0)
    messages = validate_params(bad)
    assert any("eta" in msg.lower() for msg in messages)

    eta = 1.0 / 1.33
    theta = math.radians(45.0)
    incident = (math.sin(theta), -math.cos(theta), 0.0)

    got = refract_dir(incident, UP, eta)

    theta_t = math.asin(eta * math.sin(theta))
    expected = (math.sin(theta_t), -math.cos(theta_t), 0.0)
    assert got is not None
    assert _length(got) == pytest.approx(1.0, abs=TOL, rel=0)
    assert got == pytest.approx(expected, abs=TOL, rel=0)
    assert theta_t < theta


def test_refract_total_internal_reflection_never_produces_nan():
    """缺陷 4：sin > 1 时必须返回 None，不能拿 NaN 继续算。"""
    eta = 1.33
    critical = math.asin(1.0 / eta)

    for deg in (5.0, 20.0, 40.0, 48.0, 49.0, 60.0, 75.0):
        theta = math.radians(deg)
        incident = (math.sin(theta), math.cos(theta), 0.0)  # 水 -> 空气，向上
        got = refract_dir(incident, DOWN, eta)
        if theta >= critical:
            assert got is None, "总反射未在上临界角之外返回 None: %s deg" % deg
        else:
            assert got is not None, "临界角之内不应发生全反射: %s deg" % deg
            assert all(math.isfinite(c) for c in got)
            assert _length(got) == pytest.approx(1.0, abs=TOL, rel=0)
            assert got[1] > 0.0


def test_project_to_floor_uses_forward_ray_parameter():
    """缺陷 5：t 必须为 (floor_y - origin_y)/dir_y，不能取负值投到背面。"""
    origin = (0.0, 2.0, 0.0)
    direction = _normalize((0.3, -1.0, 0.2))
    floor_y = -1.0

    got = project_to_floor(origin, direction, floor_y)

    t = (floor_y - origin[1]) / direction[1]
    expected = tuple(origin[i] + t * direction[i] for i in range(3))
    assert got is not None
    assert t > 0.0
    assert got == pytest.approx(expected, abs=TOL, rel=0)
    assert got[1] == pytest.approx(floor_y, abs=TOL, rel=0)
    assert project_to_floor((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), -1.0) is None
    assert project_to_floor((0.0, -2.0, 0.0), DOWN, -1.0) is None


def test_intensity_depth_falloff_is_inverse_square():
    """缺陷 6：深度衰减必须是 1/(1+depth)^2，不是 1/depth。"""
    near = caustic_intensity(DOWN, UP, 1.0)
    far = caustic_intensity(DOWN, UP, 4.0)

    assert near == pytest.approx(1.0 / 4.0, abs=TOL, rel=0)
    assert far == pytest.approx(1.0 / 25.0, abs=TOL, rel=0)
    assert near > far
    assert near / far == pytest.approx(6.25, abs=TOL, rel=0)


def test_intensity_includes_jacobian_focusing():
    """缺陷 7：强度必须含光线会聚（雅可比）因子，不能随角度恒定。"""
    depth = 1.0
    normal_hit = caustic_intensity(DOWN, UP, depth)
    theta = math.radians(60.0)
    grazing_dir = (math.sin(theta), -math.cos(theta), 0.0)

    grazing = caustic_intensity(grazing_dir, UP, depth)

    eta = 1.0 / 1.33
    c_i = math.cos(theta)
    c_t = math.sqrt(1.0 - (eta * math.sin(theta)) ** 2)
    expected = (c_i / c_t) / ((1.0 + depth) ** 2)
    assert normal_hit == pytest.approx(1.0 / ((1.0 + depth) ** 2), abs=TOL, rel=0)
    assert grazing == pytest.approx(expected, abs=TOL, rel=0)
    assert grazing < normal_hit
    assert normal_hit - grazing > 1e-3


def test_blur_radius_scales_with_depth():
    """缺陷 8：模糊半径必须随深度变化，不能远近一样糊。"""
    assert blur_radius(0.0, 0.5) == pytest.approx(0.0, abs=TOL, rel=0)
    shallow = blur_radius(2.0, 0.5)
    deep = blur_radius(8.0, 0.5)

    assert deep == pytest.approx(4.0, abs=TOL, rel=0)
    assert deep / shallow == pytest.approx(4.0, abs=TOL, rel=0)
    assert blur_radius(4.0, 0.2) == pytest.approx(0.8, abs=TOL, rel=0)
    assert blur_radius(4.0, 0.0) == pytest.approx(0.0, abs=TOL, rel=0)


def test_accumulate_adds_overlapping_hits():
    """缺陷 9：重叠光斑必须累加，不能后写覆盖前写。"""
    size = 4
    hits = [
        {"uv": (0.5, 0.5), "intensity": 0.5, "area": 1.0},
        {"uv": (0.5, 0.5), "intensity": 0.5, "area": 1.0},
    ]

    grid = accumulate(hits, size)

    assert len(grid) == size
    assert all(len(row) == size for row in grid)
    assert grid[2][2] == pytest.approx(1.0, abs=TOL, rel=0)
    assert sum(sum(row) for row in grid) == pytest.approx(1.0, abs=TOL, rel=0)


def test_batch_does_not_reuse_accumulation_buffer():
    """缺陷 10：每帧必须用独立缓冲，空帧不能残留上一帧的光斑。"""
    size = 4
    config = {"size": size, "params": _valid_params()}
    frame_a = {
        "t": 0.0,
        "uv_hits": [
            {"uv": (0.5, 0.5), "intensity": 0.5, "area": 1.0},
            {"uv": (0.25, 0.25), "intensity": 0.25, "area": 1.0},
        ],
    }
    frame_b = {"t": 0.25, "uv_hits": []}

    result = caustics_batch([frame_a, frame_b], config)

    assert result["count"] == 2
    assert len(result["frames"]) == 2
    assert sum(sum(row) for row in result["frames"][1]) == pytest.approx(0.0, abs=TOL, rel=0)
    single = caustics_batch([frame_a], config)
    _assert_grid_close(result["frames"][0], single["frames"][0])


# ---------------------------------------------------------------------------
# 另外 5 条基础契约测试
# ---------------------------------------------------------------------------


def test_still_water_normal_points_up():
    """静止水面法线必须是 (0, 1, 0)。"""
    for params in (_wave_params(amplitude=0.05, waves=[]), _wave_params(amplitude=0.0)):
        got = water_normal((0.3, 0.7), 1.0, params)
        assert got == pytest.approx((0.0, 1.0, 0.0), abs=TOL, rel=0)


def test_vertical_incidence_refraction_is_unchanged():
    """垂直入射时折射方向不变。"""
    incident = (0.0, -1.0, 0.0)
    for eta in (1.0 / 1.33, 1.0, 1.33):
        got = refract_dir(incident, UP, eta)
        assert got == pytest.approx(incident, abs=TOL, rel=0)


def test_total_internal_reflection_returns_none():
    """水 -> 空气、60 度入射时超过临界角，必须返回 None。"""
    theta = math.radians(60.0)
    incident = (math.sin(theta), math.cos(theta), 0.0)

    assert refract_dir(incident, DOWN, 1.33) is None


def test_caustic_map_energy_is_not_amplified():
    """光斑图总能量与入射能量之比必须 <= 1（不放大）。"""
    size = 8
    hits = []
    for idx, deg in enumerate((0.0, 20.0, 40.0)):
        theta = math.radians(deg)
        direction = _normalize((math.sin(theta), -math.cos(theta), 0.0))
        intensity = caustic_intensity(direction, UP, 1.0 + idx)
        hits.append(
            {
                "uv": ((idx + 0.5) / size, (idx + 0.5) / size),
                "intensity": intensity,
                "area": 1.0,
            }
        )

    grid = accumulate(hits, size)

    total = sum(sum(row) for row in grid)
    incident = float(len(hits))
    assert total == pytest.approx(sum(h["intensity"] for h in hits), abs=TOL, rel=0)
    assert total / incident <= 1.0 + TOL


def test_two_frames_match_separate_runs():
    """两帧一起生成，必须与各自单独生成完全相同。"""
    config = {"size": 4, "params": _valid_params()}
    frames = [
        {
            "t": 0.0,
            "uv_hits": [{"uv": (0.25, 0.75), "intensity": 0.4, "area": 1.0}],
        },
        {
            "t": 0.5,
            "uv_hits": [{"uv": (0.75, 0.25), "intensity": 0.6, "area": 1.0}],
        },
    ]

    together = caustics_batch(frames, config)
    only_first = caustics_batch([frames[0]], config)
    only_second = caustics_batch([frames[1]], config)

    assert together["count"] == 2
    _assert_grid_close(together["frames"][0], only_first["frames"][0])
    _assert_grid_close(together["frames"][1], only_second["frames"][0])
    assert together["frames"][0] is not together["frames"][1]
