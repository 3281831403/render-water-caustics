"""水面光斑（caustics）投影的核心实现。

纯标准库（只允许 ``math``），离线、无随机。

坐标约定
--------
* ``y`` 轴向上；静止水面法线为 ``(0, 1, 0)``。
* 所有方向向量都是长度为 1 的 ``tuple``；``dir`` 表示**光的传播方向**。
* ``refract_dir`` / ``caustic_intensity`` 里的 ``normal`` 指向**入射介质**一侧，
  因此光线射向界面时满足 ``dot(incident, normal) < 0``（水面入射时 normal 向上，
  光向下走）。
* 所有数值断言的容差是 ``1e-9``。
"""

import math

__all__ = [
    "water_normal",
    "refract_dir",
    "project_to_floor",
    "caustic_intensity",
    "blur_radius",
    "accumulate",
    "validate_params",
    "caustics_batch",
]


def water_normal(uv, t, params):
    """计算水面法线（单位向量，y 轴向上）。

    参数
    ----
    uv : (float, float)
        水面平面坐标 ``(u, v)``。
    t : float
        时间（秒）。
    params : dict
        ``params["waves"]`` 是波列表，每个波是
        ``{"dir": (dx, dy), "freq": f, "amp": w, "speed": s, "phase": p}``；
        ``params["amplitude"]`` 是全局高度缩放 ``A``（缺省按 0 处理）。
        ``dir`` 视为单位方向。

    契约
    ----
    高度场::

        h(u, v, t) = A * sum_i w_i * sin(2*pi*f_i*(dx_i*u + dy_i*v) + s_i*t + p_i)

    法线由高度场**梯度**给出::

        n = normalize((-dh/du, 1, -dh/dv))

    必须逐波求偏导后再求和（不能把波的方向/斜率取算术平均，否则法线偏平），
    并且返回值必须归一化为单位向量（否则折射角会错）。
    静止水面（``waves`` 为空或 ``A == 0``）必须返回 ``(0, 1, 0)``。

    容差 1e-9。
    """
    raise NotImplementedError


def refract_dir(incident, normal, eta):
    """按 Snell 定律计算折射方向（单位向量）。

    参数
    ----
    incident : tuple
        入射光传播方向（单位向量，指向界面）。
    normal : tuple
        界面法线（单位向量），指向**入射介质**一侧；
        因此要求 ``dot(incident, normal) < 0``。
    eta : float
        折射率比 ``n_incident / n_transmitted``，必须 ``> 0``。
        空气 -> 水取 ``1/1.33``（约 0.7519），水 -> 空气取 ``1.33``。
        必须按传入值使用，不能写死 ``1/1.33``。

    契约
    ----
    ``c_i = -dot(incident, normal)`` 是入射角余弦。
    若 ``eta**2 * (1 - c_i**2) >= 1`` 则发生**全反射**，返回 ``None``
    （绝不能算出 NaN）。否则 ``c_t = sqrt(1 - eta**2 * (1 - c_i**2))``，
    返回折射方向::

        n_t = normalize(eta * incident + (eta * c_i - c_t) * normal)

    垂直入射（``incident == -normal``）返回 ``incident`` 本身。

    容差 1e-9。
    """
    raise NotImplementedError


def project_to_floor(origin, dir, floor_y):
    """把光线投影到水平地板 ``y = floor_y``。

    参数
    ----
    origin : tuple
        光线起点。
    dir : tuple
        单位传播方向。
    floor_y : float
        地板高度。

    契约
    ----
    参数 ``t = (floor_y - origin[1]) / dir[1]``。
    若 ``dir[1] == 0`` 或 ``t <= 0``（光线平行于地板，或射向背离地板的一侧），
    返回 ``None``；否则返回 ``origin + t * dir``，其 y 分量必须等于 ``floor_y``。
    不能把 ``t`` 取成负值（那会投到地板背面）。

    容差 1e-9。
    """
    raise NotImplementedError


def caustic_intensity(dir, normal, depth):
    """单个采样点处的光斑强度（无量纲）。

    参数
    ----
    dir : tuple
        水面处折射后的传播方向（单位向量，指向水下）。
    normal : tuple
        水面法线（单位向量，y 轴向上）。
    depth : float
        该点到水面的深度（``>= 0``）。

    契约
    ----
    ``c_i = clamp(-dot(dir, normal), 0, 1)``（入射角余弦），
    ``eta = 1/1.33``（空气 -> 水），``sin2_t = eta**2 * (1 - c_i**2)``；
    若 ``sin2_t >= 1`` 返回 ``0.0``。否则 ``c_t = sqrt(1 - sin2_t)``::

        雅可比/会聚因子 = c_i / c_t      # 随入射角变化，垂直入射为 1，不能被当成常数
        深度衰减       = 1 / (1 + depth)**2   # 平方反比，不是 1/depth
        返回值         = 雅可比 * 深度衰减

    容差 1e-9。
    """
    raise NotImplementedError


def blur_radius(depth, roughness):
    """光斑的模糊半径，随深度增大。

    契约
    ----
    ``radius = roughness * depth``（``depth >= 0``，``roughness >= 0``）。
    ``depth == 0`` 或 ``roughness == 0`` 时为 ``0``，即模糊必须依赖深度，
    不能所有深度都用同一个半径。``depth`` 或 ``roughness`` 为负时抛
    ``ValueError``。

    容差 1e-9。
    """
    raise NotImplementedError


def accumulate(uv_hits, size):
    """把命中点累积成 ``size x size`` 的光斑图（行优先，单位是能量）。

    参数
    ----
    uv_hits : list
        采样序列，每个采样是
        ``{"uv": (u, v), "intensity": float, "area": float}``，
        其中 ``u, v`` 属于 ``[0, 1]``，``area`` 是采样覆盖的面积。
    size : int
        网格边长（``>= 1``）。

    契约
    ----
    每个采样携带能量 ``intensity * area``，用**归一化双线性核**泼溅到网格::

        col = u * size;  row = v * size
        c0 = floor(col); r0 = floor(row)      # 越界时钳到最后一行/列
        fx = col - c0;   fy = row - r0
        四个权重 (1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy 之和恒为 1

    重叠采样必须**累加**（不能覆盖），因此全图总能量等于
    ``sum(intensity * area)``。返回 ``list[list[float]]``，长度 ``size``，
    每行长度 ``size``。``uv`` 越界或 ``size < 1`` 抛 ``ValueError``。

    容差 1e-9。
    """
    raise NotImplementedError


def validate_params(params):
    """校验参数字典，返回人类可读的错误消息列表；合法时返回 ``[]``。

    检查项（按下列固定顺序输出，便于测试）
    --------------------------------------
    1. ``params`` 必须是 dict，否则返回 ``["params must be a dict"]``。
    2. ``"eta"``：有限实数且 ``0 < eta <= 3``，否则消息中含 ``"eta"``。
    3. ``"amplitude"``：有限实数且 ``0 <= amplitude <= 0.25``，
       否则消息中含 ``"amplitude"``。
    4. ``"depth"``：有限实数且 ``0 < depth <= 100``，否则消息中含 ``"depth"``。
    5. ``"roughness"``：可选；若给出则必须是有限实数且 ``0 <= roughness <= 1``，
       否则消息中含 ``"roughness"``。

    缺失的键按非法处理；本函数不抛异常。
    """
    raise NotImplementedError


def caustics_batch(frames, config):
    """批量生成多帧光斑图，各帧完全独立。

    参数
    ----
    frames : list
        每帧是 ``{"t": float, "uv_hits": [...]}``，``uv_hits`` 同
        :func:`accumulate`。
    config : dict
        必须含 ``"size"``（网格边长）与 ``"params"``（交给
        :func:`validate_params` 校验）。

    契约
    ----
    先执行 ``errors = validate_params(config["params"])``，非空则
    ``raise ValueError(errors)``。随后对每一帧用**独立的**零缓冲调用
    ``accumulate(frame["uv_hits"], config["size"])``；绝不能在帧之间复用或共享
    累积缓冲（否则第二帧会叠加到第一帧上）。

    返回 ``{"count": len(frames), "frames": [grid_0, grid_1, ...]}``，
    每个 grid 都是独立对象；``frames`` 为空时返回
    ``{"count": 0, "frames": []}``。

    容差 1e-9。
    """
    raise NotImplementedError
