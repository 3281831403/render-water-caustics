# render-water-caustics

## 一、题面

泳池底部的光斑会跟着水面乱跳、浅水区还莫名发亮：水面法线用波叠加时取了算术平均而不是高度场梯度，法线偏平；法线算完没有归一化，折射角跟着错；`refract_dir` 的 eta 写死成 1/1.33，空气→水的方向反了；折射不做全反射判断，sin>1 时继续开方算出 NaN；`project_to_floor` 的 t 取成负值，光线被投到地板背面；`caustic_intensity` 的深度衰减用 1/d 而不是 1/d²；强度完全不管光线会聚（雅可比），整片光斑一样均匀；`blur_radius` 与深度无关，远近一样糊；`accumulate` 用覆盖而不是累加，重叠光斑不增强；`caustics_batch` 复用累积缓冲，第二帧叠加到第一帧上。caustics/project.py 是核心，tests/test_project.py 15个红。要修的是：让 `caustics/project.py` 里 8 个函数按 docstring 契约工作，重点盯水面法线、折射方向、能量守恒与深度衰减，所有数值断言容差 1e-9。

## 二、运行方式

```bash
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，且没有 collection error、import error 或 syntax error（15 条测试都会调用尚未实现的桩，因此失败而非报错）。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 四、约束

- 只用 Python 标准库（`math`）与 pytest；离线、无随机数、无外部资源。
- `tests/test_project.py` **不可修改**；15 条测试即验收标准，不得增删、跳过或弱化断言。
- 8 个函数全部以 `raise NotImplementedError` 起步，docstring 即契约；坐标系、法线朝向、eta 定义、深度衰减公式与 1e-9 容差都写在 docstring 里。
- 失败必须是断言失败/`NotImplementedError`，不允许靠 import 错误或语法错误制造红。
