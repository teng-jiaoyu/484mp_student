# Geometry：配置生成与 BEV

## 当前完成范围

- 补齐世界／车体点到像素的投影，维持原 JSON 三字段接口。
- 固定 BEV 为 800×600 像素，对应横向 20 m、前向 15 m，两个方向均为 0.025 m/px。
- 提供课程模板参数和本机 GEM 参数两个明确命名的 profile。
- 独立验证不需要 ROS 节点、模型 checkpoint 或完成 `lane_detect.py`。
- 7 项自动测试通过；本机已有 3 组真实图像／掩码完成 BEV 可视化。
- BEV 阶段之后的 lane fitting、XTE / HE 已完成，见 [geometry_implementation.md](geometry_implementation.md)。

## 坐标、公式与参数来源

本机车体坐标：X 向前、Y 向左、Z 向上。相机光学坐标：X 向右、Y 向下、Z 向前。

投影采用列向量约定：

```text
P_optical = R @ P_base + t
q = K @ P_optical
u = q[0] / q[2]
v = q[1] / q[2]
```

相机安装位置 C 与投影平移 t 不是同一个量；本机相机姿态与车体对齐时：

```text
R = [[0,-1,0], [0,0,-1], [1,0,0]]
C = [0.394, 0, 0.44 + 1.19]
t = -R @ C = [0, 1.63, -0.394]
```

来源是本机 `gem_description/urdf/gem.urdf.xacro` 的 `base_link_joint` 和
`gem_front_camera_joint`，以及 `gem.gazebo` 的 800×600 图像和 1.3962634 rad 水平视场角。
由视场角独立计算的焦距 `800 / (2*tan(fov/2))` 与模板 K 的焦距一致。
这是基于本机模型文件的平地标定，不是本轮重新采集的运行时 CameraInfo/TF；
没有建模车辆俯仰、横滚、悬架变化或地面高差。

本机 BEV 四角使用 `(X,Y)`：`(15,10), (0,10), (0,-10), (15,-10)`，
分别映射到 `(0,0), (0,600), (800,600), (800,0)`。
使用连续边界坐标 800/600，与现有 `perspective_transform()` 完全一致，不改为 799/599。

```text
BEV_u = (10 - base_Y) / 0.025
BEV_v = (15 - base_X) / 0.025
```

因此前方在图像上方、车辆左侧在图像左侧、车体原点在 `(400,600)` 的下边界。
控制点位于画面外是正常的。下边界 X=0 在相机安装位置 X=.394 后方；
这些点仍可作为地面单应矩阵控制点，不能把它们当作可见物体，也不能裁剪其投影。
恰好零光学深度没有有限投影，生成器会拒绝。

## 课程模板与本机配置的区别

默认 `template` 原样保留课程 K、R、t 和角点顺序，按原注释的方向执行 `R @ P + t`。
这不是对课程外参正确性的背书：该解释下，X=15 m 的深度为 -13.454 m，
且模板的相机名称和本机不同。生成器会显式警告。
请在课程环境核对外参／坐标约定后再将这一 profile 作为最终提交标定。

`local-gem` 则使用上面的本机位姿和物理左右角点顺序。不要通过翻转 XTE/HE 符号补偿错误标定。

当前生成文件：

| 文件（相对 `code/src/mp1`） | 用途 |
| --- | --- |
| `data/bev_config.local.json` | 本机离线验证配置 |
| `data/bev_config.template.json` | 未认证的课程模板配置，供对照 |
| `data/bev_config.json` | 当前本机运行配置，内容与 local 版本相同 |
| `data/bev_validation/local-gem/` | 三组 BEV、掩码、视野掩码、对照图和 summary.json |

代码默认参数仍是课程模板；本次通过显式 `--profile local-gem --output data/bev_config.json` 激活本机配置。
直接再次运行无参数生成器并同意覆盖，会把运行配置切回未经验证的模板。

## 复现

依赖可以复用现有本机环境，但始终在本开发仓库运行脚本：

```sh
source /home/jerry/Data/ece484/env.zsh
cd /home/jerry/ECE484/MP1/code/src/mp1

# 分别生成配置；已有文件时不带 --force 会询问是否覆盖。
python scripts/generate_bev_config.py --profile template --output data/bev_config.template.json
python scripts/generate_bev_config.py --profile local-gem

# 本机 lane_detect.py 后续读取的固定路径。
python scripts/generate_bev_config.py --profile local-gem --output data/bev_config.json

# 无 GUI、无模型地生成对照图。
MPLBACKEND=Agg python scripts/verify_bev.py \
  --config data/bev_config.local.json \
  --capture-dir /home/jerry/Data/ece484/verification/data/capture \
  --output-dir data/bev_validation/local-gem

MPLBACKEND=Agg python -m unittest discover -s tests -v
```

`verify_bev.py` 要求 `capture/images/*.png` 和同名 `capture/masks/*.png`，
图像与掩码均为 800×600；接受 0/1 或 0/255 二值掩码。
它拒绝尺寸错误，不会在不调整 K 的情况下偷偷缩放图像。
RGB 使用原有透视变换，掩码使用相同矩阵加最近邻插值，输出保持 0/255。
重复运行会重写指定输出目录中的同名生成文件；请使用专用输出目录。

## 验证结果与限制

自动测试覆盖：模板常量兼容、独立针孔模型的左右/前后方向与尺度、四角对应、
非控制点往返、无穷／零深度、画外控制点、错误 profile、完整命令行合成双车道验证，
以及错误图像分辨率的拒绝行为（共 7 个测试方法）。

合成双车道位于车体 Y=±1.65 m，应出现在 BEV u=334 和 466，即间距 132 px / 3.3 m。
测试允许栅格化后每条线 2 px 的中心偏差。单像素线经最近邻插值可能断开，
因此按窄行带检查，不为通过测试而改变掩码或加粗真实输入。

真实样本 0/1/2 的非控制点往返最大误差均为约 0.000061 px，
掩码输出只有 0/255；BEV 约 55.85% 区域落在相机画面内。
黑色边角主要是未观测区域，不代表道路或车道检测失败，详见 `*_visible.png`。
往返误差衡量数值一致性，不能独立证明物理标定正确。

真实样本主要只有一侧车道，原始预处理掩码还包含天空区域；本次没有改写这些输入。
因此真实双车道宽度、全赛道精度和动态姿态误差仍需后续采集验证。
本机完整运行 `lane_detect.py` 仍依赖队友模型；后续已完成的 XTE/HE 与回调验证见 geometry_implementation.md，真实模型联调尚未验收。

生成的配置和验证图已通过 `.gitignore` 排除，代码、测试与本文档可提交；没有创建提交或推送。
