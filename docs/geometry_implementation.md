# Geometry：lane fitting、XTE / HE

本轮完成 `lane_detect.py` 的几何接入、滑窗拟合修正、带符号 XTE 与 HE、可视化和独立验证。
BEV 参数来源及课程模板差异见 [bev_implementation.md](bev_implementation.md)。

## 数据流与接口

```text
800×600 BGR image + binary mask
  -> 最近邻 BEV 掩码
  -> 两条独立车道边界的二次拟合
  -> 中心线系数 = (left_fit + right_fit) / 2
  -> 米制最近点 -> XTE / HE
  -> 原图叠加 + 带数值标注的 BEV
```

纯几何放在 `code/src/mp1/scripts/lane_geometry.py`，由实时节点和离线验证共用。
`LaneVisualizer.compute_error(poly_px)` 仍返回 `(XTE, HE, reference_px, closest_px)`；
`fit_poly_lanes(raw_img, binary_img)` 仍返回 `(overlay, binary_BEV, ret)`。
有效输入但没有可信双车道时，`overlay` 和 `ret` 为 `None`；输入尺寸或格式错误会明确报错。
实时回调捕获这类帧错误，记录原因并显示 N/A，不复用上一帧误差。

掩码接受 0/1 或 0/255，统一归一化一次。模型输入的 640×384 分辨率不能直接送入几何模块，
必须先恢复到标定使用的 800×600。变换掩码使用最近邻插值，避免产生灰度标签。
`perspective_transform()` 新增可选 `interpolation` 参数，默认仍为原来的 RGB 线性插值，旧调用兼容。

## XTE 与 HE 的坐标和符号

BEV 横轴 u 向右，纵轴 v 向下；车体前方对应图像上方，车体左侧对应图像左侧。
参考点仍为骨架的下边界中心 `(400,600)`，对应车体地面原点，不额外加相机安装位置。
中心线是 `u = A*v² + B*v + C`，默认 `Sx=Sy=.025 m/px`。

计算时先把多项式和参考点转为米：

```text
P_m(y) = (Sx*A/Sy²)*y² + (Sx*B/Sy)*y + Sx*C
reference_m = [bev_width_m/2, bev_height_m]
```

最近点满足距离导数为零：

```text
(P_m(y) - reference_x) * P_m'(y) + (y - reference_y) = 0
```

求全部数值实根，再选距离最小者；允许曲线外推，不把最近点强行夹在图像行范围内。
在米制空间求解还能正确处理 `Sx != Sy`，避免先找像素最近点再缩放带来的误差。

```text
XTE = sign(reference_x - closest_x) * distance_in_metres
HE = atan(P_m'(closest_y))
```

XTE 为车辆在中心线右侧时正、左侧时负；HE 为路径相对车辆向左转时正、向右转时负。
返回 HE 的单位是弧度，显示时才转为度。与当前界面的 `-WorldGT.XTE`、`WorldGT.HE` 对齐，
已直接用 WorldGT 构造场景核对，包括车辆 yaw 接近 ±π 时的角度环绕。

| 解析中心线 | XTE | HE |
| --- | --- | --- |
| `u=400` | 0 m | 0° |
| `u=360` | +1 m | 0° |
| `u=440` | -1 m | 0° |
| `u=.1*v+340` | 0 m | +5.7106° |
| `u=-.1*v+460` | 0 m | -5.7106° |

## 车道拟合修正

原算法在三组真实单边车道样本上都产生了虚假的左右拟合，还会把位于图像中心的单条粗线拆成两条。
两条车道同时偏在画面一侧时，原来的左右半幅直方图搜索又会失败。

本轮保留二次曲线与滑窗方法，修正为：

- 从最低的可见双峰行带选起点，允许两条边界同在画面一侧；优先选择中心最接近车辆的相邻边界。
- 每个像素只属于一侧、一个行带；空行带保留动量搜索，支持遮挡间隙。
- 迭代排除孤立残差点后拟合；每条线至少 50 个有效像素、3 个不同 y 值，纵向覆盖至少图高的 15%。
- 两条线的共同纵向覆盖至少 15%；在整个 BEV 及参考行检查不交叉。
- 默认 `margin=50` 时，拟合宽度至少 50 px、最多图像宽度，最大/最小宽度比不超过 3。
- 不推测缺失的单边车道，不引入跨帧平滑；不能确认一对车道就返回 None。

这些检查适用于当前标定下的普通车道和适度弯道。它们不是全场景车道识别器；
极端转向、严重遮挡、错误模型掩码仍可能使检测返回 N/A。宽度约束是检测有效性检查，
不意味着系统已经通过真实车道宽度标定。

原图叠加画布改为跟随实际图像尺寸，移除 1280×720 的遗留尺寸；补齐 `out_img` 供已有调试可视化使用。
BEV 显示蓝色左边界、红色右边界、黄色中心线、绿色最近点连线和紫色参考点，并直接标注 XTE / HE。

## 验证与复现

在本机已配置的 ROS/Python 环境中运行：

```sh
source /home/jerry/Data/ece484/env.zsh
cd /home/jerry/ECE484/MP1/code/src/mp1
MPLBACKEND=Agg python -m unittest discover -s tests -v

MPLBACKEND=Agg python scripts/verify_bev.py \
  --config data/bev_config.local.json \
  --capture-dir /home/jerry/Data/ece484/verification/data/capture \
  --output-dir data/bev_validation/geometry_real \
  --fit-lanes
```

22 项测试全部通过（7 项前阶段 BEV、12 项几何、3 项 ROS 方法／回调），没有跳过。
覆盖解析符号、弯道全局最近点、非等比例尺度、左右偏移、左右转弯、断线噪声、单线/交叉/短线拒绝、
原图掩码到误差的完整几何链路，以及有效帧后接失败帧时不残留旧误差。
ROS 测试使用真实 Image 消息和 CvBridge、实际回调及几何逻辑，仅替代模型输出并关闭 GUI。
这些测试不是训练模型或实时 Gazebo 驾驶的验收。

三组旧真实样本全部正确拒绝单边车道，JSON 中 `detected=false`，误差字段为 null，图上显示 N/A。
四组明确标注的合成针孔相机双车道也通过同一离线脚本处理，结果见：

| 合成场景 | 估计 XTE | 估计 HE |
| --- | --- | --- |
| 居中直道 | 约 0 m | 约 0° |
| 车辆在中心线右侧 1 m | +1.0009 m | -0.0100° |
| 左转方向，理论 +5.7106° | -0.0034 m | +5.7560° |
| 二次曲线 | -0.6595 m | +9.9938° |

合成结果包含图像栅格化误差，不能作为真实驾驶性能数据。
输出位于 `data/bev_validation/geometry_synthetic/results/`，输入和系数记录位于其相邻的 `capture/`。
可用相同脚本将 `--capture-dir` 指向该目录来复现；所有输出均受已有忽略规则保护，不混入源码提交。

## 与队友的联调入口

当前仓库的 `model_utils.py` 仍有 `data/FILL_THIS_OUT` 和未完成的 `inference()`，
`simple_enet.py` 的 InitialBlock 也仍是占位实现。因此本轮没有声称已运行完整的训练模型实时演示。
模型队友完成网络、加载和推理后，可运行：

```sh
source /home/jerry/Data/ece484/env.zsh
cd /home/jerry/ECE484/MP1/code/src/mp1
python scripts/lane_detect.py --ros-args \
  -p bev_config:=data/bev_config.local.json

# 无显示窗口的日志运行方式：
python scripts/lane_detect.py --ros-args \
  -p bev_config:=data/bev_config.local.json -p show_windows:=false
```

`bev_config` 默认仍读 `data/bev_config.json`，`show_windows` 默认 true。
摄像头和里程计话题仍为 `/camera/image_raw`、`/odom`；没有里程计时 GT 显示 N/A，不影响图像几何。
后续还需真实双车道、转弯场景与模型 checkpoint 的联调验收；课程外参待确认的问题沿用 BEV 阶段记录。
本轮没有修改 Model/System 模块，没有创建提交或推送。
