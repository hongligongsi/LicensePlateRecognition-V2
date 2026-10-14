# 检测识别引擎（`yolo`）权重

本目录存放 `yolo` 引擎使用的 ONNX 权重。**权重不随仓库分发**，需自行获取。

## 需要的文件

| 文件 | 必需 | 说明 |
|---|---|---|
| `plate_det.onnx` | ✅ | 车牌检测网络，一次前向输出 bbox + 4 角点 + 单/双层类别 |
| `plate_rec.onnx` | ⬜ | 字符识别网络；**缺失时自动回退到 RapidOCR**（沿用 `deep` 引擎的后端） |

输入尺寸默认为 **640**（`YoloEngine.input_size`）；若你导出的权重用了别的
`imgsz`，运行时需传入一致的 `input_size`，否则检测框会整体错位。

## 启用方式

```bash
pip install -e ".[yolo]"     # 只装 onnxruntime，不引 torch
lpr engines                   # 确认 yolo 一行显示为「可用」
```

## 许可证义务（重要）

若权重基于 **Ultralytics YOLO** 训练或导出，则该权重受 **AGPL-3.0** 约束：

- 本项目自身为 **GPL-3.0**，两者可组合；
- 但若你把包含该权重的系统**作为网络服务提供**，AGPL 要求你向使用者提供
  对应源码 —— 请自行评估使用场景；
- 自查权重来源，并在 `docs/THIRD_PARTY.md` 登记：来源 URL、许可、导出参数
  （`imgsz` / 是否含 keypoint 分支 / 类别数）。

## 训练与导出（建议路线）

1. **数据**：CCPD（MIT，30 万张，文件名内嵌 bbox、四顶点、号码）。
   下载：<https://github.com/detectRecog/CCPD>
2. **训练**：YOLO-pose（`kpt_shape: [4,2]`，类别 `0=single` / `1=double`）。
   ⚠️ 若使用该数据集的**子集**训练，务必核对四角点标注的**索引顺序**；
   参考实现在翻转增广时需重排关键点（`keypoints[:, [1,0,3,2], :]`），
   写错会**静默毁掉训练**（见 `docs/REFERENCE_REVIEW.md` 2.2）。
3. **导出**：`yolo export format=onnx opset=12 imgsz=640`。
4. **接入**：补全 `src/lpr/engines/yolo_engine.py` 的 `_detect` / `_recognize_all`
   （两个方法目前是 `NotImplementedError` 占位，注释里写了各自要做什么）。
5. **验证**：加入 `tests/baselines/` 真值集对比，**必须**给出与 `deep` 引擎的
   同集精度/耗时对比数据，否则无从判断是否真的更好。
