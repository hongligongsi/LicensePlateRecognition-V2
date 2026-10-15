# 检测识别引擎（`yolo`）权重

本目录存放 `yolo` 引擎使用的 ONNX 权重。**权重不随仓库分发**，需自行获取。

## 需要的文件

| 文件 | 必需 | 说明 |
|---|---|---|
| `plate_det.onnx` | ✅ | 车牌检测网络，一次前向输出 bbox + 4 角点 + 单/双层类别 |
| `plate_rec.onnx` | ⬜ | 字符识别网络（CRNN + CTC）；**缺失时自动回退到复用 `deep` 引擎的识别管线** |
| `charset.txt` | ⬜ | 仅 `plate_rec.onnx` 存在时需要：每行一个字符，**第 1 行是 CTC blank 占位** |

输入尺寸默认为 **640**（`YoloEngine.input_size`）；若你导出的权重用了别的
`imgsz`，运行时需传入一致的 `input_size`，否则检测框会整体错位。

**放了 `plate_rec.onnx` 却没放 `charset.txt`** 时，识别权重会被忽略并回退到
`deep` 引擎（同时打一条警告）—— 字符表长度必须与模型输出宽度一致，
猜不出来就只能不用。

## 启用方式

```bash
pip install -e ".[yolo]"     # 只装 onnxruntime，不引 torch
lpr engines                   # 确认 yolo 一行显示为「可用」
```

## 先确认管道可用（不需要任何权重）

在拿到真权重之前，可以先造一个**输出为常量**的桩模型，把整条推理链路
（会话 → letterbox → 解码 → NMS → 逆变换 → 角点矫正）跑一遍：

```bash
python tools/make_stub_onnx.py --layout v8-pose --out models/yolo/plate_det.onnx --selftest
```

它**不会真的检测车牌**（输出与输入图无关），验证的是管道而不是模型。
`--layout` 支持 `v8-pose` / `v5-pose` / `kpt-first` / `v8-det` / `v5-det` 五种。

## 支持的输出布局

检测张量必须是**单张量**，形状为 `(1, N, C)` 或 `(1, C, N)`，`C` 为以下之一：

| 属性数 | 布局 | 来源 |
|---|---|---|
| `4 + nc` | box + cls | Ultralytics 纯检测导出 |
| `4 + 1 + nc` | box + obj + cls | YOLOv5 纯检测导出 |
| `4 + nc + 8` | box + cls + 4×角点 | Ultralytics YOLOv8-pose |
| `4 + 1 + nc + 8` | box + obj + cls + 4×角点 | Ultralytics YOLOv5-pose |
| `4 + nc + 12` | box + cls + 4×角点(x,y,conf) | 带关键点置信度的导出 |

朝向（`C×N` 还是 `N×C`）由「属性数远小于锚点数」自动判定；
有无 objectness 与角点维度由属性数反解。

### ⚠️ 两种属性列顺序无法从形状区分

`4 + 1 + nc + 8 = 15` 这个属性数，对应**两种真实的、互不兼容的**列顺序：

| `attr_order` | 列顺序 | 典型来源 |
|---|---|---|
| `cls_first`（引擎默认） | box + obj + **cls** + **kpts** | Ultralytics `yolo export` |
| `kpt_first`（**配置默认**） | box + obj + **kpts** + **cls** | `we0091234` 系 `plate_detect.onnx` |

**猜错不会报错**，只会把角点当类别读、类别当角点读，得到一堆看似合理的错框
（随后大概率被宽高比过滤掉，表现为"什么都检测不到"）。

判定方法：去看训练/导出那一侧的推理脚本怎么切列 —— 例如 `obj` 是取第 4 位
还是第 5 位、`argmax` 是在哪两位上做。使用时：

```python
YoloEngine(attr_order="kpt_first")   # 接 we0091234 系权重时必须显式指定
```

**CLI/GUI 不用手动设**：配置默认 `engine.yolo_attr_order = "kpt_first"`，
正好对齐路径 A 的 we0091234 权重，落盘 `plate_det.onnx` 后下拉框里就能直接用。
（Ultralytics 导出的权重请在配置里改回 `"cls_first"`。）
显式实参 > 配置 > 引擎默认值（`cls_first`），三层优先级见
`EngineConfig.yolo_attr_order` 的文档。

## 获取权重的三条路

### A. 用现成权重

```bash
python tools/fetch_yolo_weights.py --list                    # 已核实的来源（含许可与布局）
python tools/fetch_yolo_weights.py --inspect <模型.onnx>      # 检查布局能不能被本引擎解析
python tools/fetch_yolo_weights.py --url <URL> --sha256 <HASH>   # 下载（哈希必填）
python tools/fetch_yolo_weights.py --github owner/repo --asset '*det*.onnx'
```

从任意 URL 下载**强制要求 `--sha256`**（不给就得加 `--insecure-no-hash` 并承担
投毒风险）。下载完会自动跑一次布局检查。

### B. 自己训练（推荐，许可最干净）

数据用 CCPD（MIT，30 万张，文件名内嵌 bbox、四顶点、号码）：

```bash
python tools/ccpd_to_yolo.py --ccpd datasets/ccpd_raw --out datasets/ccpd_pose
python tools/train_plate_yolo.py --data datasets/ccpd_pose/data.yaml --export-onnx
```

`ccpd_to_yolo.py` 会处理好两个**写错也不报错**的地方：

1. **四角点顺序**：CCPD 是「右下→左下→左上→右上」，YOLO-pose 要
   「左上→右上→右下→左下」，脚本按 `[2,3,0,1]` 重排；
2. **`flip_idx`**：水平翻转后关键点必须重排，顺序为 `[1,0,3,2]`，写进 `data.yaml`。

`train_plate_yolo.py` 需要 `ultralytics` + `torch`。这两个**刻意不进本项目
的任何 extra**（PyTorch 一个包就比本项目全部运行时依赖加起来还大，且
Ultralytics 是 AGPL-3.0），请自行安装：

```bash
pip install ultralytics
```

导出用的是 Ultralytics 的默认列顺序 `cls_first`，正好是本引擎默认值，
**不需要**设 `attr_order`。

### C. 先只做检测

`recognize_backend="none"` 时引擎返回**带框但 `text` 为空**的结果，而不是
空列表。这不是失败而是明确降级 —— 定位评测（`tools/eval_ccpd.py` 之类）
只需要框，静默返回空列表反而会让人以为模型坏了。

**复用 `deep` 识别时也遵循同一条口径**：检测到但 OCR 一个字都没读出来的
候选，会保留框与 ROI，`text` 留空，`meta["ocr_failed"] = True`。这样
"没检到车牌"（结果列表为空）与"检到了但没认出"（有框、文本空）在数据上
可区分，而不是都表现为"未识别到车牌"。

## 许可证义务（重要）

| 来源 | 许可 | 能否与本项目组合 |
|---|---|---|
| 本项目自身 | GPL-3.0-only | — |
| `we0091234/Chinese_license_plate_detection_recognition` | GPL-3.0 | ✅ 同为 GPL-3.0 |
| Ultralytics YOLO 训练/导出的权重 | **AGPL-3.0** | ⚠️ 见下 |
| CCPD 数据集 | MIT | ✅ |
| `opencv/opencv_zoo` LPD YuNet | Apache-2.0 | ✅（但本引擎暂不支持其多头输出） |

**Ultralytics 权重受 AGPL-3.0 约束**：若你把包含该权重的系统**作为网络服务
提供**，AGPL 要求你向使用者提供对应源码。自训权重同样受此约束（因为训练
用的是 Ultralytics 代码）。请自行评估使用场景。

自查权重来源，并在 `docs/THIRD_PARTY.md` 登记：来源 URL、许可、导出参数
（`imgsz` / 是否含 keypoint 分支 / 类别数 / `attr_order`）。

## 验证

```bash
python tools/fetch_yolo_weights.py --inspect models/yolo/plate_det.onnx   # 布局自检
lpr predict samples/ --engine yolo                                        # 端到端
```

接入新权重后**必须**给出与 `deep` 引擎的**同集精度/耗时对比**，
否则无从判断它是否真的更好：

```bash
python tools/benchmark.py --images samples --repeat 3
```
