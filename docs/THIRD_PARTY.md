# 第三方组件与许可

本文件列出 lpr-v2 直接依赖的第三方组件及其许可，供分发时履行署名与合规义务。

> 数据来源：各组件已安装版本的分发元数据（`importlib.metadata`），
> 经 `python -c` 实读 `License-Expression` / `License` / `Classifier` 字段获得，
> 非凭记忆填写。版本号为本仓库当前锁定的版本。

## 运行时依赖（`dependencies`）

| 组件 | 版本 | 许可 | 说明 |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause（含 0BSD / MIT / Zlib / CC0-1.0 混合） | 数值计算基础 |
| opencv-python | 4.14.0.94 | Apache-2.0 | 图像处理；`<5` 上界见 pyproject 注释 |
| opencv-python-headless | 4.14.0.94 | Apache-2.0 | CI / 无显示环境 |
| Pillow | 12.3.0 | MIT-CMU | 图像读写与中文路径支持 |

## 可选依赖（`optional-dependencies`）

| 组件 | 版本 | 许可 | 对应 extra |
|---|---|---|---|
| rapidocr-onnxruntime | 1.2.3 | Apache-2.0 | `deep` |
| onnxruntime | 1.31.0 | MIT | `deep`（rapidocr 的推理后端） |
| pyclipper | 1.4.0 | MIT（OSI Approved） | `deep`（rapidocr 传递依赖） |
| Shapely | 2.2.0 | BSD-3-Clause | `deep`（rapidocr 传递依赖） |
| six | 1.17.0 | MIT | `deep`（rapidocr 传递依赖） |
| PySide6-Essentials | 6.12.0 | **LGPL-3.0-only** 或 GPL-2.0-only 或 GPL-3.0 | `gui` |
| paddleocr / paddlepaddle | — | Apache-2.0 | `deep-paddle`（未在 CI 中启用） |

### 关于 PySide6（LGPL-3.0）

`PySide6-Essentials` 允许在 **LGPL-3.0** 下使用，这要求：

- **动态链接**使用（本项目以独立 wheel 依赖方式导入，不静态编译进产物，满足此条）；
- 若你**再分发**包含 PySide6 的安装包，需一并提供其许可全文与获取源码的途径；
- 修改 PySide6 自身源码后，你的修改部分亦须以 LGPL-3.0 提供。

本项目自身以 GPL-3.0 发布，与 LGPL-3.0 兼容（GPL-3.0 §13 允许与 LGPL-3.0 组合）。

## 开发依赖（`dev`）

| 组件 | 版本 | 许可 |
|---|---|---|
| pytest | 9.1.1 | MIT |
| pytest-timeout | 2.4.0 | DFSG approved（MIT 系） |
| ruff | 0.17.0 | MIT |
| mypy | 2.4.0 | MIT |
| onnx | 1.23.2 | Apache-2.0 |

> `onnx` 只用于**测试与造桩**（`tools/make_stub_onnx.py` 构造常量输出的
> `.onnx` 文件、校验形状），**运行时不需要** —— 运行时只依赖 `onnxruntime`。
> 把它放进 `dev` 而不是 `yolo` extra，是为了让 `pip install -e ".[yolo]"` 的用户
> 不必为一个造测试数据的包买单。

## 模型权重：本仓库不分发

这是本项目**唯一需要用户自行判断许可**的部分，所以单列。

| 权重 | 来源 | 许可 | 仓库是否分发 |
|---|---|---|---|
| `models/svm.dat`、`models/svmchinese.dat` | 原项目 EasyPR 样本训练 | MIT（随原项目） | **是**（`config.json` 内置 SHA256） |
| `models/yolo/plate_det.onnx` | 用户自行准备 | **取决于用户所选来源** | **否**（`.gitignore` 已排除） |
| `models/yolo/plate_rec.onnx` | 用户自行准备 | 同上 | **否** |
| `models/yolo/charset.txt` | 需与识别权重配套 | 同上 | **否** |
| RapidOCR / PaddleOCR 的 OCR 模型 | 首次运行自动下载 | Apache-2.0 | 否（缓存到用户目录） |

**为什么 `yolo` 权重不进仓库**：主流车牌检测权重多派生自 Ultralytics
（AGPL-3.0）或 `we0091234`（GPL-3.0）。把它们**随仓库分发**意味着本项目要
额外承担 AGPL 的传染义务（尤其 AGPL 对网络服务分发的要求），而本项目自身
是 GPL-3.0-only。让用户自己取权重、仓库只提供**加载与校验工具**，是刻意
选择的边界。

**用户自取权重时的义务提示**（`tools/fetch_yolo_weights.py --list` 会打印）：

- Ultralytics 血统的权重 → **AGPL-3.0**：把它们用在与他人交互的网络服务里，
  需按 AGPL 提供完整源码（含你对本项目的修改）；
- `we0091234/Chinese_license_plate_detection_recognition` 的权重 → **GPL-3.0**，
  与本项目兼容，但再分发时同样要提供源码；
- OpenCV Model Zoo 的 LPD 系（Apache-2.0）许可最宽松，但**输出格式不同**
  （YuNet 多输出头，非单张量 pose 式），**本引擎不支持**，工具会明确拒绝
  并说明原因，而不是让它悄悄加载失败。

## 训练链路的额外依赖（不进任何 extra）

`tools/train_plate_yolo.py` 与 `tools/ccpd_to_yolo.py` 的自训练路线需要：

| 组件 | 许可 | 说明 |
|---|---|---|
| PyTorch | BSD-3-Clause | 训练与导出；体积数百 MB 起 |
| Ultralytics | **AGPL-3.0** | `yolo11n-pose.pt` 等预训练骨架与训练器 |
| CCPD 数据集 | MIT（学术使用） | 约 12 GB / 30 万张，标注编码在文件名里 |

这三者**不在 `pyproject.toml` 的任何 extra 里**，需用户自行安装。原因是体积
（PyTorch 一个包就超过本项目全部运行时依赖之和）与许可（AGPL）都不适合
设为默认路径。转换与训练脚本在缺少它们时会给出明确的安装提示。

## 历史与参考

- 上游原项目 `hongligongsi/License-Plate-Recognition`（wzh191920，2021）：**MIT**。
  其声明依 MIT 条款保留在本仓库 `LICENSE` 末尾与 `LICENSE.MIT.original`。
- 架构思路参考（**未引入任何代码或权重**）：`we0091234/Chinese_license_plate_detection_recognition`
  （GPL-3.0，检测用 Ultralytics YOLOv5 权重 → AGPL-3.0）。本项目仅借鉴「YOLO 检测 +
  CRNN 识别」的公开架构思路，未复制其代码、未分发其权重。
- 若后续新增引擎需要引入第三方权重，**必须在本文件登记其许可与来源**，并在运行时
  以显式提示告知用户相应的许可义务。
