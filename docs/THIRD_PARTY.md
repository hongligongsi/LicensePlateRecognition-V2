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

## 历史与参考

- 上游原项目 `hongligongsi/License-Plate-Recognition`（wzh191920，2021）：**MIT**。
  其声明依 MIT 条款保留在本仓库 `LICENSE` 末尾与 `LICENSE.MIT.original`。
- 架构思路参考（**未引入任何代码或权重**）：`we0091234/Chinese_license_plate_detection_recognition`
  （GPL-3.0，检测用 Ultralytics YOLOv5 权重 → AGPL-3.0）。本项目仅借鉴「YOLO 检测 +
  CRNN 识别」的公开架构思路，未复制其代码、未分发其权重。
- 若后续新增引擎需要引入第三方权重，**必须在本文件登记其许可与来源**，并在运行时
  以显式提示告知用户相应的许可义务。
