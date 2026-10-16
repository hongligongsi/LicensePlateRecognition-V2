# 车牌识别系统 v2.7.3（License-Plate-Recognition 升级版）

用 Python + OpenCV + 深度学习做的**中国车牌识别**，支持蓝牌 / 黄牌 / 绿牌（新能源）/ 白牌 / 黑牌 / 危险品运输牌。

![界面截图](docs/screenshot.png)

本项目是 [hongligongsi/License-Plate-Recognition](https://github.com/hongligongsi/License-Plate-Recognition) 的现代化升级版。原项目最后一次提交停留在 2021 年，其依赖（Python 3.4 / OpenCV 3.4 / numpy 1.14）在如今的机器上已无法安装。v2 在保留原算法思想的基础上全面重构。

**一句话上手**（Windows，已装 Python 3.10+）：

```bat
双击 start.bat
```

**一句话评效果**：`yolo` 引擎在 100 张 CCPD 上严格命中 85%，定位 100%；`deep` 引擎在特殊牌（港/警/领/学/双层）样张上 10/10。**没有全能引擎**，选哪个取决于你的车牌构成 —— 实测依据见 [docs/ACCURACY.md](docs/ACCURACY.md)。

---

## 目录

- [1. 它是什么](#1-它是什么)
- [2. 快速开始](#2-快速开始)
- [3. 安装与启动](#3-安装与启动)
- [4. 四个引擎](#4-四个引擎)
- [5. 命令行完整参考](#5-命令行完整参考)
- [6. 图形界面](#6-图形界面)
- [7. Python API](#7-python-api)
- [8. 配置参考（config.json 全字段）](#8-配置参考（configjson-全字段）)
- [9. 安全与隐私](#9-安全与隐私)
- [10. 架构与算法](#10-架构与算法)
- [11. 准确率实测](#11-准确率实测)
- [12. 项目结构](#12-项目结构)
- [13. 测试、评测与 CI](#13-测试、评测与-ci)
- [14. 故障排查（FAQ）](#14-故障排查（faq）)
- [15. 已知限制](#15-已知限制)
- [16. 开发与发布](#16-开发与发布)
- [17. 许可](#17-许可)
- [附：版本演进](#附：版本演进)

---

## 1. 它是什么

### 1.1 能力范围

| 维度 | 支持情况 |
|---|---|
| **车牌底色** | 蓝、黄、绿（新能源）、白、黑，另加**危险品运输牌（黑底黄字）** |
| **号牌格式** | **13 类 / 19 条格式规则**：普通民用、新能源、警用、军队、武警、使领馆、挂车、教练车、危险品运输、港澳入出境、短牌、农机、民航（`民航A0001`，依据 CCAR-331SB-R1） |
| **版式维度** | 单行 / 双层（与底色**正交**的两个维度，结果里是独立字段） |
| **输入来源** | 图片、目录（可递归）、视频文件、摄像头流、numpy 数组 |
| **输出格式** | 控制台、JSON、CSV、标注图片、标注视频 |
| **运行环境** | Windows / Linux / macOS，**纯 CPU，不需要 GPU、不需要 CUDA** |

### 1.2 三条互相正交的描述维度

车牌的描述由三个**正交**维度构成，不要混为一谈：

```
text   —— 号码本身   京A12345
color  —— 底色       蓝牌 / 黄牌 / 绿牌 / 白牌 / 黑牌 / 危险品运输牌
layout —— 版式       单行 / 双层
```

颜色与版式必须分开表达：大型汽车号牌是"黄底双行"，小型汽车是"蓝底单行"，大型新能源车还有"黄绿双拼双行"。仅用底色或仅用版式都会丢信息。

### 1.3 它不做什么（诚实边界）

- **不做训练**。仓库只做推理与评测；自训练路线需要你自己装 PyTorch + Ultralytics（见 [§4.5](#45-yolo--检测识别引擎)）。
- **不随仓库分发 `yolo` 权重**（体积 + 许可双重原因），所以全新克隆后 `yolo` 大概率是「可见但不可用」。
- **没有 CI 级别的准确率门槛**（现有的是"行为不漂移"快照），原因见 [§13.3](#133-为什么-ci-里没有准确率门槛)。
- 详细的边界清单见 [§15 已知限制](#15-已知限制)。

---

## 2. 快速开始

### 2.1 Windows：双击 `start.bat`

脚本自动找解释器、校验版本、补齐依赖、自检权重与引擎、启动界面。第二次双击复用 `.venv`，秒开。详见 [§3.4](#34-四种启动方式)。

### 2.2 通用：装成命令

```bash
pip install -e ".[deep,gui]"     # CV 引擎 + 深度学习引擎 + 图形界面
lpr gui                          # 开界面
```

### 2.3 三条命令跑通

```bash
# ① 看看当前环境有哪些引擎可用（以及为什么某些不可用）
lpr engines

# ② 识别单张图片
lpr predict img/single_blue.jpg --engine deep

# ③ 校验号码是否符合国家规范（离线、不需要图片）
lpr check 京AD77972 京A1234危 --fix
```

### 2.4 真实的输出长什么样

`lpr engines`（本机实测，五个引擎全部可用）：

```text
引擎               状态       说明
------------------------------------------------------------------------------
传统 CV 引擎         可用       OpenCV 形态学定位 + 投影分割 + SVM 字符识别，纯 CPU、轻量、可解释
深度学习引擎           可用       定位引导 OCR（跳过检测网络）+ 超分 + 语法投票，速度快、中文与模糊车牌准确率高
三合一引擎            可用       同时编排 cv / deep / yolo：并联投票融合（默认）/ 顺序兜底链 / 特征选优，缺权重或依赖的成员自动退席；融合结果保留各成员读数与票数明细
检测识别引擎           可用       车牌专用检测网络（bbox + 四角点 + 单/双层）+ 识别，对侧倾/模糊车牌的定位召回优于轮廓法；需自行下载 ONNX 权重
```

> `lpr engines` 的表格列宽用朴素的 `f"{name:<16}"` 计算，中文按"字符数"而不是"显示宽度"对齐，所以汉字列看起来会错位。这是显示问题，不影响内容。

`lpr predict`（`deep` 引擎，单图两块牌）：

```text
✓ single_blue.jpg: 辽DU4356 [蓝牌/普通民用] 置信度=0.91 (3072 ms)
✓ single_blue.jpg: 浙B2V9L7 [蓝牌/普通民用] 置信度=0.86 (3072 ms)

共 1 张 | 命中 1 张 (100.0%) | 平均 3072 ms/张
去重后车牌: 辽DU4356, 浙B2V9L7
```

第一个 `✓`/`?` 是置信度标记（≥ 0.7 记 `✓`，否则 `?`）；括号里是**整张图**的耗时而不是单条结果的耗时。

`lpr check`：

```text
京A1234危        ✓ 合法       类型=危险品运输
               省份=京 发牌机关=A 新能源=否
京AD77972       ✓ 合法       类型=新能源
               省份=京 发牌机关=A 新能源=是
皖A123          ✗ 不合法      类型=未知
               省份=皖 发牌机关=A 新能源=否
               → 无法自动纠正（未发现可修正项）

2/3 个号码合法
```

---

## 3. 安装与启动

### 3.1 最低要求

| 项目 | 要求 | 说明 |
|---|---|---|
| **操作系统** | **Windows 10 (1809+) / Windows 11（64 位）**、Linux（64 位） | 兼容矩阵见 [§3.2](#32-系统兼容矩阵) |
| **Python** | **3.10 ~ 3.13，必须 64 位** | `requires-python = ">=3.10"`；CI 覆盖 3.10 / 3.11 / 3.12 / 3.13 |
| **CPU** | 任意 x86-64 | **不需要 GPU、不需要 CUDA**，推理全程纯 CPU |
| **内存** | 可用 2 GB 以上 | deep 引擎加载 OCR 模型约占 500 MB |
| **磁盘** | 源码约 30 MB；装完依赖约 300 MB | 仓库自带 SVM 模型 8 MB；OCR 模型 13.7 MB 随 wheel 分发；`PySide6-Essentials` 约 100 MB、`rapidocr-onnxruntime` 约 130 MB |
| **网络** | **只在 `pip install` 时需要** | 装完后从识别到界面**全程离线**（模型都已随包分发） |
| **摄像头** | 可选 | 只有「摄像头实时识别」功能需要 |

### 3.2 系统兼容矩阵

| 系统 | 命令行 | 图形界面 | 说明 |
|---|---|---|---|
| **Windows 11（64 位）** | ✅ | ✅ | 本项目开发与实测环境（Python 3.13） |
| **Windows 10 1809+（64 位）** | ✅ | ✅ | CI 在 `windows-latest` 上覆盖 Python 3.12 |
| Windows 10（1809 之前，如 1607 / LTSB 2016） | ⚠️ | ⚠️ | 未验证。PySide6（Qt 6.6+）官方要求 1809 以上 |
| Windows 8.1（64 位） | ⚠️ | ❌ | Python 3.13 官方仍支持 8.1，但 Qt 6 已放弃 → **界面用不了** |
| Windows 7 | ❌ | ❌ | 见 [§3.3](#33-为什么-windows-7--8--xp-不行) |
| Windows XP | ❌ | ❌ | 同上 |
| 32 位 Windows（任意版本） | ⚠️ | ❌ | Qt 6 没有 32 位 Windows 版 → 界面用不了 |
| **Linux（Ubuntu 20.04+ / 22.04+，64 位）** | ✅ | ✅ | CI 覆盖 Ubuntu × Python 3.10~3.13；摄像头走 V4L2 |
| macOS 12+ | ✅ | ✅ | Qt 6 要求 macOS 12+；本项目**没有** macOS 的 CI 覆盖 |

### 3.3 为什么 Windows 7 / 8 / XP 不行

两道**互相独立**的硬门槛，任何一道都足以挡住：

1. **Python 本体**。CPython 的规则是「只在 Microsoft 仍处于延长支持期的 Windows 上支持」。
   最后一个能装到 **Windows 7** 的版本是 **Python 3.8**（官方文档原话：
   *"If you require Windows 7 support, please install Python 3.8"*）；
   **Windows XP** 更早，止步于 **Python 3.4**。
   而本项目要求 `Python >= 3.10` —— 门直接关着。
2. **图形界面**。界面基于 **PySide6（Qt 6）**，官方支持矩阵是 **64 位 Windows 10 1809+**。
   Qt 6 明确放弃了 Windows 7 / 8 / 8.1 以及**全部 32 位 Windows**
   （要 Win7 就得退到 Qt 5.15，要 8.1 就得退到 Qt 5.12）。

> 想让它在 Win7 上跑，属于一次**独立移植**，不是改个配置就行：要降到 Python 3.8 +
> `PySide2`(Qt5) + 旧版 numpy/opencv，还得放宽 `requires-python`，并自行确认那些旧
> wheel 在 Win7 上装得上。本仓库没做、也没承诺这条路径。

### 3.4 四种启动方式

#### ① 双击 `start.bat`（Windows 最省事）

```
双击 start.bat
```

脚本会自动走五步：

1. **找解释器**：`LPR_PYTHON` 环境变量 → 项目内 `.venv` → `py -3` → `python`；
2. **校验版本**：低于 3.10 直接报错退出（而不是抛一堆 `ImportError`）；
3. **补齐依赖**：探测 **numpy / OpenCV / Pillow / PySide6 四项**是否齐全，
   缺则装 `-e ".[deep,gui]"`（核心库 + 深度引擎 + 界面，约 200 MB，只在首次）；
4. **自检模型权重与引擎**：看 `models/` 下的权重、逐个探测引擎后端
   （`rapidocr_onnxruntime` / `paddleocr` / `onnxruntime`），报出四个引擎谁可用；
5. **启动**：无参数开界面，有参数透传给命令行。

第 4 步只说事实，**不阻断启动**：缺了 `yolo` 的权重只是这个引擎不可用，界面和
其余引擎照常。它每次都会打印一行结论，例如：

```text
[4/5] 模型权重与引擎
      可用：传统 CV 引擎 / 深度学习引擎 / 三合一引擎
      不可用：检测识别引擎 —— 未安装 onnxruntime（检测推理后端）；缺少模型权重：models/yolo/plate_det.onnx
              修复：pip install -e ".[yolo]"
                    python tools/fetch_yolo_weights.py --list          # 已知来源与许可
                    python tools/fetch_yolo_weights.py --github <owner/repo> --asset "*det*.onnx"
      失去的只是对应引擎，界面与其他引擎照常可用；
      要看完整报告执行：start.bat doctor
```

缺依赖时**装进项目内的 `.venv`**，不会往你系统里已有的 Python 塞包；如果那个
解释器本来就齐全，则直接复用、不重复下载。第二次双击复用 `.venv`，秒开。

> **为什么逻辑不在 `.bat` 里？** `start.bat` 只是一个纯 ASCII 的引导壳，
> 真正的启动逻辑和中文提示都在 `tools/launcher.py`（只依赖标准库）。
> 原因很硬：cmd.exe 计算批处理「下一行」的偏移时，拿上一行的**字符数**当
> **字节偏移**用，文件里只要出现一个汉字，下一行就会被超前若干字节开始读，
> 剩下的残片会被当成命令执行：
>
> ```text
>        完成 —— CV 引擎、深度引擎、图形界面均已就绪。
> '，未改动你系统里的' 不是内部或外部命令，也不是可运行的程序。
> ```
>
> 更坑的是它不稳定 —— 短文件里超前量往往刚好落在下一行的缩进空格里，
> 看起来完全正常。所以中文一律交给 Python 打印（Python 走 `WriteConsoleW`，
> 与控制台代码页无关）。

参数透传给命令行：

```bat
start.bat engines                                :: 查看引擎状态
start.bat predict car.jpg --engine deep          :: 单图识别
start.bat --help                                 :: 命令行用法
```

**`start.bat doctor` —— 只体检，不启动**

环境不对劲时（下拉框里某个引擎写「（不可用）」、装完依赖仍报缺模块），先跑它。
它把**解释器 / 运行依赖 / 模型权重 / 四个引擎**一次列全，每条不可用都附修复命令：

```bat
start.bat doctor          :: 也可以写 --doctor 或 --check
```

```text
运行依赖
  [ok] numpy                  可用     运行库（必需）
  [ok] cv2                    可用     运行库（必需）
  [ok] PIL                    可用     运行库（必需）
  [ok] PySide6                可用     图形界面（必需）
  [ok] rapidocr_onnxruntime   可用     deep 的默认识别后端（推荐，轻量）
  [--] paddleocr              未安装   deep 的备选识别后端（更大更慢），未安装不影响
  [!!] onnxruntime            不可用   yolo 的唯一硬依赖（默认安装规格不含）

模型权重
  [ok] SVM 分类器（字母数字）         4.38 MB   models/svm.dat
  [ok] 车牌检测网络（bbox + 四角点）  未提供    models/yolo/plate_det.onnx
        python tools/fetch_yolo_weights.py --list          # 已知来源与许可
  [--] 车牌字符识别（可选）           未提供    models/yolo/plate_rec.onnx
```

两条刻意的设计：

- **只读**：不建 `.venv`、不装包、不改任何文件，所以环境已经坏掉时也能放心跑；
  四个引擎全部可用时退出码 0，否则为 1（便于脚本判断）。
- **不制造假警报**：可选的东西缺了标 `[--]`（例如只装了 `rapidocr` 没装
  `paddleocr`），只有真正拖垮了某个引擎才会标 `[!!]`。

两个环境变量可覆盖默认行为（完整清单见 [§8.7](#87-环境变量)）：

```bat
set LPR_PYTHON=C:\Python312\python.exe   :: 指定解释器（跳过搜索）
set LPR_NO_VENV=1                        :: 不建 .venv，直接装进找到的解释器
```

#### ② 装成命令（推荐长期使用）

```bash
pip install -e ".[deep,gui]"    # 一次性：CV + 深度学习引擎 + 图形界面
lpr gui                          # 开界面
lpr predict car.jpg --engine deep
```

#### ③ 不安装，直接用源码

```bash
# Windows
set PYTHONPATH=src
# Linux / macOS
export PYTHONPATH=src

python -m lpr                    # 无参数 → 直接开图形界面
python -m lpr predict car.jpg --engine deep
python -m lpr.cli --help         # 只看命令行用法
```

#### ④ 最小安装（只要命令行 + CV 引擎，零额外依赖）

```bash
pip install -e .
lpr engines                      # 看看当前环境有哪些引擎可用
```

### 3.5 引擎 × 依赖 × 权重对照表

| 引擎 / 功能 | 额外依赖 | 装完后还要下载模型吗 |
|---|---|---|
| `cv`（基础） | 无（numpy + opencv + Pillow） | 否，`models/*.dat` 随仓库分发 |
| `deep`（推荐） | `rapidocr-onnxruntime==1.2.3` | 否，模型 13.7 MB 在 wheel 内 |
| `deep-paddle` | `paddleocr` + `paddlepaddle` | **首次运行会联网下载模型**，离线环境别用 |
| `yolo`（需自备权重） | `onnxruntime>=1.16` | **是**，`models/yolo/*.onnx` 需自行准备 |
| `fusion`（三合一） | 无额外依赖（用成员各自所需的那份） | 取决于成员：`cv`/`deep` 装完即用，`yolo` 需权重 |
| 图形界面 | `PySide6-Essentials>=6.5` | 否 |

> 界面只用到 `QtCore` / `QtGui` / `QtWidgets`，因此依赖 `PySide6-Essentials`
> 而不是元包 `PySide6`——后者会连带装 `PySide6-Addons`（WebEngine / Charts /
> Multimedia / 3D 等，约 130 MB），本项目一个都不用。

> **纯离线环境请用 `deep`（RapidOCR）**：它和 `cv` 一样装完即用；
> `deep-paddle` 首次运行需要联网拉模型。

### 3.6 安装命令一览

```bash
# 基础（传统 CV 引擎）
pip install -e .

# 推荐：+ 深度学习引擎 + 界面
pip install -e ".[deep,gui]"

# 可选：PaddleOCR 后端（更大更慢但部分场景更准）
pip install -e ".[deep-paddle]"

# 可选：车牌检测网络引擎（ONNX 推理，需自备权重）
# 只装 yolo 则字符识别缺席（纯检测模式）；要和识别一起用请装 ".[yolo,deep]"
pip install -e ".[yolo]"

# 开发（含测试）
pip install -e ".[dev]"
```

> **`.[yolo]` 装在哪儿、权重放哪儿**：extras 只装 `onnxruntime`；**权重不进仓库**。
> 装完先跑 `python tools/make_stub_onnx.py --out models/yolo/plate_det.onnx --selftest`
> 验证管道通不通（这条不联网、不要第三方权重），再按
> [models/yolo/README.md](models/yolo/README.md) 换成真权重。

> **注意**：`opencv-python` 必须是 **4.x**（`>=4.10,<5`）。OpenCV 5.0 移除了 `cv2.ml`，原版 SVM 模型将无法加载，`pyproject.toml` 已锁定上限。

SVM 模型（`models/svm.dat`、`models/svmchinese.dat`）已随仓库附带，来自原项目（EasyPR 样本训练）。

### 3.7 目录归属：哪些是资产、哪些是产物

仓库根目录下有几个"看起来像源码目录其实是一次性产物"的文件夹，这里一次说清：

| 目录 | 谁写的 | 能否删 | 说明 |
|---|---|---|---|
| `img/`、`samples/` | 仓库自带 + 你下载的 | ❌ 不能删 | **入库**的真实样张。CI 的回归基线与准确率门槛都用它们。注意 `img/` 也被当成下载样本的落图目录（`tools/fetch_ccpd.py` 下到 `datasets/ccpd_raw` 后常被搬进来），2026-10-16 时已 403 张 / 75.7MB —— 往这里加图必须重新生成基线，否则 CI 会红（见 §13.4 与 `docs/ROADMAP.md` 第 4 项） |
| `models/` | 仓库自带 + 你放的 | ❌ | `svm*.dat` 入库；`yolo/*.onnx` 需自备（已 gitignore） |
| `annotated/` | `lpr predict --save-annotated` | ✅ 可删 | 标注图的**默认**输出目录（不写参数时用它）。已 gitignore |
| `out/` | 文档示例里的 `--report out/` | ✅ 可删 | 只是文档惯用的输出目录名，**不是代码里的常量**。已 gitignore |
| `reports/` | 评测/基准/调参脚本 | ✅ 可删可重生成 | `eval_*.py`、`benchmark.py`、`tune_*.py` 的 JSON/DTO 落盘。已 gitignore |
| `images/` | 一次性集中副本 | ✅ 可删 | `img/` + `samples/` + `datasets/ccpd_labeled/` 的拷贝，为省事集中放一起。已 gitignore（体积）。**2026-10-16 时已空**，那份活实际被 `img/` 兼了（见上一行） |
| `datasets/` | `tools/fetch_*.py` | ✅ 可删可重下 | 下载的数据集。已 gitignore |
| `tests/baselines/hard/` | `tools/make_hard_samples.py` | ✅ 可删可重生成 | 确定性生成的对抗退化样张。已 gitignore |
| `logs/` | 程序自己 | ✅ 可删 | **运行日志**（下面的 §3.8）。已 gitignore，删了只是丢掉历史记录 |

### 3.8 运行日志：文件在哪、怎么关、怎么排错

**每次运行都会留下日志文件**，不用加任何参数：

```text
logs/
├── lpr-20261016.log            # 主进程的日志（同一天多次运行追加到同一个文件）
├── lpr-20261016-p34240.log     # --workers 多进程时，每个子进程一个文件（名字带 pid）
└── lpr-20261016-p46668.log
```

命令行启动时第一行就会告诉你它在哪：

```text
17:32:22 [INFO] lpr: 运行日志 → C:\...\lpr-v2\logs\lpr-20261016.log
```

`lpr predict ./photos/ --engine yolo --workers 2` 的真实日志（3 个文件各 3 行）：

```text
# logs/lpr-20261016.log（主进程）
17:32:22 [INFO] lpr: 运行日志 → C:\...\lpr-v2\logs\lpr-20261016.log
17:32:22 [INFO] lpr.engines.yolo: 检测识别引擎就绪 | 检测=plate_det.onnx | 识别=复用 deep 引擎 | 后端=AzureExecutionProvider | 输入=640
17:32:22 [INFO] lpr.batch: 并行批量识别: 201 个文件 / 2 进程 / 8 分片

# logs/lpr-20261016-p34240.log（子进程）
17:32:23 [INFO] lpr: 运行日志 → C:\...\lpr-v2\logs\lpr-20261016-p34240.log
17:32:24 [INFO] lpr.engines.yolo: 检测识别引擎就绪 | 检测=plate_det.onnx | 识别=复用 deep 引擎 | 后端=AzureExecutionProvider | 输入=640
17:32:25 [INFO] lpr.engines.deep: 深度学习引擎就绪 | 后端=rapidocr | 定位引导=启用 | 纯识别加速=启用 | 第二模型=关闭
```

关键约定：

| 事项 | 行为 |
|---|---|
| 文件大小 | 单文件超过 `log_max_mb`（默认 5MB）轮转为 `*.log.1`、`*.log.2`…，保留 `log_backups` 份（默认 5） |
| 日期切分 | 文件名带日期，**跨天自动换新文件**；同一天多次运行是**追加**，不是覆盖 |
| 多进程 | `--workers` 时每个子进程写自己的 `-p<pid>.log` —— 多个进程轮流轮转同一个文件会在 Windows 上抛 `PermissionError`，而且"哪个进程读了哪张图"本身就是排查信息 |
| 隐私 | 日志同样是 `mask_logs` 的管辖范围，车牌号照常在落盘前打码（见 §9.1） |
| 写不进去时 | 目录只读 / 磁盘满 / 被杀软占用 → **只降级为"不写文件"**并在 stderr 说明，绝不阻止程序启动 |
| 关闭 | 命令行 `--no-log-file`，或环境变量 `LPR_LOG_FILE=0`，或 `config.json` 里 `"log_to_file": false` |
| 界面里看 | 主窗口右下角有「运行日志」面板（读内存缓冲，不读文件），下面一行显示文件路径 + 「打开日志目录」按钮 |

默认级别是 `INFO`，想看细节用 `lpr predict a.jpg --log-level DEBUG`。

---

## 4. 四个引擎

四个后端**共用同一套接口**（`BaseEngine`），文件格式与调用方式完全一致，`lpr engines` 可列出可用性。选择引擎只改一个参数。

### 4.1 总览

| 引擎 | 一句话 | 依赖 | 需要权重 | 实测中位耗时 |
|---|---|---|---|---|
| `cv` | 传统 CV，快但精度有限 | 无 | 自动（入库） | **28 ms**（`samples/`）/ 36 ms（CCPD） |
| `deep` | 定位引导 OCR，精度主力 | `rapidocr-onnxruntime` | 自动（随 wheel） | **194 ms**（`samples/`）/ 716 ms（CCPD） |
| `cascade` | `cv` 定位 + `deep` 识别（= deep 的别名） | 同 `deep` | 自动 | 同 `deep` |
| `yolo` | 车牌专用检测网络 + 识别 | `onnxruntime` | **需自备** | **125 ms**（CCPD） |
| `fusion` | 把上面三个编排成一个 | 成员各自所需 | 取决于成员 | 190 ms（chain）～ 1737 ms（ensemble） |

> CCPD 是 720×1160 的整景照片，`samples/` 是 4096×2304 的高分辨率样张 ——
> 两组数字分母不同，不要跨列比较。耗时在 Windows + CPU 上实测，绝对值不可横向引用。

### 4.2 `cv` —— 传统 CV 引擎

OpenCV 形态学定位 + 投影分割 + SVM 字符识别（兼容原版模型），纯 CPU、零额外依赖、**约 28 ms/张**。

> ⚠️ **`cv` 的 SVM 对任何输入都输出置信度 1.0（严重过自信），字符识别错误率高。**
> 100 张 CCPD 实测 **0/100 命中**，输出 691 条误检，而它的**定位**其实是能找到车牌
> 的（定位命中 76%、平均 IoU 0.643）—— 它是"找到了牌但每个字都读错"。
> **只把它当快速筛候选用**，精度请用 `deep` 或 `yolo`。机制分析见
> [ACCURACY.md §3.4](docs/ACCURACY.md)。

### 4.3 `deep` —— 深度学习引擎

基于 RapidOCR（ONNX）/ PaddleOCR。中文省份简称与模糊车牌的准确率远高于 SVM，**约 190 ms/张**。

核心加速思路：**跳过 OCR 的文本检测网络**。车牌的文本检测（DBNet）本身是重型网络，而我们已用 CV 定位拿到了车牌区域，因此直接把整块 ROI 当作文本框送入识别网络（`get_boxes_img_without_det`）。

识别侧不是一次前向就完事，而是一套**多尺度投票**流程（详见 [§10.6](#106-识别、融合与纠错)）：

- **多尺度投票**：超分（目标字符高 48px）× 锐化（USM）组合投票，
  票重 = OCR 置信度 × 语法合法性；
- **早停收敛**：首个尺度即得到合法号码且票数占比过半就停止，典型场景只跑 1～2 轮；
- **补救预处理**（v2.7.2）：主计划全部落空时，换**增强方式**（去噪 / CLAHE / OTSU 二值化）再试一轮；
- **语法纠错**：`plate_rules` 按真实车牌规则纠错（I/O 不合法、发牌机关须为字母、末位降级）。

### 4.4 `cascade` —— 级联（默认引擎）

CV 快速定位给出候选 → 深度引擎精识别，是**精度与速度的折中**，也是 `config.json` 里的默认值。

> 注意：`engine.default = "cascade"` 里的 `cascade` 只是 **`deep` 的别名**
> （`deep` 引擎内部本来就是"CV 定位 → 纯识别网络"的级联）。
> 与之**正交**的另一个字段是 `engine.mode`：`"cascade"` 时主引擎没命中会自动
> 再试另一个可用引擎（跨引擎兜底），`"single"` 只跑主引擎。
> 所以 `default="cascade"` + `mode="single"` 并不矛盾。详见 [§8.2](#82-engine--引擎选择)。

### 4.5 `yolo` —— 检测识别引擎

车牌专用检测网络，一次前向同时给出 **bbox + 四角点 + 单/双层分类**，再用四点透视变换把倾斜车牌拉正后识别字符。对**侧倾、透视变形、模糊**车牌的定位召回优于轮廓法（`cv`/`deep` 靠形态学找轮廓，侧倾大了就找不到候选）。ONNX 推理，**不需要 PyTorch**。

100 张 CCPD 实测：**严格命中 85%、定位 100%、定位精确率 98%、平均 IoU 0.878，中位 125 ms** —— 综合表现最好的单引擎。

**三条取权重的路**（权重不放仓库：体积 + 许可双重原因）：

1. **公开权重校验下载**：`python tools/fetch_yolo_weights.py --list` 看已知来源与许可，
   `--url <url> --sha256 <hash>` 带校验下载，`--inspect` 体检；
2. **CCPD 自训练导出**：`tools/ccpd_to_yolo.py` 转数据集 → `tools/train_plate_yolo.py`
   训练并导出 ONNX（需自行装 PyTorch + Ultralytics）；
3. **纯检测模式**：不配识别权重，`yolo` 只给框和单/双层分类，`text` 为空。

放好 `models/yolo/plate_det.onnx` 即自动启用。

**一个必须知道的配置**：`engine.yolo_attr_order`。检测模型输出的属性列顺序有两种流派，
**属性个数完全相同、只看张量形状无法区分**：

| 值 | 布局 | 典型来源 |
|---|---|---|
| `kpt_first`（默认） | `box + obj + kpts + cls` | `we0091234` 系权重（`models/yolo/README.md` 路径 A） |
| `cls_first` | `box + cls + kpts` | Ultralytics `export.py` 导出 |

给错的典型症状是框正常但角点乱飞、或分数恒为 0。判定方法写在 [models/yolo/README.md](models/yolo/README.md)。

### 4.6 `fusion` —— 三合一引擎

把 `cv` / `deep` / `yolo` 编排成一个引擎，三种策略通过 `engine.fusion_strategy` 切换：

| 策略 | 机制 | 什么时候用 |
|---|---|---|
| `ensemble`（默认） | **并联投票融合** —— 所有可用成员各跑一次，落在同一处的框归为一组（IoU ≥ `fusion_group_iou`），组内按 `引擎权重 × 置信度` 投票，票高者胜出 | 想要**最少误检**、不在意耗时 |
| `chain` | **顺序兜底链** —— 按 `fusion_members` 顺序依次尝试，拿到 `confidence ≥ fusion_chain_accept`（默认 0.75）就停 | 想要**快**且能容忍弱成员的垃圾 |
| `auto` | **特征选优** —— 按长边分辨率与模糊度排序后按序试 | 只想**挑一个**引擎跑，但不想写死是哪个 |

结果里带 `meta["fusion"]`，写明每个成员读了什么、谁投了几票 —— **少数派一目了然**。

三条设计细节：

- **成员自动退席**：缺权重或缺依赖的成员少一票，不影响其余成员。
- **准入规则**：孤立候选要凑到 `fusion_min_support`（默认 2）个成员的票才留，
  或者票权达到 `fusion_trusted_weight`（默认 1.0）的成员可以独存。
  为什么不是 1（并集）？实测 `cv` 在 4096×2304 的真实样张上给 9 条候选
  （7 条误检），而 `yolo` 给 2 条全中、`deep` 给 1 条全中 —— 只做并集会让最弱成员
  污染输出。
- **丢候选不静默**：被准入规则滤掉的候选记在 `FusionEngine.last_dropped` 里，
  日志也会列出号码。想取并集（最大召回）把 `fusion_min_support` 设成 1。

**实测：融合目前不提升识别率**（100 张 CCPD 三种策略的严格命中都等于最强单成员
`yolo` 的 84%），因为 `deep` 的命中集合是 `yolo` 的**子集** —— 成员之间没有互补。
它的真实价值是**更少的误检**（ensemble 5 条 vs yolo 7 / deep 14 / cv 691）。
详见 [ACCURACY.md §6](docs/ACCURACY.md)。

### 4.7 怎么选

```text
常规车牌为主（蓝牌/绿牌，场景清晰）        → yolo（若有权重），否则 deep
含港/警/领/学/双层等特殊牌                 → deep
想要最少误检、不在乎 2 秒/张               → fusion(ensemble)
要快、能容忍误检                           → fusion(chain)
只想挑一个但不想写死                       → fusion(auto) + fusion_auto_clear_first
只想快速筛一遍候选、精度不重要              → cv
```

**不要**把默认引擎写死成 `yolo` 或 `deep` 中任何一个：CCPD 常规牌上 `yolo` 领先 5 个百分点
（84% vs 79%），而特殊牌自有样张上 `deep` 全对（10/10 vs yolo 8/10）—— 两套数据**排名反转**。
按你的车牌构成选，或用 `fusion(auto)` 交给配置。

---

## 5. 命令行完整参考

安装后命令是 `lpr`；不安装则用 `python -m lpr`（参数完全一致）。

```text
lpr {predict,video,gui,engines,check,models,config}
```

所有子命令都会**先加载 `config.json`**，命令行参数优先于配置文件（见 [§8.1](#81-加载优先级)）。

### 5.1 `lpr predict` —— 识别图片（单张 / 目录）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `inputs` | 路径…（必填） | — | 图片路径或目录，可给多个 |
| `--engine` | `auto`\|`cv`\|`deep`\|`cascade`\|`yolo`\|`fusion` | 取 `config.engine.default` | 识别引擎 |
| `--recursive` | 开关 | 关 | 输入是目录时递归查找子目录 |
| `--save-annotated [DIR]` | 路径（可省略值） | `annotated` | 保存标注图到指定目录 |
| `--report DIR` | 路径 | — | 输出 JSON + CSV 报告到指定目录 |
| `--json` | 开关 | 关 | 在终端以 JSON 输出 |
| `--mask` | 开关 | 关 | 对输出中的车牌号打码（覆盖 `security.mask_output`） |
| `--workers N` | 整数 | 1 | 并行进程数；`>1` 走多进程，`0` 按 CPU 自动。**多进程结果不含 ROI** |
| `--color` | 字符串 | — | 只保留指定底色，可多次或用逗号分隔。可选：`黑色/蓝色/绿色/白色/黄色/危险品` |

```bash
lpr predict car.jpg --engine deep                 # 单图识别
lpr predict ./photos/ --engine deep --recursive   # 目录批量（递归）
lpr predict ./photos/ --save-annotated out/ann --report out/   # 标注图 + JSON/CSV
lpr predict car.jpg --engine deep --mask --json   # 脱敏 + JSON 输出
lpr predict ./photos/ --color 危险品              # 只保留危险品运输牌
lpr predict ./photos/ --color 黄色,危险品          # 多底色过滤（逗号分隔）
lpr predict ./photos/ --engine deep --workers 8   # 多进程并行（大批量）
```

底色过滤的多个值可以用英文逗号或中文逗号分隔（`,` 与 `，` 都行）。

### 5.2 `lpr video` —— 识别视频或摄像头

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `input` | 字符串（必填） | — | 视频路径，或摄像头编号（如 `0`） |
| `--engine` | 同上 | 取配置 | 识别引擎 |
| `--stride N` | 整数 | 10 | 每 N 帧识别一次（越大越快，越小越不容易漏） |
| `--output PATH` | 路径 | — | 输出标注视频路径（`mp4v` 编码） |
| `--mask` | 开关 | 关 | 输出中打码 |

```bash
lpr video parking.mp4 --stride 10 --output out.mp4
lpr video 0 --engine deep --stride 5
```

借助 `MultiPlateTracker` 做**跨帧跟踪去重**：同一辆车只上报一次，并用多帧投票纠正单帧的偶发错误。输出的每行带 `出现N次` 与 `一致率`。

### 5.3 `lpr gui` —— 启动图形界面

无参数。详见 [§6](#6-图形界面)。

### 5.4 `lpr engines` —— 查看引擎状态

无参数。逐个打印四个引擎的可用性、说明与**不可用原因**。

### 5.5 `lpr check` —— 校验号码是否符合国家规范

| 参数 | 类型 | 说明 |
|---|---|---|
| `plates` | 字符串…（必填） | 待校验的车牌号，可给多个 |
| `--fix` | 开关 | 对不合法的号码尝试自动纠正并列出修改项 |

```bash
lpr check 京A1234危 京AD77972 --fix
```

**完全离线、不需要图片**，可用于清洗一张号码表。退出码：全部合法为 0，有不合法为 2。

### 5.6 `lpr models` —— 查看模型文件信息与完整性校验

| 参数 | 类型 | 说明 |
|---|---|---|
| `--verify` | 开关 | 执行 SHA256 完整性校验 |

校验结果三态，**符号区分**：`✓` 已校验通过 / `—` 未配置哈希白名单故未校验 /
`✗` 不匹配。「未配置白名单」不会显示成「✓ 通过」。

```bash
lpr models --verify
#   svm.dat                    4.38 MB  2026-10-11 23:05:06
#     ✓ 哈希校验通过 (ce3fdffb3e9529fc…)
```

### 5.7 `lpr config` —— 生成 / 查看配置

| 参数 | 类型 | 说明 |
|---|---|---|
| `--path PATH` | 路径 | 写入位置（默认 `config.json`） |
| `--show` | 开关 | 只打印当前生效配置，不写文件 |

### 5.8 退出码

| 码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | `predict` 未找到可处理的图片；`models` 模型目录为空；`doctor` 有引擎不可用 |
| `2` | 参数错误（如未知底色名）；`check` 有号码不合法 |
| `3` | 被安全策略拒绝（`SecurityError`） |
| `130` | 用户 Ctrl+C 中断 |

### 5.9 `--workers` 多进程并行

```bash
lpr predict ./photos/ --engine deep --workers 8    # 8 进程
lpr predict ./photos/ --engine deep --workers 0    # 按可调度 CPU 数自动
```

行为约定：

- **顺序不变**：多进程结果顺序与传入顺序一致，和串行路径完全一样；
- **结果不带 ROI**：车牌 ROI 图像跨进程传输开销远大于收益，`workers > 1` 时
  `roi` 恒为 `None`；需要 ROI 就用 `workers=1`；
- **单条失败被隔离**：某个文件读不出/识别异常只会记一条 warning，不会中断整批；
- **子进程各写各的日志**：`--workers` 时每个子进程写 `logs/lpr-<日期>-p<pid>.log`
  （父进程写主文件）—— 见 §3.8；
- **小批量自动退回串行**：文件数不足以摊平子进程启动 + 模型重新加载的固定成本时
  （见 `lpr/batch.py` 的 `MIN_FILES_PER_WORKER`），即使指定了 `--workers` 也会走串行。

> ⚠️ **先量再开。** 并行收益强依赖机器：本项目的开发/CI 环境 CPU 配额受限
> （`os.cpu_count()` 报 20，但**纯 CPU 计算**用 4 进程实测只有 `0.49x`），
> 在那里并行只会更慢 —— 这不是实现的锅。换到真实多核机器上用基准脚本量一下：
>
> ```bash
> python tools/bench_workers.py --images ./photos --engine deep --workers 1 4 8 --cpu-control
> ```
>
> `--cpu-control` 会同时测一遍纯 CPU 计算的并行扩展性，用来区分"是机器没余量"
> 还是"是识别本身没被并行化"。

### 5.10 日志选项（所有子命令通用）

```bash
lpr predict a.jpg --log-level DEBUG        # 打开调试日志
lpr predict a.jpg --log-dir D:/logs        # 换个日志目录
lpr predict a.jpg --no-log-file            # 只打控制台，不落盘
lpr --log-level DEBUG predict a.jpg        # 写在子命令前面同样生效
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--log-level LEVEL` | 配置文件 / `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`。写别的值会被 argparse 直接拒掉（不会静默按 INFO 处理 —— 那是配置文件的宽松策略） |
| `--log-dir DIR` | 配置文件 / `logs` | 日志目录。传空串等价于关闭文件日志 |
| `--no-log-file` | 关 | 强制不写文件 |

**优先级：命令行 > 环境变量 > `config.json` > 内置默认值**（见 §8.1）。这三个选项在
子命令前后都能写：实现上挂的是同一组 `--log-*`，子命令那份用
`argparse.SUPPRESS` 当默认值，所以不会用 `None` 把前面填好的值覆盖掉。

---

## 6. 图形界面

运行 `lpr gui`。

### 6.1 功能

- **打开图片**（Ctrl+O）/ **批量识别文件夹** / **拖拽导入**（图片或文件夹直接拖进窗口即可识别）
- **摄像头实时识别**：采集与识别在**两个独立线程**里跑 —— 采集线程只管 `cap.read()`
  和刷画面，识别线程始终取"最新一帧"推理。识别慢于采集时不会积压帧，
  画面按摄像头原生帧率刷新
- **引擎下拉框热切换**，不可用的引擎自动置灰并给出安装提示
- **结果区**：号码按国标配色渲染成车牌的样子（蓝/黄/绿/白/黑）、车牌缩略图、
  颜色徽章、一键复制号码；一张图检出多块车牌时**候选表**列出全部（点行回显），
  不再只显示最高分的一条
- **历史行点击回显**该条结果，识别历史可导出 CSV
- **浅色 / 深色双主题**：视图菜单或 Ctrl+1 / Ctrl+2 切换，选择写回 `config.json`
  （`ui.theme`）下次启动沿用
- **「运行日志」面板**：实时显示最近的日志（内存缓冲，最多 200 行），下面一行是
  **日志文件路径**（可选中复制）+「打开日志目录」按钮。启动时窗口标题下方也会打
  一条 `运行日志 → <路径>`
- 窗口缩放后图片自动按新尺寸重绘（去抖 200ms，不做实时重采样）

### 6.2 快捷键

| 快捷键 | 动作 |
|---|---|
| `Ctrl+O` | 打开图片 |
| `Ctrl+E` | 导出历史记录 CSV |
| `Ctrl+Shift+C` | 复制当前识别号码 |
| `Ctrl+1` / `Ctrl+2` | 切换浅色 / 深色主题 |
| 拖放 | 把图片或文件夹拖进窗口即识别 |

### 6.3 两个刻意的设计

- **车牌展示卡不跟主题走**：国标配色是物理属性，蓝牌在深色模式下也得是蓝的。
  主题令牌只管界面控件，不碰车牌皮肤。
- **主题切换不重建控件**：两套令牌共用同一个 `build_qss()`，切主题只是重放一次样式
  表并重放一次徽章状态 —— 卡片投影的效果对象是幂等复用的。

> v2.3.1 修复：此前 `CameraWorker` 号称"采集与识别解耦"，实际是把
> `engine.predict(frame)` 写在采集循环里**同步执行**（deep 引擎约 200ms/张，
> 采集帧率被拖到 ~5fps），而 `_recognizing` 标志只在 `__init__` 赋值一次、
> 从没被读过 —— 是段死逻辑。现在是真的双线程，并有源码级回归护栏。

---

## 7. Python API

### 7.1 `Recognizer` —— 推荐入口

```python
from lpr import Recognizer, PrivacyPolicy

# 统一高层 API：一句话识别
rec = Recognizer(engine="auto")          # auto / cv / deep / cascade / yolo / fusion
outcome = rec.recognize("car.jpg")
print(outcome.best.text, outcome.best.confidence)

# 批量 + 落盘（自动创建父目录）
batch = rec.recognize_many(["a.jpg", "b.jpg"], annotate_dir="out/ann")
batch.save_json("out/report.json", PrivacyPolicy(mask_output=True))
batch.save_csv("out/report.csv")

# 按底色筛选（五种底色 + 危险品）
from lpr import parse_color
batch.keep_colors({parse_color("危险品"), parse_color("黄色")})

# 视频流：跨帧跟踪，同一辆车只上报一次
for plate in rec.recognize_video("parking.mp4", stride=10):
    print(plate.text, plate.meta["hits"], plate.meta["hit_rate"])
```

构造参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `engine` | `"auto"` | `auto` / `cv` / `deep` / `cascade` / `yolo` / `fusion` |
| `config` | `Config.load()` | 全局配置 |
| `limits` | `config.security.to_limits()` | 安全限制；不传则回落配置 |
| `privacy` | `config.security.to_privacy()` | 隐私策略；不传则回落配置 |

方法：

| 方法 | 说明 |
|---|---|
| `recognize(source)` | 单图识别（路径或已解码的 `np.ndarray`）→ `RecognizeOutcome` |
| `recognize_many(sources, annotate_dir=None, workers=1)` | 批量 → `BatchOutcome` |
| `recognize_dir(dir, recursive=False, annotate_dir=None, workers=1)` | 目录 → `BatchOutcome` |
| `recognize_video(source, stride=10, output_video=None, tracker=None)` | 生成器，逐条 yield 新确认的车牌 |
| `warmup()` | 预热：提前加载模型，把首次识别的耗时挪到启动阶段 |
| `describe()` | 当前识别器配置概览（含引擎可用性报告） |
| `engine_report()` | 引擎可用性报告（`describe_engines()` 的薄封装） |

> ⚠️ **两个容易踩的点**：
>
> 1. `Recognizer(engine=...)` 的默认值是字符串 `"auto"`，**不是**
>    `config.engine.default`。想让 API 也用配置里的默认引擎（例如 `fusion`），
>    要显式传 `Recognizer(engine=config.engine.default)`。CLI 与 GUI 都是显式传的，
>    不受影响。
> 2. `engine` 传具体引擎名时**不做可用性替换**：你点名 `yolo` 却拿到 `deep` 的结果
>    比直接报错更难排查，所以权重缺失时会抛 `EngineUnavailable` 并给出放权重的确切路径。
>    只有 `auto` / `cascade` 这两个**策略**才会替你挑一个具体引擎。

### 7.2 结果对象

`RecognizeOutcome`（单图）：

| 成员 | 类型 | 说明 |
|---|---|---|
| `source` | `str` | 输入来源 |
| `results` | `list[PlateResult]` | 全部检出（含多牌） |
| `elapsed_ms` | `float` | 整图耗时 |
| `engine` | `str` | 实际使用的引擎名 |
| `warnings` | `list[str]` | 非致命问题（如被安全策略拒绝） |
| `.best` | `PlateResult \| None` | 最可信的一条（按 `plate_score` 再按置信度排序） |
| `.texts` | `list[str]` | 全部号码 |
| `.to_dict(privacy=None)` | `dict` | 可序列化字典（整棵树过脱敏） |

`BatchOutcome`（批量）：

| 成员 | 说明 |
|---|---|
| `outcomes` | `list[RecognizeOutcome]` |
| `.total` / `.hit` | 总张数 / 有命中的张数 |
| `.elapsed_ms` | 总耗时 |
| `.all_texts` | 去重后的号码（按出现次数降序） |
| `.keep_colors(colors)` | 就地按底色过滤，返回 self 便于链式调用 |
| `.color_summary()` | 各底色计数 |
| `.save_json(path, privacy=None)` | 写 JSON 报告 |
| `.save_csv(path, privacy=None)` | 写 CSV 报告 |

`PlateResult`（单个车牌）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | `str` | 号码，如 `"京A12345"`；识别失败时为空串 |
| `color` | `PlateColor` | 底色枚举（`black/blue/green/white/yellow/hazmat/unknown`） |
| `layout` | `PlateLayout` | 版式（`single/double/unknown`） |
| `bbox` | `tuple[int,int,int,int] \| None` | 原图中的 `(x, y, w, h)` |
| `roi` | `np.ndarray \| None` | 车牌区域图像（多进程模式下恒为 `None`） |
| `confidence` | `float` | 整体置信度 0~1 |
| `engine` | `str` | 产生该结果的引擎名 |
| `char_confidences` | `list[float]` | 字符级置信度，用于定位是哪个字错了 |
| `meta` | `dict` | 引擎附加信息（耗时、候选数、`fusion` 明细等） |
| `.ok` | `bool` | 是否成功识别出号码 |

### 7.3 底层引擎接口

```python
from lpr import get_engine, describe_engines, available_engines, setup_logging

setup_logging("INFO")
engine = get_engine("deep")
for r in engine.predict("car.jpg"):
    print(r.text, r.color.label, r.confidence, r.bbox)

# 引擎清单与可用性（GUI 的引擎下拉框就是用它填的）
for row in describe_engines():
    print(row["name"], row["display_name"], row["available"], row.get("reason", ""))
```

引擎是**按 `(name, config fingerprint)` 共享的单例**，推理通过 `INFERENCE_LOCK` 串行化
（批量识别与摄像头识别可能来自不同线程，不加锁会互相踩内存/抢 CPU）。

### 7.4 规则与跟踪

```python
from lpr import is_valid, correct, describe_plate, plate_score
from lpr import MultiPlateTracker, merge_same_image

is_valid("京AD77972")              # True
correct("京AD77g72")               # ('京AD77972', ['第7位 G→9 ...'])
describe_plate("京A1234危")        # {'type': '危险品运输', 'province': '京', ...}
plate_score("京AD77972")           # 0~1 的"像不像真车牌"评分

merge_same_image(results)          # 同图去重：同号合并、位置近但号码异则都留
tracker = MultiPlateTracker()
for plate in tracker.update(engine.predict(frame)):
    ...                            # 跨帧投票后的新确认车牌
```

### 7.5 输出文件格式

**JSON**（`save_json`）：

```json
{
  "generated_at": "2026-10-16 15:30:00",
  "total": 12,
  "hit": 11,
  "hit_rate": 0.9167,
  "elapsed_ms": 2480.5,
  "unique_plates": ["京AD77972", "京A1234危"],
  "items": [
    {
      "source": "car.jpg",
      "engine": "deep",
      "elapsed_ms": 194.2,
      "count": 1,
      "warnings": [],
      "plates": [
        {
          "text": "京AD77972",
          "color": "green",
          "color_label": "绿牌（新能源）",
          "layout": "single",
          "layout_label": "单行",
          "bbox": [139, 499, 404, 108],
          "confidence": 0.91,
          "engine": "deep",
          "char_confidences": [0.99, 0.97, 0.93, ...],
          "meta": {},
          "plate_type": "新能源"
        }
      ]
    }
  ]
}
```

**CSV**（`save_csv`）列：`来源, 号码, 颜色, 类型, 置信度, 引擎, bbox, 耗时ms`。
用 `utf-8-sig` 写出，Excel 直接双击不乱码。

> 开启 `mask_output` 时**整棵树**都会被脱敏，包括 `source` 路径（文件名里可能嵌着号码）
> 和 `unique_plates` 汇总字段 —— 只处理 `text` 会留下后门，见 [§9](#9-安全与隐私)。

### 7.6 包级导出

`from lpr import ...` 可拿到：`Recognizer` / `RecognizeOutcome` / `BatchOutcome`、
`Config` / `SecurityConfig` / `Limits` / `PrivacyPolicy` / `SecurityContext` / `SecurityError`、
`PlateResult` / `PlateColor` / `COLOR_NAMES` / `parse_color`、
`get_engine` / `describe_engines` / `available_engines` / `register_engine` / `CVEngine` / `DeepEngine`、
`is_valid` / `correct` / `describe_plate` / `plate_score`、
`MultiPlateTracker` / `merge_same_image` / `mask_plate` / `verify_model` / `setup_logging` / `__version__`。

本包声明了 `py.typed`，**带类型标注**，下游可直接依赖。

---

## 8. 配置参考（config.json 全字段）

### 8.1 加载优先级

```text
命令行参数  >  环境变量（LPR_*）  >  config.json  >  代码内默认值
```

- `config.json` 在**项目根目录**，不存在也能跑（全用默认值）；
- 支持**点号路径的浅层覆盖**，例如 `{"engine": {"locate": {"max_side": 1600}}}`；
- 未知键会被**静默忽略**（不会因为配置里多了个字段就启动失败）；
- `lpr config` 生成默认配置，`lpr config --show` 只打印不落盘；
- 生成配置文件时会**自动备份**旧文件为 `config.json.bak`。

### 8.2 `engine.*` —— 引擎选择

| 字段 | 默认 | 取值 | 说明 |
|---|---|---|---|
| `default` | `"cascade"` | `cv` / `deep` / `cascade` / `yolo` / `fusion` | 默认引擎。`cascade` 是 `deep` 的别名 |
| `mode` | `"single"` | `single` / `cascade` | **跨引擎兜底**：`cascade` 时主引擎返回空会自动再试另一个可用引擎 |
| `use_confidence` | `true` | 布尔 | CV 引擎字符识别是否启用 SVM 置信度加权投票 |

`default` 与 `mode` 是**正交**的两个维度：前者选引擎*对象*，后者决定要不要跨引擎兜底。

### 8.3 `engine.locate.*` —— 定位参数

所有长度量纲都是**比例**（相对图宽/图高）而不是像素，这是同一套参数能适配 640px～4000px 输入的原因。

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_side` | `1200` | 送进算法前把长边缩放到不超过该值，兼顾速度与精度 |
| `blur_kernel` | `3` | 高斯模糊核（**奇数**，`0` 表示跳过） |
| `open_kernel_ratio` | `0.02` | 开运算核大小，相对**图宽**比例 |
| `morph_kernel_ratio` | `(0.005, 0.021)` | 闭/开运算核 `(高, 宽)`，相对**图高 / 图宽**比例 |
| `canny_low` / `canny_high` | `100` / `200` | Canny 双阈值 |
| `min_area_ratio` | `0.0015` | 候选矩形的最小面积（占全图面积比例） |
| `min_aspect` / `max_aspect` | `2.0` / `5.5` | 单行候选的长宽比范围 |
| `double_min_aspect` | `1.3` | 双行牌召回下限。落在 `[1.3, 2.0)` 的轮廓不直接丢弃，经"强底色"闸门后进独立候选池 |
| `double_color_ratio` | `0.20` | 双行牌区间候选的底色强度下限（核心区内单色占比） |
| `double_max_candidates` | `2` | 双行牌候选的**独立配额**，不挤占常规池名额 |
| `color_ratio` | `0.5` | 底色判定基准（**不是**绝对过半门槛，见下） |
| `color_margin` | `0.04` | 胜出颜色与第二名的最小**占比差**（相对胜出判定） |
| `color_row_ratio_green` | `0.5` | 绿牌渐变严重，横向阈值单独放宽 |
| `color_edge_ratio` | `0.15` | 精定位裁剪边界时行/列的边缘占比下限 |
| `detect_hazmat` | `true` | 是否启用危险品运输牌识别（需一次二值化极性分析） |
| `hazmat_bg_ratio` | `0.45` | 危险品判定：黑色主体像素占比下限 |
| `hazmat_fg_ratio` | `0.03` | 危险品判定：黄色前景像素占比下限 |
| `hazmat_fg_share` | `0.06` | 危险品判定：黄/(黄+黑) 比值下限 |
| `hazmat_white_ratio` | `0.7` | 白色像素达黄色的该倍数时判定为白字（排除黑底白字牌） |
| `box_padding` | `5` | 候选矩形外扩像素，防止车牌边缘被裁掉 |
| `min_chars` | `6` | 字符分割后要求的最少字符数 |
| `max_candidates` | `5` | 最多保留多少个候选车牌进入字符识别 |

三个**反直觉**的字段，值得单独说明：

- **`color_ratio` 不再是"绝对过半"门槛**。真实车牌上字符占了大量面积，底色占比通常
  只有 10%~30%，用 0.5 做绝对门槛会导致大量车牌被判成未知。它现在只作为**自适应下限的基准**。
- **`color_edge_ratio` 从 0.8 降到 0.15**。旧实现用绝对底色阈值定边界，只能罩住"纯底色"
  的行列；车牌**末位字符紧贴牌缘**时，穿过该字符的列底色占比会被笔画咬到 0.3~0.7，
  达不到 0.8 就被当边界裁掉 —— **整字丢失**（实测 `皖AN9491`→`皖AN949`）。
- **`hazmat_white_ratio` 是为了排除黑底白字牌**。真危险品牌上黄字是绝对前景、白色像素≈0；
  实测误判的黑底白字牌 white/yellow ≈ 1.0~3.2，取 0.7 隔开。

### 8.4 `engine.*` —— 识别与融合

| 字段 | 默认 | 取值 | 说明 |
|---|---|---|---|
| `secondary_rec_model` | `""` | 路径 | 第二识别模型（ONNX）路径，用于**低置信交叉验证**。空 = 关闭 |
| `secondary_vote_weight` | `0.8` | `(0, 1]` | 第二模型读数的票权折扣 |
| `yolo_attr_order` | `"kpt_first"` | `kpt_first` / `cls_first` | yolo 检测输出的属性列顺序（见 [§4.5](#45-yolo--检测识别引擎)） |
| `yolo_recognize_backend` | `"auto"` | `auto` / `crnn` / `deep` / `none` | yolo 的字符识别后端。`auto` = 优先自带 `plate_rec.onnx`，没有则复用 deep 的 OCR |
| `yolo_models_dir` | `""` | 路径 | yolo 权重目录。空 = `<项目根>/models/yolo` |
| `fusion_strategy` | `"ensemble"` | `ensemble` / `chain` / `auto` | 融合策略 |
| `fusion_members` | `("yolo","deep","cv")` | 成员元组 | 融合成员及顺序（顺序即 `chain` 的尝试顺序）。只能填具体引擎 |
| `fusion_engine_weights` | `{"yolo":1.0,"deep":1.0,"cv":0.6}` | 字典 | 各成员票权（仅 `ensemble` 用） |
| `fusion_group_iou` | `0.4` | `(0, 1]` | 归为一组的 IoU 阈值 |
| `fusion_chain_accept` | `0.75` | `[0, 1]` | `chain` 的"够可信"门槛 |
| `fusion_auto_clear_first` | `"deep"` | 成员名 | `auto` 在**清晰大图**上先试哪个成员 |
| `fusion_min_support` | `2` | `≥1` | 融合结果要留下所需的最少一致成员数 |
| `fusion_trusted_weight` | `1.0` | `≥0` | 单成员独存所需的票权阈值 |

**`secondary_rec_model` 为什么默认关闭**：100 张 CCPD 实测开启后严格命中**一张未变**
（失败集合逐张相同），`deep` 耗时却 ×2.9（716→2042 ms）。机制没问题（只在该失败的路径上激活，
零回归），问题是它救不回任何东西 —— 这些失败不是"低置信"而是"高置信地错"。
详见 [ACCURACY.md §5.1](docs/ACCURACY.md)。

**`fusion_auto_clear_first` 为什么留成配置而不是写死**：两套实测数据指向相反
（CCPD 要 `yolo`，特殊牌样张要 `deep`），写死任何一边都会在另一边倒退。

### 8.5 `ui.*` —— 界面

| 字段 | 默认 | 说明 |
|---|---|---|
| `window_title` | `"车牌识别系统"` | 窗口标题。**不带版本号**（版本号在状态栏显示，写进标题必然漂移） |
| `window_size` | `(1280, 800)` | 初始窗口尺寸 |
| `camera_interval` | `0.35` | 摄像头识别间隔（秒），原版固定 2 秒 |
| `camera_resolution` | `(1280, 720)` | 摄像头采集分辨率 |
| `theme` | `"light"` | 主题：`light` / `dark`（非法值回落 `light`） |
| `max_history` | `200` | 历史记录最多保留多少条 |

### 8.6 `security.*` —— 安全与资源限制

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_file_mb` | `32.0` | 单文件大小上限（MB） |
| `max_pixels` | `50000000` | 解码后像素总数上限（防解压炸弹） |
| `max_side` | `20000` | 图像单边长度上限（px） |
| `max_batch_files` | `5000` | 单次批量最大文件数（超过会截断并记 warning） |
| `mask_output` | `false` | 输出中是否对车牌号打码 |
| `mask_keep_tail` | `3` | 打码保留的尾部字符数 |
| `mask_logs` | `true` | 日志中是否自动脱敏车牌号（强烈建议保持 `true`） |
| `keep_roi` | `true` | 结果中是否保留车牌 ROI 图像 |
| `predict_timeout` | `30.0` | 单张识别超时（秒），`0` 为不限制 |
| `max_qps` | `20.0` | 每秒最多识别次数（令牌桶限流） |
| `model_hashes` | 内置两个 SVM 哈希 | 模型哈希白名单 `{文件名: sha256}`，供 `lpr models --verify` 使用 |

### 8.7 环境变量

| 变量 | 作用 |
|---|---|
| `LPR_ENGINE` | 覆盖 `engine.default`（**优先级高于 config.json**） |
| `LPR_LOG_LEVEL` | 覆盖 `log_level` |
| `LPR_LOG_DIR` | 覆盖 `log_dir` |
| `LPR_LOG_FILE` | `0` / `false` / `off` → 关闭文件日志；`1` / `true` / `on` → 强制开启 |
| `LPR_MASK_OUTPUT` | `1` / `true` / `yes` → 打开输出脱敏 |
| `LPR_PYTHON` | 启动器用：指定解释器，跳过搜索 |
| `LPR_NO_VENV` | 启动器用：`1` 时不建 `.venv`，直接装进找到的解释器 |

### 8.8 日志（`log_level` / `log_to_file` / `log_dir` / `log_max_mb` / `log_backups`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `log_level` | `"INFO"` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`。名字写错时**按 INFO 处理并发一条警告**（静默降级会让人以为参数没生效）；命令行 `--log-level` 走 argparse 的 `choices`，写错直接被拒 |
| `log_to_file` | `true` | 是否把运行日志写入文件。关掉就只剩控制台输出，**关窗即丢** |
| `log_dir` | `"logs"` | 日志目录，相对**当前工作目录**（不是包安装目录）；空串同样表示不写文件 |
| `log_max_mb` | `5` | 单文件上限（MB），超出后轮转为 `*.log.1`…。下限 1 |
| `log_backups` | `5` | 保留的历史份数，`0` 表示不保留 |

日志的三个出口（控制台 / 文件 / 界面内存缓冲）与文件命名见 §3.8。手改
`config.json` 时的宽松解析：`"log_to_file": "false"` 当假处理，`log_max_mb: 0`
夹到 1，`log_backups: "abc"` 保持默认 5 —— 日志设置不值得让配置加载失败。

---

## 9. 安全与隐私

`lpr.utils.security` 提供完整的输入防护与隐私脱敏：

| 模块 | 能力 |
|---|---|
| `Limits` | 文件大小上限（32MB）、像素上限（5000 万）、单边上限（20000px）、最小尺寸、批量文件数上限、扩展名白名单 → 防解压炸弹与畸形尺寸 |
| `check_file` / `check_image` | 读取前校验文件与解码后数组，异常抛 `SecurityError` |
| `PrivacyPolicy` | 输出/日志脱敏策略（`mask_output`、`mask_keep_tail`、`mask_logs`、`keep_roi`） |
| `mask_plate` / `scrub_value` | 单值打码 + **递归清洗任意嵌套结构**（`meta.variants` 这类旁路字段也会被打掉） |
| `MaskingFilter` | 日志过滤器，日志里的车牌号自动打码 |
| `RateLimiter` | 令牌桶限流，防并发识别打满机器 |
| `verify_model` / `check_model` / `sha256_file` | 模型哈希白名单校验，检测模型篡改。**三态结果**（`ok` / `skipped` / `mismatch`），"未配置白名单"不会再显示成"✓ 通过" |
| `safe_output_path` | 防路径穿越（`../../etc/passwd` 这类会被拦回基准目录） |
| `anonymize_path` | 路径脱敏（保留尾部若干级） |

**脱敏是端到端生效的**：控制台输出、JSON、CSV、嵌套 `meta`、连 `source` 路径名都覆盖，均有单元测试守住。

### 9.1 安全层覆盖三个入口（v2.3.1 修复）

| 入口 | 输入校验 | 输出脱敏 | 日志脱敏 | 模型校验 |
|---|---|---|---|---|
| `lpr` CLI | ✅ | ✅ | ✅ | ✅ |
| Python API（`Recognizer`） | ✅ | ✅ | ✅ | — |
| 图形界面（`lpr gui`） | ✅ | ✅ | ✅ | — |

> v2.3.1 之前，Python API 会**静默忽略** `config.json` 的 `security` 段
> （`LPR_MASK_OUTPUT` 也对 API 无效），图形界面则**完全没有接入安全层**
> —— 图片不校验、CSV 导出裸写明文。现在三条入口共用同一份 `config.security`。

「日志脱敏」由 `MaskingFilter` 挂在 `lpr` 命名空间的 logger 上，因此
**新增的日志文件出口同样受它管辖**（v2.7.3 起日志会落盘，如果只有 stderr
脱敏、文件不脱敏，那等于开了个后门）。`lpr.ui`、`lpr.engines.*` 等子 logger
也各挂一份，见 `tests/test_logging_setup.py::TestMasking`。

### 9.2 模型完整性校验

白名单开箱可用（`config.json` 的 `security.model_hashes` 已内置随仓库分发的两个 SVM 模型哈希）。
**换自训练模型后必须同步更新**，否则会被判为不匹配。

```bash
lpr models --verify
#   svm.dat                    4.38 MB  2026-10-11 23:05:06
#     ✓ 哈希校验通过 (ce3fdffb3e9529fc…)
# （✓=已校验通过，—=未配置哈希白名单故未校验，✗=不匹配）
```

### 9.3 隐私用法示例

```bash
lpr predict car.jpg --engine deep --mask          # 控制台输出打码
lpr predict ./photos/ --engine deep --mask --report out/   # 报告落盘同样打码
```

---

## 10. 架构与算法

### 10.1 分层

```text
┌──────────────────────────────────────────────────────────────┐
│ 入口层   lpr.cli (7 个子命令)   lpr.ui.window (PySide6)      │
│ 门面层   lpr.api.Recognizer                                  │
│          引擎解析 / 级联兜底 / 安全校验 / 脱敏 / 结果导出    │
│          lpr.batch (多进程分片与按序拼回)                    │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ 引擎层   registry (注册表 + 单例 + INFERENCE_LOCK)           │
│   cv_engine  deep_engine  yolo_engine  fusion_engine         │
│   └─ locate.py / segment.py / svm_classifier.py              │
│   统一契约：BaseEngine -> list[PlateResult]  (types.py)      │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ 领域层   plate_rules (格式/纠错/评分)   tracker (跨帧跟踪)   │
│          config (参数)   utils.image (IO/标注)               │
│          utils.security (限制/脱敏/校验)                     │
└──────────────────────────────────────────────────────────────┘
```

### 10.2 单图识别的调用链

```text
Recognizer.recognize(source)
  ├─ _load()                      check_file → imread(中文路径/EXIF) → check_image
  ├─ _predict()                   with INFERENCE_LOCK:
  │    ├─ engine.predict(img)     ← 四引擎共用的入口
  │    │    ├─ CVEngine:   locate → segment → SVM → 投票
  │    │    ├─ DeepEngine: locate → 纯识别(跳检测网络) → 多尺度投票 → 补救预处理
  │    │    ├─ YoloEngine: letterbox → ONNX 前向 → 解码 → NMS → 逆变换 → 透视矫正 → 识别
  │    │    └─ FusionEngine: 按策略编排上面几个
  │    └─ 若 mode=cascade 且无结果 → 兜底引擎再试一次
  ├─ merge_same_image()           同图去重（同号合并、位置近但号码异则都留）
  └─ privacy.keep_roi=False 时清空 roi
```

### 10.3 定位

传统路径（`cv` / `deep` 共用 `locate.py`）：

1. 长边缩放到 `max_side` → 灰度 → 高斯模糊 → Canny 边缘；
2. 形态学闭运算把字符连成块（核大小按**图宽/图高比例**算）；
3. 找轮廓 → 按面积/长宽比筛候选 → **双行牌单独一个候选池**（用 `double_min_aspect` +
   底色强度闸门放行，独立配额，不挤占常规池）；
4. 底色判定：HSV 多区间掩膜（蓝/绿/黄/白/黑各自一组 HSV 上下界，红色还有跨越
   H 环的两段区间）+ **相对胜出**判定（胜出色与第二名的占比差要 ≥ `color_margin`）；
   危险品牌另外做一次**明暗极性分析**（它是黑底黄字，与普通黄牌底色相反，
   仅看"有没有黄色"无法区分）；
5. 精定位裁剪：按 `color_edge_ratio` 逐行逐列收缩边界（保住紧贴牌缘的尾字）；
6. 四点透视矫正。

检测路径（`yolo`）：一次前向直接给出 bbox + 四角点 + 单/双层分类，角点顺序
固定为「左上→右上→右下→左下」，再做透视变换 —— 不依赖轮廓，所以侧倾/模糊下召回更好。

### 10.4 矫正几何

角点排序在侧倾 >45° 时曾把车牌矫正成**竖图**（v2.3.4 修复）。现在 ±89° 全范围都能正确归位，
按 GA 36-2018 /《警车管理规定》核对了三处号牌规范口径。相关测试有 206 项
（`tests/test_quad_ordering.py`），是本仓库最大的单个测试文件。

> 注意区分**矫正几何**与**定位召回**：矫正只在"找到了候选"之后才起作用。
> 整景照片里侧倾 >45° 的车牌可能根本找不到候选 —— 这是定位层的问题，不是几何问题。

### 10.5 字符分割（传统 CV 路径）

投影波峰切分 + 动态对齐到 7 字符：若切出来的段数不对，按波峰高度与间距重新对齐，
而不是硬按等宽切。单行 `min_chars` 默认 6（留一位容错）。

### 10.6 识别、融合与纠错

`deep` / `yolo` 的识别侧共用同一套管线：

1. **纯识别加速**：跳过 DBNet 文本检测网络，把整块 ROI 直接当文本框送进识别网络；
2. **多尺度投票**：超分（目标字符高 48px，倍数 3.0/2.0/4.0）× 锐化（USM 0.0/0.8）
   组合，票重 = `OCR 置信度 × 语法合法性`；
3. **早停**：首个尺度就得到合法号码且票数过半即停（典型只跑 1～2 轮）；
4. **ROVER 风格读数融合**：多轮读数按编辑距离对齐后逐字符投票；
5. **语法纠错**：`plate_rules` 修正 I/O 不合法、发牌机关必须是字母、末位降级等；
6. **补救预处理**（v2.7.2，仅在上述全部落空时触发）：
   依次用「非局部均值去噪 / CLAHE 对比度均衡 / OTSU 二值化」各配一档 3× 放大重试，
   拿到合法读数即停。每图预算 `_RESCUE_BUDGET = 2` 个候选。

第 6 步的来历值得一读：主识别计划的 6 轮**只换尺度**，从不换**增强方式** ——
对"噪点把 U 的底部糊成 N"这类失败无能为力（`皖AUF027` 被稳定读成 `皖ANF027`，
只有去噪能读对）。预算也是实测逼出来的：没有预算时 `deep` 每图要空跑 5 个候选 × 3 变体，
耗时 716→1344 ms 而**命中一张没多**。详见 [ACCURACY.md §5.3](docs/ACCURACY.md)。

### 10.7 跨帧跟踪

`MultiPlateTracker` 用 IoU 关联帧间目标，同一轨迹内按号码一致性投票：

- 同一辆车**只上报一次**（首次确认即 yield）；
- 多帧投票纠正单帧偶发错误；
- 结果带 `meta["hits"]`（出现次数）与 `meta["hit_rate"]`（一致率）。

`merge_same_image` 处理**单图内**的去重：同一号码无论位置都合并为一条观测；
位置相近但号码不同则**同时保留**（不漏报）。

### 10.8 多进程批量

`lpr.batch` 把文件按 `MIN_FILES_PER_WORKER` 分片丢给子进程，配置通过
`Config.to_payload()` 转成纯字典传递（避免 dataclass 在子进程里构造参数不一致），
结果按 `source` 拼回原序。文件数不足以摊平启动成本时自动退回串行。

---

## 11. 准确率实测

100 张 CCPD（720×1160 整景）+ 10 张自有样张（特殊牌为主），六个配置横向对比。
**完整报告与复现命令见 [docs/ACCURACY.md](docs/ACCURACY.md)**，这里只放主表：

| 配置 | 严格命中 | 字符级 | 误检 | 定位命中 | 定位精确率 | 平均IoU | 中位ms |
|---|---|---|---|---|---|---|---|
| `cv` | **0/100** | 18.9% | 691 | 76.0% | 16.6% | 0.643 | 36 |
| `deep` | 79/100 | 87.9% | 14 | 82.0% | 88.5% | 0.670 | 716 |
| **`yolo`** | **85/100** | **93.6%** | 9 | **100.0%** | 98.0% | **0.878** | **125** |
| `fusion(ensemble)` | 84/100 | 87.4% | **5** | 92.0% | 97.9% | 0.805 | 1737 |
| `fusion(chain)` | 84/100 | 92.1%⚠️ | 38 | 96.0% | 76.8% | 0.840 | 190 |
| `fusion(auto)` 默认 | 80/100 | 89.9% | 15 | 92.0% | 89.6% | 0.764 | 749 |
| `fusion(auto)`@yolo | **84/100** | 90.7% | 7 | **100.0%** | 98.0% | **0.878** | 146 |

⚠️ `chain` 的字符级"最高"是**假象**：它兜底时会带出更多候选，而字符级指标
**奖励"多输出候选"**（一个输出 10 条垃圾的引擎，只要其中一条碰巧只差 1 个字符，
这项指标就变好）。它的严格命中与 `yolo` 持平，误检却是 `yolo` 的 5 倍。

五个反直觉的结论：

1. **没有全能引擎** —— 自有样张（特殊牌）上 `deep` 10/10 而 `yolo` 8/10，**排名反转**；
2. **融合目前不提升识别率** —— 三种策略的严格命中都等于最强单成员，因为 `deep` 的
   命中集合是 `yolo` 的**子集**（完美路由上界 = 84% = `yolo` 单独）；融合的价值是**降误检**；
3. **识别瓶颈在定位/裁剪** —— `yolo` 与 `deep` 用的是**同一个识别器**，
   `yolo` 的领先全部来自更好的框（平均 IoU 0.878 vs 0.670）；
4. **第二识别模型零收益** —— 失败集合逐张相同，耗时 ×2.9；
5. **`cv` 是 0/100 但定位是对的** —— 它在真牌区上把每个字都读错，且置信度饱和在 1.000。

---

## 12. 项目结构

```text
src/lpr/
├── types.py            # PlateResult / PlateColor / PlateLayout 统一数据契约
├── config.py           # 集中配置（JSON + 环境变量，含 SecurityConfig）
├── api.py              # 高层 API：Recognizer / RecognizeOutcome / BatchOutcome
├── batch.py            # 多进程批量识别（分片、按序拼回、小批量退回串行）
├── plate_rules.py      # 车牌语法规则：格式识别、纠错、合法性评分
├── tracker.py          # 跨帧多车牌跟踪（IoU + 号码一致性投票）
├── cli.py              # 命令行入口
├── __main__.py         # python -m lpr 入口（无参数直接开界面）
├── py.typed            # 声明本包带类型标注（下游可依赖）
├── engines/
│   ├── base.py         # 引擎抽象接口
│   ├── registry.py     # 引擎注册表 + 推理串行锁 INFERENCE_LOCK
│   ├── locate.py       # 车牌定位（向量化、透视矫正、颜色判定）
│   ├── segment.py      # 字符分割（投影波峰 + 动态对齐到 7 字符）
│   ├── svm_classifier.py  # SVM 字符分类（兼容原版模型 + 置信度）
│   ├── cv_engine.py    # 传统 CV 引擎
│   ├── deep_engine.py  # 深度学习引擎（定位引导 + 纯识别加速 + 双行牌 + 补救预处理）
│   ├── yolo_engine.py  # 检测识别引擎（bbox + 四角点 + 单/双层，ONNX）
│   └── fusion_engine.py  # 三合一引擎（并联投票 / 顺序兜底链 / 特征选优）
├── ui/
│   ├── window.py       # PySide6 主窗口（已接入安全层，浅色/深色双主题）
│   └── worker.py       # 采集与识别分离的线程（CameraWorker + _RecognizeThread）
└── utils/
    ├── image.py        # 图像 IO（中文路径 / EXIF）、标注绘制（Pillow 中文字体）
    ├── security.py     # 安全层：Limits / PrivacyPolicy / 脱敏 / 限流 / 模型校验三态
    └── logging_setup.py
tools/
├── launcher.py               # 启动器：找解释器 / 建 .venv / 装依赖 / 权重与引擎自检 / 启动（只依赖标准库）
├── no-python.txt             # 找不到 Python 时的中文提示（UTF-8，由 start.bat 用 type 输出）
├── benchmark.py              # 引擎性能基准（耗时/QPS/命中率）
├── bench_workers.py          # 多进程批量加速基准（含纯 CPU 并行对照）
├── shot.py                   # 界面离屏渲染截图（美化前后对比 / 人工核验）
├── eval_locate.py            # 定位客观评测 + 网格搜索（--save-baseline/--baseline）
├── eval_weak.py              # 无标注真实图的弱监督评测（合法率/候选率，--override 调参对照）
├── eval_ccpd.py              # 多引擎准确率评测（识别+定位、失败四分类，--engines / fusion:策略）
├── eval_report.py            # 把 eval_ccpd 的 JSON 排成 markdown 表（总览/分子集/失败分类/互补性）
├── tune_thresholds.py        # 无监督阈值调参（没有标注时的退路）
├── tune_io.py                # 调参结果落盘（写进 engine.locate，自动备份）
├── fetch_ccpd.py             # 稳健下载 CCPD 样张
├── fetch_hg.py               # 下载带 YOLO 标注的评测集（支持 --limit 小规模验证）
├── fetch_yolo_weights.py     # 取检测权重：--list 看来源 / --url 带 SHA256 校验下载 / --inspect 体检
├── ccpd_labels.py            # CCPD 文件名标注解析（兼容 & 与 , 两种坐标分隔符）
├── ccpd_to_yolo.py           # CCPD → YOLO-pose 数据集转换（角点重排 + 单/双层分类 + 确定性划分）
├── train_plate_yolo.py       # 训练 YOLO-pose 车牌检测并导出 ONNX（--export-only 可跳过训练）
├── make_stub_onnx.py         # 生成**常量输出**的桩 ONNX 模型，无第三方权重也能自检整条管道
├── make_color_samples.py     # 合成六色样张，核对底色/危险品判定
├── make_hard_samples.py      # 生成确定性对抗退化样张（模糊/降采样/曝光/噪点/倾斜/压缩）
└── make_regression_baseline.py  # 生成真实样张行为快照基线
models/                 # 预训练 SVM 模型（原版权重，config.json 内置哈希）
models/yolo/            # 检测识别引擎权重放置目录（自带说明，权重需自行准备）
samples/  img/          # 真实测试样张（回归基线与准确率门槛的数据来源）
                        #   img/ 同时被当下载样本的落图目录用，403 张 / 75.7MB —— §3.7
logs/                   # 运行日志（自动创建；主进程 + 每个子进程一个文件）—— §3.8
start.bat               # Windows 一键启动（纯 ASCII 引导壳 → tools/launcher.py → 界面）
                        #   加 doctor 参数为只读环境体检
tests/                  # 1195 项 pytest 测试（含 tests/baselines/ 行为基线）
.github/workflows/ci.yml   # CI：ruff + mypy + 多版本 pytest
docs/UPGRADE.md         # 与原项目的详细对比（逐版本）
docs/ACCURACY.md        # 准确率实测与提升路线（v2.7.0，v2.7.2 增补）
docs/ROADMAP.md         # 待完善清单（收益/风险排序 + 归档）
docs/THIRD_PARTY.md     # 第三方依赖许可证清单
docs/screenshot.png     # 界面截图（README 顶部引用）
```

---

## 13. 测试、评测与 CI

### 13.1 跑测试

```bash
pip install -e ".[dev,gui]"                         # 测试需要 Qt（GUI 用例）
pytest tests/ -q                                    # 1195 项单元/集成测试
pytest tests/ -q -m "not slow"                      # 跳过会起多进程的较重用例
pytest tests/ -q --timeout=300                       # 建议加超时（CI 就是这么跑的）
ruff check src tools tests && mypy                  # 静态检查
```

### 13.2 测试规模（1195 项，按文件）

| 测试文件 | 项数 | 覆盖 |
|---|---|---|
| `test_quad_ordering.py` | 206 | 角点排序与透视矫正（±89° 全范围） |
| `test_fusion_engine.py` | 83 | 融合三策略、准入规则、成员退席、auto 路由 |
| `test_special_plates.py` | 81 | 军/警/使领馆/武警等特种号牌语法与判色 |
| `test_yolo_engine.py` | 78 | letterbox / 解码 / NMS / 逆变换 / 角点矫正 |
| `test_launch_env.py` | 73 | 启动器：资产清单、引擎判定、doctor 只读性、预检、日志目录 |
| `test_logging_setup.py` | 60 | 运行日志：落盘与轮转、幂等、配置/命令行接线、界面日志面板 |
| `test_plate_color.py` | 53 | 六色判定 + 危险品牌极性分析 |
| `test_twelve_kinds.py` | 53 | 13 类号牌格式逐条 |
| `test_v2_3_quality.py` | 49 | v2.3 质量专项（召回分层、形近字纠错） |
| `test_eval_ccpd.py` | 47 | 评测器本身（分词、taxonomy 键序、JSON 序列化） |
| `test_plate_rules.py` | 47 | 格式识别 / 纠错 / 评分 |
| `test_security.py` | 40 | 限制、脱敏、限流、模型三态、路径穿越 |
| `test_v2_3_1_fixes.py` | 37 | 三个真 Bug 的回归护栏 |
| `test_tools_tuning.py` | 37 | 调参工具与基线生成 |
| `test_ccpd_to_yolo.py` | 35 | 数据集转换（角点重排、单/双层、划分确定性） |
| `test_batch_parallel.py` | 31 | 多进程：顺序、隔离、ROI 取舍、退回串行、子进程日志 |
| `test_deep_preprocess.py` | 30 | 多尺度放大 / 锐化 / 早停 |
| `test_regression_baseline.py` | 22 | 真实样张定位行为快照 |
| `test_tracker.py` | 22 | 跨帧跟踪与同图去重 |
| `test_eval_weak.py` | 18 | 弱监督评测 |
| `test_api.py` | 17 | 高层 API 与导出 |
| `test_rescue_preprocess.py` | 16 | 补救预处理（纯函数契约 + 零回归 + 预算） |
| `test_config_and_types.py` | 13 | 配置校验与数据契约 |
| `test_deep_accuracy.py` | 13 | deep 引擎准确率门槛 |
| `test_camera_probe.py` | 8 | 摄像头设备枚举与排查提示 |
| `test_hard_samples_accuracy.py` | 8 | 对抗退化样张（模糊/噪点/倾斜…） |
| `test_segment.py` | 7 | 字符分割 |
| `test_secondary_cross_validation.py` | 6 | 第二识别模型交叉验证接线 |
| `test_engines.py` | 3 | 引擎注册表 |
| `test_real_samples_accuracy.py` | 2 | 真实样张准确率门槛（CI 拦截） |

### 13.3 为什么 CI 里没有准确率门槛

带标注的真实数据集不适合进仓库（体积 + 版权）：

- CCPD 的标注在镜像里被抹掉；
- 公开 YOLO 集只有 320 张。

我们试过用**合成图**替代 —— 不可行：合成牌只有放大到占画面一半宽才勉强被定位到，
且找到的是字符区而不是车牌区，拿它算准确率是自欺欺人，所以那条路已经放弃。

真实准确率请在本地用自己的数据集评：

```bash
python tools/fetch_hg.py --out datasets/hg --limit 60   # 拉一套带标注的公开集（可选）
python tools/eval_locate.py datasets/hg/test            # 打基线
python tools/eval_locate.py datasets/hg/test --grid quick --save-baseline reports/base.json
python tools/eval_locate.py datasets/hg/test --baseline reports/base.json   # 之后每次改参数都跑这个
```

### 13.4 CI 跑什么

`.github/workflows/ci.yml` 在 push / PR 时跑两个 job：

1. **`lint`（静态检查，Ubuntu + Python 3.12）**：
   `ruff check src tools tests` / `ruff format --check src tools tests` / `mypy`；
2. **`test`（多版本 pytest，5 个组合）**：
   Ubuntu × Python 3.10 / 3.11 / 3.12 / 3.13，外加 **Windows × 3.12**。
   Windows 单独覆盖的理由：本项目大量中文路径 / 路径分隔符逻辑与 Windows 强相关。
   Qt 用例在 `QT_QPA_PLATFORM=offscreen` 下跑（CI 没有 X）；Linux 作业还会额外装
   `libegl1 / libgl1 / libfontconfig1 / libdbus-1-3` —— PySide6 的 wheel 只带 Qt 本体，
   `QtGui`/`QtWidgets` 在 Linux 上还要 dlopen 这些系统库，缺了连
   `import PySide6.QtWidgets` 都会 ImportError（v2.7.3 首次推送把 5 个 ubuntu 作业
   全部弄红的真实原因，详见 `docs/UPGRADE.md`）。

**回归防线**（都包含在上面的 pytest 里，不单独占 job）：

- **定位行为回归基线**：用确定性 `cv` 引擎跑 `img/` + `samples/` 下的**全部**图片
  （2026-10-16 时为 403 张 —— 其中含一批下载来的 CCPD 样张，见 `docs/ROADMAP.md` 第 4 项），
  与 `tests/baselines/real_samples.json` 对照 —— 候选数、top 候选框（IoU ≥ 0.85）、
  top 底色**任一漂移即失败**。往 `img/` 里加图会让这条绊线报错，
  那是故意的：新增样张必须重新生成基线（`python tools/make_regression_baseline.py --write`）；
- **真实样张准确率门槛**：`deep` 引擎 + `tests/baselines/ground_truth.json` 人工真值；
- **对抗退化样张**：`make_hard_samples.py` 确定性生成（模糊/降采样/曝光/噪点/倾斜/压缩），
  跨运行结果一致。

### 13.5 常用工具脚本

```bash
# 性能基准（输出耗时/QPS/命中率、安全防护逐项检查、覆盖范围统计）
python tools/benchmark.py --images samples --repeat 3 --json out/bench.json

# 多进程加速到底有多少收益（--cpu-control 同时测纯 CPU 并行，用来排除"机器没余量"）
python tools/bench_workers.py --images ./photos --engine deep --workers 1 4 8 --cpu-control

# 多引擎准确率评测（失败四分类 + 互补性分析）
python tools/eval_ccpd.py --engines cv,deep,yolo,fusion:ensemble --taxonomy --json reports/eval.json
python tools/eval_report.py reports/eval.json        # 排成 markdown 表

# 界面截图（离屏渲染，不需要显示器）
python tools/shot.py reports/ui.png --sample samples/1.jpg
```

---

## 14. 故障排查（FAQ）

> **先跑 `start.bat doctor`（Windows）或对照 `lpr engines` 的输出** —— 大多数"用不了"
> 是权重/依赖缺失，这两条命令会直接告诉你缺什么、装什么。
> 排查完还想看细节，就去 `logs/` 翻运行日志（§3.8 / §14.6）。

### 14.1 安装与启动

| 症状 | 原因 | 处置 |
|---|---|---|
| 界面里某个引擎写着「（不可用）」 | 缺依赖或缺权重 | `start.bat doctor`，照打印的修复命令做 |
| `ModuleNotFoundError: No module named 'onnxruntime'` | `yolo` 的硬依赖没装 | `pip install -e ".[yolo]"` |
| `EngineUnavailable` 并附带一条绝对路径 | `yolo` 没有权重 | 按 [models/yolo/README.md](models/yolo/README.md) 取权重，或 `python tools/fetch_yolo_weights.py --list` |
| `start.bat` 一闪而过 | 解释器版本太低或依赖装失败 | 在 cmd 里手动运行 `start.bat` 看完整输出；确认 `python --version` ≥ 3.10 且是 64 位 |
| `pip install` 报 OpenCV 版本冲突 | `rapidocr` 会拉 `opencv-python` 5.x | `pyproject.toml` 已锁 `<5`；若手工装过 5.x 请 `pip install "opencv-python>=4.10,<5"` |
| 准确率突然掉到门槛线 | `rapidocr-onnxruntime` 版本漂移 | 本项目**钉死 1.2.3**（1.4.x 移除了纯识别快路径，1.2.13 模型有漂移） |

### 14.2 识别效果

| 症状 | 原因 | 处置 |
|---|---|---|
| 结果为空 / 只报垃圾号码 | 用了 `cv` 引擎 | 换 `--engine deep` 或 `yolo`；`cv` 在 CCPD 上 0/100 |
| 车牌侧倾就找不到 | 轮廓法在侧倾大时定位失效 | 用 `yolo`（检测网络不依赖轮廓） |
| 双层牌（黄牌大车后牌）漏检 | 双行形态未进候选池 | 检查 `double_min_aspect` / `double_color_ratio`；或直接用 `yolo`（自带单/双层分类） |
| 末位字符总是被吃掉 | 精定位裁剪过狠 | 调大 `engine.locate.color_edge_ratio`（默认 0.15） |
| 警/军牌末位红字漏检 | 小字号红色字符是通用 OCR 的弱区 | 已知限制，见 [§15](#15-已知限制) |
| `yolo` 框正常但角点乱飞 / 分数恒为 0 | `yolo_attr_order` 给错了 | Ultralytics 导出的权重改成 `"cls_first"` |
| `yolo` 只给框、`text` 为空 | 纯检测模式（没配识别权重，也没装 `deep`） | `pip install -e ".[yolo,deep]"` |
| 检测到了但没认出字 | OCR 无产出 | 结果里**保留框**、`text` 为空、`meta["ocr_failed"] = True` —— 统计命中率时请认 `text` 非空 |

### 14.3 输出与编码

| 症状 | 原因 | 处置 |
|---|---|---|
| 重定向/管道输出中文报 `UnicodeEncodeError` | 管道编码取决于 locale | 已在 `cli._make_pipes_utf8()` 里把**非 tty** 的 stdout/stderr 拨到 UTF-8（控制台保持 cp936 不动） |
| `lpr engines` 列对不齐 | 用字符数而非显示宽度算 CJK 宽度 | 已知显示问题，不影响内容 |
| CSV 用 Excel 打开乱码 | 编码 | 已用 `utf-8-sig` 写出，正常不该乱码；若仍有问题请检查是否被别的工具重写过 |
| 图片路径带中文读不出 | 早期版本用 `cv2.imread` 直读 | 已改为 `imdecode` + `fromfile` 路线并自动纠正 EXIF 旋转 |

### 14.4 性能与并发

| 症状 | 原因 | 处置 |
|---|---|---|
| `--workers 8` 反而更慢 | 机器 CPU 配额不足，或文件数太少 | 先跑 `tools/bench_workers.py --cpu-control` 量一下；小批量会自动退回串行 |
| `fusion(ensemble)` 2 秒/张 | 三引擎各跑一次，耗时≈成员之和 | 要快就用 `chain`（命中即停）或 `auto`（单引擎） |
| `deep` 比文档上慢 | 补救预处理在垃圾候选上消耗预算 | 每图最多补 2 个候选；吞吐敏感场景见 [docs/ROADMAP.md](docs/ROADMAP.md) 的开关化提案 |
| 摄像头画面卡 | 旧版本把推理写在采集循环里 | v2.3.1 已改为双线程；若仍卡请确认用的是新版本 |

### 14.5 摄像头

打不开时提示是**可行动的**：自动枚举可用设备 + 给出排查清单，并在工具栏提供摄像头选择。

- Linux：走 V4L2，确认 `/dev/video*` 权限；
- Windows：别被其他程序（会议软件）占用；
- 编号从 `0` 开始试：`lpr video 0`、`lpr video 1`。

### 14.6 运行日志

| 症状 | 原因 | 处置 |
|---|---|---|
| 找不到日志在哪 | 默认在**当前工作目录**的 `logs/` | 命令行启动时第一行就印了绝对路径；界面里看「运行日志」面板下面那行。双击 `start.bat` 时工作目录就是项目根 |
| 界面「运行日志」面板是空的 | 面板读的是**内存缓冲**（最近 500 条），启动初期本来就没几条日志 | 识别一张图后就会有内容；真正的历史记录看文件 |
| 面板有内容但没有文件 | `log_to_file` 被关了，或目录不可写（stderr 会有 `[警告] 无法写入日志文件`） | `lpr config --show` 看生效值；把 `log_to_file` 设回 `true` |
| `logs/` 里一堆 `-p<数字>.log` | `--workers` 多进程，每个子进程一个文件 | 正常行为，见 §3.8。只看主进程的 `lpr-<日期>.log` 即可 |
| 日志里有 `皖A***45` 这种掩码 | `security.mask_logs`（默认开）在起作用 | 这是隐私保护，不是 bug；确实要看明文就把它关掉，但**别在真实数据上关** |
| 日志文件长到几十 MB | `log_max_mb` 调大了，或某次跑了极长的批 | 默认 5MB 就轮转，保留 5 份，不会无限增长；嫌多把 `log_backups` 调小 |

---

## 15. 已知限制

### 15.1 引擎层面

- **`fusion`（三合一）的耗时≈成员之和**：`ensemble` 策略下三个引擎各跑一次，
  实测 `samples/1.jpg`（4096×2304，两块车牌）约 **2.2 s/张**，而单引擎
  `yolo` ≈ 0.9 s、`deep` ≈ 0.8 s。它换的是**精度与召回**，不是速度；
  要快就用 `chain`（命中即停）或 `auto`（单引擎）。
- **`fusion` 目前不提升识别率（v2.7.0 实测）**：100 张 CCPD 上三种策略的严格命中
  都等于最强单成员（`yolo` 84%），因为 `deep` 的命中集合是 `yolo` 的**子集** ——
  成员之间没有互补，融合救不回任何图。它的实测价值是**更少的误检**
  （ensemble 5 条 vs yolo 7 / deep 14 / cv 691）。详见 [ACCURACY.md §6](docs/ACCURACY.md)。
- **`fusion` 的 `auto` 策略本质是单引擎**：它只按图像特征挑一个跑。实测
  `samples/1.jpg` 是"大图 + 清晰"，按规则选了 `deep`，结果只报出 **1 块**
  车牌（远景那块小牌漏了），而 `yolo` 能报出 2 块 —— 所以**要召回就别用 auto**。
  v2.7.0 起清晰大图的首选成员可配置（`fusion_auto_clear_first`，默认 `deep` 行为不变）。
- **`fusion` 的准入规则会丢候选，且丢的是"低票孤立结果"**：默认要求"至少两个
  成员一致"或"票权 ≥ 1.0 的成员独存"。被滤掉的候选不是静默丢弃：
  `FusionEngine.last_dropped` 里能查到，日志也会列出号码。想取并集（最大召回）
  把 `engine.fusion_min_support` 设成 1。
- **第二识别模型（低置信交叉验证）默认关闭，实测也应保持关闭**：100 张 CCPD 上
  开启后严格命中一張未变（失败集合逐张相同），`deep` 耗时却 ×2.9。
- **补救预处理会让 `deep` 慢约三成（v2.7.2 实测 716→946 ms）**：它只在候选
  完全读不出字时触发，`deep` 的失败是"没框"或"读错了"都不触发，但候选池里
  的垃圾候选会消耗预算空跑（每图最多补 2 个候选）。`yolo` 每图只有 1~2 个
  候选，实测中位耗时基本不变。收益端：yolo +1 张、字符级 +2.9pp。
- **`cv` 的 SVM 严重过自信**：对任意输入都输出接近 1.0 的置信度，且中文省份识别
  尤其差。**需要精度请用 `deep` 或 `yolo`。**

### 15.2 识别准确率

- **没有"一键可用"的官方 `yolo` 权重**：主流来源要么是 Git LFS 指针（拿到的是几十字节的
  文本），要么在网盘后面。`tools/fetch_yolo_weights.py` 因此定位为**校验与体检工具**，
  而不是自动装配器。
- **`yolo` 的输出布局存在形状上无法消解的歧义**：Ultralytics 系是 `box + cls + kpts`，
  `we0091234` 系是 `box + obj + kpts + cls`，两者**属性个数完全相同**，只看形状
  无法区分。默认 `kpt_first`；给错的典型症状是框正常但角点乱飞、或分数恒为 0。
- **`yolo` 的字符识别默认复用 `deep` 引擎的 OCR**。只装 `.[yolo]` 没装 `.[deep]`
  时退化为**纯检测模式**（有框、无文本）。想要完整识别请装 `pip install -e ".[yolo,deep]"`。
- **"检到了但没认出"与"没检到"是两种结果**：检测成功而 OCR 无产出时，结果里
  **保留框**、`text` 为空、`meta["ocr_failed"] = True`（不会静默丢框）。所以
  "结果条数"与"识别成功条数"不是一回事。
- **军牌 / 警牌（白牌特种牌）末位红字漏检仍在**：v2.3.2 已补齐语法层（警/军队/使领馆/
  武警新旧式严格校验、后缀与字头形近字纠错、序号位 I/O 修正）与判色层（红色字符信号
  识别阴影白底警牌），但小字号红色字符是通用 OCR 模型的弱区，`鄂F8888警` 可能读成 `鄂F8888`。
- **新能源牌与普通民用牌在 8 位串形上存在固有歧义**（`京AQ7797Q` 与 `京AD77972`
  串形无法区分），依赖 OCR 置信度与多帧投票消解。
- **侧倾召回的**定位**仍偏弱**：矫正几何已修复（±89° 内不再转成竖图），但整景照片里
  侧倾 >45° 的车牌可能根本找不到候选（矫正只在"找到了候选"之后才起作用）。
- 双行牌已支持自动拆行，但极端倾斜/污损样本仍可能失败。

### 15.3 集成与部署

- **`Python API` 的 `engine` 参数默认是 `"auto"`，不读 `config.engine.default`**：
  想让 API 也用配置里的默认引擎，要显式传 `Recognizer(engine=config.engine.default)`。
- **`doctor` 与启动器的引擎判定只看默认权重路径**：它检查的是 `models/svm*.dat` 与
  `models/yolo/plate_det.onnx`（即 `EngineConfig` 的默认值）。如果你用
  `engine.yolo_models_dir` 之类的配置把权重放到别处，启动器仍会报「不可用」——
  因为 `yolo_engine.is_available()` 这个**类方法**同样只认默认目录（`lpr engines` 也走它）。
  实际 `predict` 时用的是实例自己解析的路径，能正常跑。
- **多进程批量（`--workers`）未在真实多核机器上验证过收益**：开发/CI 环境 CPU
  配额受限（纯 CPU 计算 4 进程 0.49x），因此"并行更快"这个结论**我们没有证到**。
  功能本身完整且有测试，但请自行用 `tools/bench_workers.py --cpu-control` 确认。
- **摄像头实时识别已做到采集与识别真解耦（双线程），但 GUI 路径没有单元测试覆盖**
  （Qt 线程测试成本高），只有源码级回归护栏。
- **深度引擎首次运行需下载 OCR 模型**（`deep-paddle` 路线约 10MB），之后离线可用。

### 15.4 评测与数据

- **定位准确率没有 CI 门槛**（原因见 [§13.3](#133-为什么-ci-里没有准确率门槛)）；
  现有的是"行为不变"快照，能拦住回归，但拦不住"一直都在错"。阈值调参已用弱监督评测
  复核过一轮（`tools/eval_weak.py`）：三组候选参数无一在留出集复现改善，**维持默认参数**。
- **实测样本量有限**：CCPD 100 张、自有样张 10 张。所有百分比都带 ±几个点的抽样误差，
  **不要把 84% 与 79% 的差当成精确的引擎差距** —— 但"定位 100% vs 82%"这类结构性
  差异是可信的（每张图都测了）。
- **CCPD 以 `皖` 牌为主**，省份混淆对的方向不可迁移（实测两个引擎都把 `赣`/`苏` 读成 `皖`）。
  **不要**据此在生产里加"省份白名单"。
- `img/` 现在是**样张全集**：既有原本入库的 9 张样张，也有从 CCPD 下载后搬进来的样本
  （2026-10-16 时共 403 张 / 75.7MB）。原先设想的 `images/`（一次性集中副本）现在
  **是空目录**，所以"`img/` 是 `images/` 的子集"这个说法已不成立；测试与基线 JSON
  引用的一直是 `img/`。想把体积降回原量级，应当把下载图放进 `datasets/`（已 gitignore），
  而不是继续往 `img/` 里搬。见 [docs/ROADMAP.md](docs/ROADMAP.md) 第 4 项。

### 15.5 运行日志

- **界面「运行日志」面板只显示内存缓冲的最近 200 行，不读文件**：它是"刚才发生了什么"
  的随手看，不是日志查看器。翻历史要去 `logs/`（或用文件管理器打开那个目录）。
- **日志按天 + 按进程分文件，没有合并视图**：`--workers 8` 跑一天会产生 1 个主文件
  + 8 个子进程文件。要追一次具体运行得自己按 pid 对时间戳，工具链里没有聚合器。
- **日志只落在本地**：没有上报、没有轮转之外的保留策略（`log_backups` 默认 5 份，
  更早的会被删掉）。要长期留档请自行把 `logs/` 纳入备份，或调大 `log_backups`。
- **日志文件名里的 pid 在容器/长跑进程里可能很难对应上是谁**：目前只有 pid，
  没有"哪个分片"的信息。

---

## 16. 开发与发布

### 16.1 本地开发

```bash
git clone https://github.com/hongligongsi/LicensePlateRecognition-V2
cd LicensePlateRecognition-V2
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev,gui,deep]"                    # 开发所需全部

pytest tests/ -q --timeout=300                      # 全量测试
ruff check src tools tests                          # lint
ruff format src tools tests                          # 格式化（CI 会查 --check）
mypy                                                 # 类型检查（只查 src/lpr）
```

三条硬约定：

- **`tools/launcher.py` 只允许依赖标准库**（有守卫测试不许 `import lpr`），
  因为它在"环境还没装好"时就要能跑；
- **`tools/launcher.py` 的值对象用 `NamedTuple` 而不是 `@dataclass`**：
  启动器按文件路径加载模块（`spec_from_file_location` + `exec_module`，不注册
  `sys.modules`），`@dataclass` 在裸加载下装饰期会去读
  `sys.modules[cls.__module__]` 并抛 `AttributeError`；
- **`start.bat` 必须保持纯 ASCII + CRLF**（原因见 [§3.4](#34-四种启动方式)）。

### 16.2 版本号的唯一来源

`src/lpr/__init__.py` 的 `__version__` 是**唯一来源**；`cli.VERSION` 从包元数据取，
界面标题不写版本号（状态栏显示）。`pyproject.toml` 与 `tools/launcher.py` 各有一份
需要同步，**有测试钉住这几处一致**，改版本时四处一起改。

### 16.3 发布流程

```bash
# 1. 同步版本号（pyproject.toml / src/lpr/__init__.py / tools/launcher.py）
# 2. 全量校验
ruff check src tools tests && ruff format --check src tools tests && mypy
pytest tests/ -q --timeout=300
# 3. 提交并打 tag
git commit -am "vX.Y.Z ..."
git tag vX.Y.Z && git push origin main --tags
# 4. 建 Release（顺带写发布说明）
```

> 实战经验：`git push` 会遇到间歇性 502（`CONNECT tunnel failed, response 502`）。
> 兜底是走 **GitHub Git Data API** 上传对象并更新引用（SHA 与本地一致），
> Release 用 REST API 建。

### 16.4 第三方许可

新增依赖或权重时**必须**在 [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) 登记来源与许可，
并在运行时以显式提示告知用户相应的许可义务。特别地，`yolo` 权重不入库的原因之一是
主流权重派生自 AGPL-3.0 或 GPL-3.0（见该文档）。

---

## 17. 许可

**GNU GPL-3.0**（`GPL-3.0-only`），全文见 [LICENSE](LICENSE)。

原项目 [hongligongsi/License-Plate-Recognition](https://github.com/hongligongsi/License-Plate-Recognition)
由 wzh191920 于 2021 年以 **MIT** 许可发布。MIT 允许再许可，因此本项目可整体升级为
GPL-3.0；但依据 MIT 的保留条款，**原作者的版权与许可声明必须随源保留** —— 该声明完整
列在 [LICENSE](LICENSE) 文件末尾，并另存一份于 [LICENSE.MIT.original](LICENSE.MIT.original)。

由此带来的实际影响：

- 你可以自由使用、修改、再分发，但**再分发时必须提供完整源码**并以 GPL-3.0 授权；
- 若你将本项目（或其修改版）作为**网络服务**提供给他人，需注意 GPL-3.0 的源码提供义务
  适用于你分发的软件副本；
- 本项目引用的第三方组件各有其许可（如 RapidOCR 为 Apache-2.0、ONNXRuntime 为 MIT、
  PySide6 为 LGPL-3.0），清单见 [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)。

> 说明：`v2.4.3` 及更早版本以 MIT 发布，其许可状态不受本次变更影响 —— 已获取的
> 旧版本仍可按 MIT 条款使用。

---

## 附：版本演进

> 逐版本的详细记录（含每个版本踩过的坑、修掉的真 Bug、被否掉的方案）见
> [docs/UPGRADE.md](docs/UPGRADE.md)（2508 行）。这里只列主线。

| 版本 | 主题 |
|---|---|
| **v2.1** | 更精准、更快速、更安全、更全面（四引擎骨架、定位引导加速、安全层、全类型车牌） |
| **v2.2** | 补上「版式维度」（单行/双层）与 13 类车牌 |
| **v2.3** | 召回分层与形近字纠错 |
| **v2.3.1** | 修掉 4 个真 Bug（安全层覆盖、采集/识别解耦、模型校验三态），补上 CI |
| **v2.3.2** | 特种号牌专项（军/警/领/学语法补齐 + 红色字符判色） |
| **v2.3.3** | 启动入口（`python -m lpr`）、`start.bat`、环境与系统要求 |
| **v2.3.4** | 修掉侧倾 >45° 把车牌矫正成竖图的缺陷（±89° 全范围归位） |
| **v2.3.5** | `start.bat` 依赖误判：只查 PySide6 不查 numpy/OpenCV |
| **v2.3.6** | 中文提示从 `.bat` 搬走（cmd 的字节偏移 bug） |
| **v2.3.7** | 「打不开摄像头」变成可行动的提示 + 工具栏摄像头选择 |
| **v2.3.8** | 界面视觉重做（工具条、深色车牌展示卡、颜色徽章、表格/滚动条/菜单） |
| **v2.3.9** | 二次打磨并揪出两个老 Bug |
| **v2.4.0** | 加严早停并引入 ROVER 风格读数融合（识别率专项） |
| **v2.4.1** | 修掉引擎下拉弹层条目「隐形」（白底白字） |
| **v2.4.3** | 历史行点击回显；CI 从 5 项红修到全绿；许可从 MIT 切到 GPL-3.0-only |
| **v2.4.4** | 新增 `yolo` 检测识别引擎骨架 |
| **v2.5.0** | 把 `yolo` 从骨架补成可用引擎（ONNX 全链路 + 桩模型自检） |
| **v2.5.1** | 用真实权重跑通，`attr_order` 接进配置 |
| **v2.6.0** | 新增第四引擎 `fusion`（三合一：并联投票 / 顺序兜底 / 特征选优） |
| **v2.7.0** | 准确率实测：六配置对比、失败按环节拆解，修掉 `auto` 路由 bug |
| **v2.7.1** | 启动器补上模型权重自检 + 只读体检 `doctor` |
| **v2.7.2** | 提准确率（补救预处理）+ 界面双主题，并整理待办清单 |
| **v2.7.3** | 运行日志落盘（轮转 + 多进程分文件），修好界面「运行日志」面板 |

**待完善清单**见 [docs/ROADMAP.md](docs/ROADMAP.md)（按「用户可感知收益 ÷ 实现风险」排序）。
