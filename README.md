# 车牌识别系统 v2.3.3（License-Plate-Recognition 升级版）

用 Python + OpenCV + 深度学习做的**中国车牌识别**，支持蓝牌 / 黄牌 / 绿牌（新能源）/ 白牌 / 黑牌。

本项目是 [hongligongsi/License-Plate-Recognition](https://github.com/hongligongsi/License-Plate-Recognition) 的现代化升级版。原项目最后一次提交停留在 2021 年，其依赖（Python 3.4 / OpenCV 3.4 / numpy 1.14）在如今的机器上已无法安装。v2 在保留原算法思想的基础上全面重构，详见 [UPGRADE.md](docs/UPGRADE.md)。

> **v2.1 主题：更精准、更快速、更安全、更全面。** 详见下方「v2.1 四大升级」。
> v2.2 补上「版式维度」与 12 大类车牌；v2.3 做召回分层与形近字纠错；
> v2.3.1 修掉 4 个真 Bug（安全层覆盖、采集/识别解耦、模型校验三态）并补上 CI；
> v2.3.2 专项提升军牌/警牌/白牌等特种号牌（语法补齐 + 红色字符判色）。
> v2.3.3 补上启动入口（`python -m lpr`）、Windows 一键启动 `start.bat`
> 与「环境与系统要求」说明，并修掉摄像头后端的跨平台硬编码。
> 逐版本记录见 [docs/UPGRADE.md](docs/UPGRADE.md)。

## 核心特性

- **双引擎可切换**
  - `cv` 传统 CV 引擎：OpenCV 形态学定位 + 投影分割 + SVM 字符识别（兼容原版模型），纯 CPU、**约 28 ms/张**、零额外依赖
  - `deep` 深度学习引擎：基于 RapidOCR（ONNX）/ PaddleOCR，中文省份简称与模糊车牌的准确率远高于 SVM，**约 190 ms/张**
  - `cascade` 级联：CV 快速定位给出候选 → 深度引擎精识别（精度与速度的折中）
- **现代化 GUI**（PySide6）：单图 / 批量 / 摄像头实时识别，识别不再冻结画面，历史记录可导出 CSV
- **命令行工具**：`lpr predict`、`lpr video`、`lpr gui`、`lpr engines`、`lpr check`、`lpr models`、`lpr config`
- **参数自适应**：定位参数以比例表达，同一套配置适配 640px～4000px 输入（原版换分辨率就失效）
- **多车牌**：一张图里的多个车牌全部识别（原版只返回第一个）
- **置信度**：字符级 + 整体置信度，标注图可视化
- **中英文路径**：图片读写全兼容中文路径，自动纠正 EXIF 旋转

## v2.1 四大升级

### 一、更精准

| 机制 | 说明 |
|---|---|
| **定位引导识别** | 先用轻量 CV（~50ms）圈出车牌候选区，只对候选区做 OCR，避免全图噪声干扰 |
| **多尺度投票** | 超分（目标字符高 48px）× 锐化（USM）组合投票，票重 = OCR 置信度 × 语法合法性 |
| **早停收敛** | 首个尺度即得到合法号码且票数占比过半就停止，典型场景只跑 1～2 轮 |
| **语法纠错** | `plate_rules` 按真实车牌规则纠错（I/O 不合法、发牌机关须为字母、末位降级） |
| **跨帧投票** | 视频/摄像头用 `MultiPlateTracker` 关联轨迹，多帧投票纠正单帧偶发错误 |
| **同图去重** | 同一号码无论位置都合并为一条观测，位置相近但号码不同则同时保留（不漏报） |

### 二、更快速

关键突破是**跳过 OCR 的文本检测网络**：车牌的文本检测（DBNet）本身就是重型网络，而我们已用 CV 定位拿到了车牌区域，因此直接把整块 ROI 当作文本框送入识别网络（`get_boxes_img_without_det`）。

实测（`samples/`，每张重复 3 次）：

| 引擎 | 冷启动 | 中位耗时 | P95 | QPS |
|---|---|---|---|---|
| `deep` | 541 ms | **194 ms** | 507 ms | 3.8 |
| `cv` | 143 ms | **28 ms** | 31 ms | 35.5 |

对比 v2.0 的全图 OCR 路径（约 1600 ms/张），`deep` 引擎提升约 **8 倍**。

> 注：`cv` 引擎虽快，但 SVM 对任何输入都输出置信度 1.0（严重过自信）且字符识别错误率高，**仅供快速筛候选**，精度请用 `deep` 或 `cascade`。

### 三、更安全

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

**脱敏是端到端生效的**：控制台输出、JSON、CSV、嵌套 `meta` 全部覆盖，均有单元测试守住。

**安全层覆盖三条入口**（v2.3.1 修复）：

| 入口 | 输入校验 | 输出脱敏 | 日志脱敏 | 模型校验 |
|---|---|---|---|---|
| `lpr` CLI | ✅ | ✅ | ✅ | ✅ |
| Python API（`Recognizer`） | ✅ | ✅ | ✅ | — |
| 图形界面（`lpr gui`） | ✅ | ✅ | ✅ | — |

> v2.3.1 之前，Python API 会**静默忽略** `config.json` 的 `security` 段
> （`LPR_MASK_OUTPUT` 也对 API 无效），图形界面则**完全没有接入安全层**
> —— 图片不校验、CSV 导出裸写明文。现在三条入口共用同一份 `config.security`。

模型哈希白名单开箱可用（`config.json` 的 `security.model_hashes` 已内置随仓库
分发的两个 SVM 模型哈希），换自训练模型后需同步更新，否则会被判为不匹配：

```bash
lpr models --verify
#   svm.dat                    4.38 MB  2026-10-11 23:05:06
#     ✓ 哈希校验通过 (ce3fdffb3e9529fc…)
# （✓=已校验通过，—=未配置哈希白名单故未校验，✗=不匹配）
```

```bash
lpr predict car.jpg --engine deep --mask          # 控制台输出打码
lpr predict ./photos/ --engine deep --mask --report out/   # 报告落盘同样打码
```

### 四、更全面

- **车牌底色（五种 + 危险品）**：

  | 底色 | 适用车辆 |
  |---|---|
  | 蓝色 | 小型汽车（最常见） |
  | 黄色 | 大型汽车、教练车 |
  | 绿色 | 新能源、农用 |
  | 白色 | 警用、军队、特种车辆 |
  | 黑色 | 使领馆、境外车辆 |
  | **危险品运输** | 黑底**黄字**，危险品运输车辆专用 |

  > 危险品牌与普通黄牌**底色相反**（普通黄牌是黄底黑字），仅看"有没有黄色"无法区分。
  > 实现上用**明暗极性判定**：要求「黑色主体占比 ≥ 45%」且「黄色前景占比 ≥ 3%」且
  > 「黄字只占少数（≥ 6% 的 黄/(黄+黑) 比值）」，第 1 条排除黄底黑字、第 2 条排除黑底白字。
  > 可用 `--color` 按底色过滤，例如只挑出危险品运输车：`lpr predict ./photos/ --color 危险品`

- **车牌类型（14 种格式 / 12 大类）**：普通民用（单行蓝牌、单行黄牌）/ 新能源（含双层绿牌）/ 警用（白色警用车牌，含双层白牌）/ 军队 / 武警 / 使领馆（使用馆车牌）/ 挂车 / 教练车 / **危险品运输** / 港澳入出境（粤Z 港澳牌）/ 短牌 / **民航**（`民航A0001`，绿底白字，依据 CCAR-331SB-R1）
- **版式维度（单行 / 双层）**：v2.2 起每个识别结果同时给出 `layout` 字段——版式与底色是**相互独立**的两个维度（同样是黄牌，小型教练车是单行、大型汽车后号牌是双层）。判定依据：行分割证据 + 宽高比先验（单行 440×140mm ≈ 3.1:1，双层 440×220mm ≈ 2:1）
- **双层牌处理**（借鉴 [YOLO26-plate](https://github.com/we0091234/Chinese_license_plate_detection_recognition)）：**先在候选阶段判定单/双层**，只有双层才拆行拼接——单行牌不做无谓的双倍识别。拆行用「裁左右边框 + 行投影」，不依赖固定比例，对带框车牌与任意宽高比都成立
- **输入来源**：图片 / 目录（可递归）/ 视频文件 / 摄像头流 / numpy 数组
- **输出格式**：控制台 / JSON / CSV / 标注图片 / 标注视频

## 环境与系统要求

### 最低要求

| 项目 | 要求 | 说明 |
|---|---|---|
| **操作系统** | **Windows 10 (1809+) / Windows 11（64 位）**、Linux（64 位） | 兼容矩阵见下 |
| **Python** | **3.10 ~ 3.13，必须 64 位** | `requires-python = ">=3.10"`；CI 覆盖 3.10 / 3.11 / 3.12 / 3.13 |
| **CPU** | 任意 x86-64 | **不需要 GPU、不需要 CUDA**，推理全程纯 CPU |
| **内存** | 可用 2 GB 以上 | deep 引擎加载 OCR 模型约占 500 MB |
| **磁盘** | 约 30 MB | 仓库自带 SVM 模型 8 MB；OCR 模型 13.7 MB 随 wheel 分发 |
| **网络** | **只在 `pip install` 时需要** | 装完后从识别到界面**全程离线**（模型都已随包分发） |
| **摄像头** | 可选 | 只有「摄像头实时识别」功能需要 |

### 系统兼容矩阵

| 系统 | 命令行 | 图形界面 | 说明 |
|---|---|---|---|
| **Windows 11（64 位）** | ✅ | ✅ | 本项目开发与实测环境（Python 3.13） |
| **Windows 10 1809+（64 位）** | ✅ | ✅ | CI 在 `windows-latest` 上覆盖 Python 3.12 |
| Windows 10（1809 之前，如 1607 / LTSB 2016） | ⚠️ | ⚠️ | 未验证。PySide6（Qt 6.6+）官方要求 1809 以上 |
| Windows 8.1（64 位） | ⚠️ | ❌ | Python 3.13 官方仍支持 8.1，但 Qt 6 已放弃 → **界面用不了** |
| Windows 7 | ❌ | ❌ | 见下「为什么 Windows 7 / 8 / XP 不行」 |
| Windows XP | ❌ | ❌ | 同上 |
| 32 位 Windows（任意版本） | ⚠️ | ❌ | Qt 6 没有 32 位 Windows 版 → 界面用不了 |
| **Linux（Ubuntu 20.04+ / 22.04+，64 位）** | ✅ | ✅ | CI 覆盖 Ubuntu × Python 3.10~3.13；摄像头走 V4L2 |
| macOS 12+ | ✅ | ✅ | Qt 6 要求 macOS 12+；本项目**没有** macOS 的 CI 覆盖 |

### 为什么 Windows 7 / 8 / XP 不行

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

### 四种启动方式

**① 双击（Windows 最省事）**

```
双击 start.bat
```

脚本会自动：找 Python（优先 `LPR_PYTHON` 环境变量 → `.venv` → `py -3` → `python`）
→ 校验是否 ≥ 3.10 → 缺 PySide6 就装 → 启动界面。带参数运行则透传给命令行：

```bat
start.bat engines                                :: 查看引擎状态
start.bat predict car.jpg --engine deep          :: 单图识别
start.bat --help                                 :: 命令行用法
```

**② 装成命令（推荐长期使用）**

```bash
pip install -e ".[deep,gui]"    # 一次性：CV + 深度学习引擎 + 图形界面
lpr gui                          # 开界面
lpr predict car.jpg --engine deep
```

**③ 不安装，直接用源码**

```bash
# Windows
set PYTHONPATH=src
# Linux / macOS
export PYTHONPATH=src

python -m lpr                    # 无参数 → 直接开图形界面
python -m lpr predict car.jpg --engine deep
python -m lpr.cli --help         # 只看命令行用法
```

**④ 最小安装（只要命令行 + CV 引擎，零额外依赖）**

```bash
pip install -e .
lpr engines                      # 看看当前环境有哪些引擎可用
```

### 引擎与依赖的对应关系

| 引擎 / 功能 | 额外依赖 | 装完后还要下载模型吗 |
|---|---|---|
| `cv`（基础） | 无（numpy + opencv + Pillow） | 否，`models/*.dat` 随仓库分发 |
| `deep`（推荐） | `rapidocr-onnxruntime` | 否，模型 13.7 MB 在 wheel 内 |
| `deep-paddle` | `paddleocr` + `paddlepaddle` | **首次运行会联网下载模型**，离线环境别用 |
| 图形界面 | `PySide6>=6.5` | 否 |

> **纯离线环境请用 `deep`（RapidOCR）**：它和 `cv` 一样装完即用；
> `deep-paddle` 首次运行需要联网拉模型。

## 安装

```bash
# 基础（传统 CV 引擎）
pip install -e .

# 推荐：+ 深度学习引擎 + 界面
pip install -e ".[deep,gui]"

# 可选：PaddleOCR 后端（更大更慢但部分场景更准）
pip install -e ".[deep-paddle]"

# 开发（含测试）
pip install -e ".[dev]"
```

> **注意**：`opencv-python` 必须是 **4.x**（`>=4.10,<5`）。OpenCV 5.0 移除了 `cv2.ml`，原版 SVM 模型将无法加载，`pyproject.toml` 已锁定上限。

SVM 模型（`models/svm.dat`、`models/svmchinese.dat`）已随仓库附带，来自原项目（EasyPR 样本训练）。

## 使用

### 命令行

```bash
lpr engines                                       # 查看引擎状态与可用性
lpr predict car.jpg --engine deep                 # 单图识别
lpr predict ./photos/ --engine deep --recursive   # 目录批量（递归）
lpr predict ./photos/ --save-annotated out/ann --report out/   # 标注图 + JSON/CSV 报告
lpr predict car.jpg --engine deep --mask --json   # 脱敏输出
lpr predict ./photos/ --color 危险品              # 只保留危险品运输牌
lpr predict ./photos/ --color 黄色,危险品          # 多底色过滤（逗号分隔）
lpr predict ./photos/ --engine deep --workers 8   # 多进程并行（大批量时用）
lpr video parking.mp4 --stride 10 --output out.mp4 # 视频识别（跳帧 + 标注输出）
lpr check 京A1234危 京AD77972 --fix               # 校验号码合法性并纠错
lpr models --verify                               # 查看模型信息与哈希校验
lpr gui                                           # 图形界面
```

### 批量并行（`--workers`）

```bash
lpr predict ./photos/ --engine deep --workers 8    # 8 进程
lpr predict ./photos/ --engine deep --workers 0    # 按可调度 CPU 数自动
```

行为约定：

- **顺序不变**：多进程结果顺序与传入顺序一致，和串行路径完全一样
- **结果不带 ROI**：车牌 ROI 图像跨进程传输开销远大于收益，`workers > 1` 时
  `roi` 恒为 `None`；需要 ROI 就用 `workers=1`
- **单条失败被隔离**：某个文件读不出/识别异常只会记一条 warning，不会中断整批
- **小批量自动退回串行**：文件数不足以摊平子进程启动 + 模型重新加载的固定成本时
  （见 `lpr/batch.py` 的 `MIN_FILES_PER_WORKER`），即使指定了 `--workers` 也会走串行

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

### Python API

```python
from lpr import Recognizer, PrivacyPolicy

# 统一高层 API：一句话识别
rec = Recognizer(engine="auto")          # auto / cv / deep / cascade
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

更低层的引擎接口：

```python
from lpr import get_engine, setup_logging

setup_logging("INFO")
engine = get_engine("deep")
for r in engine.predict("car.jpg"):
    print(r.text, r.color.label, r.confidence, r.bbox)
```

### 界面

运行 `lpr gui`：
- **打开图片**（Ctrl+O）/ **批量识别文件夹**：结果表格展示，可导出 CSV
- **摄像头实时识别**：采集与识别在**两个独立线程**里跑 —— 采集线程只管 `cap.read()`
  和刷画面，识别线程始终取"最新一帧"推理。识别慢于采集时不会积压帧，
  画面按摄像头原生帧率刷新
- 引擎下拉框热切换，不可用的引擎自动置灰并给出安装提示

> v2.3.1 修复：此前 `CameraWorker` 号称"采集与识别解耦"，实际是把
> `engine.predict(frame)` 写在采集循环里**同步执行**（deep 引擎约 200ms/张，
> 采集帧率被拖到 ~5fps），而 `_recognizing` 标志只在 `__init__` 赋值一次、
> 从没被读过 —— 是段死逻辑。现在是真的双线程，并有源码级回归护栏。

## 项目结构

```
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
│   └── deep_engine.py  # 深度学习引擎（定位引导 + 纯识别加速 + 双行牌）
├── ui/
│   ├── window.py       # PySide6 主窗口（已接入安全层）
│   └── worker.py       # 采集与识别分离的线程（CameraWorker + _RecognizeThread）
└── utils/
    ├── image.py        # 图像 IO（中文路径 / EXIF）、标注绘制（Pillow 中文字体）
    ├── security.py     # 安全层：Limits / PrivacyPolicy / 脱敏 / 限流 / 模型校验三态
    └── logging_setup.py
tools/
├── benchmark.py              # 引擎性能基准（耗时/QPS/命中率）
├── bench_workers.py          # 多进程批量加速基准（含纯 CPU 并行对照）
├── eval_locate.py            # 定位客观评测 + 网格搜索（支持 --save-baseline/--baseline）
├── eval_weak.py              # 无标注真实图的弱监督评测（合法率/候选率，支持 --override 调参对照）
├── tune_thresholds.py        # 无监督阈值调参（没有标注时的退路）
├── tune_io.py                # 调参结果落盘（写进 engine.locate，自动备份）
├── fetch_ccpd.py             # 稳健下载 CCPD 样张
├── fetch_hg.py               # 下载带 YOLO 标注的评测集（支持 --limit 小规模验证）
├── ccpd_labels.py            # CCPD 文件名标注解析
├── make_color_samples.py     # 合成六色样张，核对底色/危险品判定
└── make_regression_baseline.py  # 生成真实样张行为快照基线
models/                 # 预训练 SVM 模型（原版权重，config.json 内置哈希）
samples/  img/          # 真实测试样张（回归基线的数据来源）
start.bat               # Windows 一键启动（找解释器 → 校验版本 → 装 GUI 依赖 → 开界面）
tests/                  # 538 项 pytest 测试（含 tests/baselines/ 行为基线）
.github/workflows/ci.yml   # CI：ruff + mypy + 多版本 pytest
docs/UPGRADE.md         # 与原项目的详细对比
```

## 测试与基准

```bash
pip install -e ".[dev,gui]"                         # 测试需要 Qt（GUI 用例）
pytest tests/ -q                                    # 538 项单元/集成测试
pytest tests/ -q -m "not slow"                      # 跳过会起多进程的较重用例
python tools/benchmark.py --images samples --repeat 3 --json out/bench.json
ruff check src tools tests && mypy                  # 静态检查（CI 会跑）
```

基准脚本会输出引擎性能（耗时/QPS/命中率）、安全防护逐项检查、覆盖范围统计，可直接引用。

### 回归防线（CI 拦截）

CI（`.github/workflows/ci.yml`）在 push / PR 时跑三件事：

1. **ruff + mypy** 静态检查（`ruff check` / `ruff format --check` / `mypy`）
2. **多版本 pytest**：Ubuntu 上 Python 3.10~3.13，外加 Windows 3.12（本项目大量
   中文路径 / 路径分隔符逻辑与 Windows 强相关，值得单独覆盖）
3. **定位行为回归基线**（在 pytest 内）：用确定性 `cv` 引擎跑仓库自带的
   11 张真实样张（`img/` + `samples/`），与 `tests/baselines/real_samples.json`
   对照 —— 候选数、top 候选框（IoU ≥ 0.85）、top 底色任一漂移即失败

关于"准确率"要**如实说明**：带标注的真实数据集（CCPD 的标注在镜像里被抹掉，
公开 YOLO 集只有 320 张）不适合进仓库（体积 + 版权），所以 CI 里**没有**准确率门槛。
我们试过用**合成图**替代 —— 不可行：合成牌只有放大到占画面一半宽才勉强被定位到，
且找到的是字符区而不是车牌区，拿它算准确率是自欺欺人，所以那条路已经放弃。

真实准确率请在本地用自己的数据集评：

```bash
python tools/fetch_hg.py --out datasets/hg --limit 60   # 拉一套带标注的公开集（可选）
python tools/eval_locate.py datasets/hg/test            # 打基线
python tools/eval_locate.py datasets/hg/test --grid quick --save-baseline reports/base.json
python tools/eval_locate.py datasets/hg/test --baseline reports/base.json   # 之后每次改参数都跑这个
```

## 已知限制

- 传统 CV 引擎的 SVM 字符识别精度有限（原版模型训练样本太少，中文省份尤其差），
  且对任意输入都输出接近 1.0 的置信度 —— **需要精度请用 `deep` 或 `cascade`**
- 深度引擎首次运行需下载 OCR 模型（约 10MB），之后离线可用
- 双行牌已支持自动拆行，但极端倾斜/污损样本仍可能识别失败
- 军牌 / 警牌（白牌特种牌）v2.3.2 已补齐语法层（警/军队/使领馆/武警新旧式
  严格校验、后缀与字头形近字纠错、序号位 I/O 修正）与判色层（红色字符信号
  识别阴影白底警牌），但**末位红字（警/军字头）的 OCR 漏检仍在**——
  小字号红色字符是通用 OCR 模型的弱区，`鄂F8888警` 可能读成 `鄂F8888`
- 新能源牌与普通民用牌在 8 位串形上存在固有歧义（`京AQ7797Q` 与 `京AD77972`
  串形无法区分），依赖 OCR 置信度与多帧投票消解
- **多进程批量（`--workers`）未在真实多核机器上验证过收益**：开发/CI 环境 CPU
  配额受限（纯 CPU 计算 4 进程 0.49x），因此"并行更快"这个结论**我们没有证到**。
  功能本身完整且有测试（顺序、隔离、ROI 取舍都有断言），但请自行用
  `tools/bench_workers.py --cpu-control` 在目标机器上确认
- **定位准确率没有 CI 门槛**（原因见上）；现有的是"行为不变"快照，能拦住回归，
  但拦不住"一直都在错"。阈值调参已用弱监督评测复核过一轮
  （`tools/eval_weak.py`，调参段 + 留出集双重验证）：三组候选参数无一在
  留出集复现改善，**维持默认参数**，过程记录见 `docs/UPGRADE.md` 第九节
- 摄像头实时识别已做到采集与识别真解耦（双线程），但 GUI 路径**没有**单元测试
  覆盖（Qt 线程测试成本高），只有源码级回归护栏

## 许可

MIT（与原项目一致）
