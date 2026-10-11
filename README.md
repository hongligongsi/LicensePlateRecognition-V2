# 车牌识别系统 v2.1（License-Plate-Recognition 升级版）

用 Python + OpenCV + 深度学习做的**中国车牌识别**，支持蓝牌 / 黄牌 / 绿牌（新能源）/ 白牌 / 黑牌。

本项目是 [hongligongsi/License-Plate-Recognition](https://github.com/hongligongsi/License-Plate-Recognition) 的现代化升级版。原项目最后一次提交停留在 2021 年，其依赖（Python 3.4 / OpenCV 3.4 / numpy 1.14）在如今的机器上已无法安装。v2 在保留原算法思想的基础上全面重构，详见 [UPGRADE.md](docs/UPGRADE.md)。

> **v2.1 主题：更精准、更快速、更安全、更全面。** 详见下方「v2.1 四大升级」。

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
| `verify_model` / `sha256_file` | 模型哈希白名单校验，检测模型篡改 |
| `safe_output_path` | 防路径穿越（`../../etc/passwd` 这类会被拦回基准目录） |

**脱敏是端到端生效的**：控制台输出、JSON、CSV、嵌套 `meta` 全部覆盖，均有单元测试守住。

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

- **车牌类型**：普通民用 / 新能源 / 警用 / 军队 / 武警 / 使领馆 / 挂车 / 教练车 / **危险品运输** / 港澳入出境 / 短牌，共 **13 种格式定义**
- **双行牌支持**：自动检测「上短下长」的双行排布，拆行后横向拼接成单行再识别（商用车前后牌、摩托车牌）
- **输入来源**：图片 / 目录（可递归）/ 视频文件 / 摄像头流 / numpy 数组
- **输出格式**：控制台 / JSON / CSV / 标注图片 / 标注视频

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
lpr video parking.mp4 --stride 10 --output out.mp4 # 视频识别（跳帧 + 标注输出）
lpr check 京A1234危 京AD77972 --fix               # 校验号码合法性并纠错
lpr models --verify                               # 查看模型信息与哈希
lpr gui                                           # 图形界面
```

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
- **摄像头实时识别**：采集与识别分离，画面流畅不掉帧（原版每 2 秒冻结一次）
- 引擎下拉框热切换，不可用的引擎自动置灰并给出安装提示

## 项目结构

```
src/lpr/
├── types.py            # PlateResult / PlateColor 统一数据契约
├── config.py           # 集中配置（JSON + 环境变量，含 SecurityConfig）
├── api.py              # 高层 API：Recognizer / RecognizeOutcome / BatchOutcome
├── plate_rules.py      # 车牌语法规则：格式识别、纠错、合法性评分
├── tracker.py          # 跨帧多车牌跟踪（IoU + 号码一致性投票）
├── cli.py              # 命令行入口
├── engines/
│   ├── base.py         # 引擎抽象接口
│   ├── registry.py     # 引擎注册表（可扩展第三方引擎）
│   ├── locate.py       # 车牌定位（向量化、透视矫正、颜色判定）
│   ├── segment.py      # 字符分割（投影波峰 + 动态对齐到 7 字符）
│   ├── svm_classifier.py  # SVM 字符分类（兼容原版模型 + 置信度）
│   ├── cv_engine.py    # 传统 CV 引擎
│   └── deep_engine.py  # 深度学习引擎（定位引导 + 纯识别加速 + 双行牌）
├── ui/
│   ├── window.py       # PySide6 主窗口
│   └── worker.py       # 识别/摄像头线程
└── utils/
    ├── image.py        # 图像 IO（中文路径 / EXIF）、标注绘制（Pillow 中文字体）
    ├── security.py     # 安全层：Limits / PrivacyPolicy / 脱敏 / 限流 / 模型校验
    └── logging_setup.py
tools/benchmark.py      # 基准脚本（量化四项升级收益）
tools/make_color_samples.py  # 合成六色样张，核对底色判定
models/                 # 预训练 SVM 模型（原版权重）
samples/                # 测试样张
tests/                  # 232 项 pytest 测试
docs/UPGRADE.md         # 与原项目的详细对比
```

## 测试与基准

```bash
pytest tests/ -q                                    # 232 项单元/集成测试
python tools/benchmark.py --images samples --repeat 3 --json out/bench.json
```

基准脚本会输出引擎性能（耗时/QPS/命中率）、安全防护逐项检查、覆盖范围统计，可直接引用。

## 已知限制

- 传统 CV 引擎的 SVM 字符识别精度有限（原版模型训练样本太少，中文省份尤其差），且对任意输入都输出置信度 1.0 —— **需要精度请用 `deep` 或 `cascade`**
- 深度引擎首次运行需下载 OCR 模型（约 10MB），之后离线可用
- 双行牌已支持自动拆行，但极端倾斜/污损样本仍可能识别失败
- 军牌 / 警牌（白牌特殊字符）识别率一般
- 新能源牌与普通民用牌在 8 位串形上存在固有歧义（`京AQ7797Q` 与 `京AD77972` 串形无法区分），依赖 OCR 置信度与多帧投票消解

## 许可

MIT（与原项目一致）
