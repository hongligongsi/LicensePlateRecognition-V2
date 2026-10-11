# 升级说明：原版 → v2

对照原项目 [hongligongsi/License-Plate-Recognition](https://github.com/hongligongsi/License-Plate-Recognition)（最后提交 2021-01），逐项说明 v2 改了什么、为什么。

原版完整源码备份在本目录：`original_predict.py`、`original_surface.py`、`original_config.js`。

## 1. 环境：从"装不上"到开箱即用

| 项 | 原版 | v2 |
|---|---|---|
| Python | 3.4.4（2014 年 EOL）| ≥ 3.10 |
| OpenCV | 3.4 | **4.x（必须 <5，见下）** |
| numpy | 1.14 | ≥ 1.24（2.x 兼容）|
| 界面 | tkinter + PIL 5 | PySide6 ≥ 6.5 |

**关键坑：OpenCV 5.0 已移除 `cv2.ml.SVM_load`**（`cv2.ml` 模块在 5.0 里是空壳），原版模型彻底无法加载。v2 在 `CVEngine.is_available()` 里探测此情况并给出明确安装指引，`pyproject.toml` 锁定 `opencv-python-headless>=4.10,<5`。

## 2. 算法问题逐条修复（predict.py → engines/）

| # | 原版问题 | v2 方案 |
|---|---|---|
| 1 | 绝对像素参数（`np.ones((20,20))`、`Min_Area=2000`、`row_num_limit=21`），换分辨率即失效 | `locate.py` 全部改为相对比例，随图尺寸自适应 |
| 2 | `config.js` 实为 JSON、需重启生效、两套参数硬编码二选一 | `config.py` dataclass + `config.json` + 环境变量，运行时可调 |
| 3 | 车牌倾斜矫正用仿射变换猜正负角度，斜切车牌容易切歪 | `locate._warp_plate` 透视变换 + 标准四点排序 |
| 4 | 颜色统计用逐像素 `.item(i,j)` 双层循环（几十万次 Python 调用）| `cv2.inRange + countNonZero` 向量化，提速约 50-100 倍 |
| 5 | `find_waves` 循环外引用循环变量 `i`（line 38）、空直方图越界 | `segment.find_waves` 重写，边界严谨 |
| 6 | 字符数不足 6 直接放弃整个候选；汉字合并阈值 0.6 靠拍脑袋 | `segment._merge_to_count/_split_to_count` 动态对齐到 7 字符 |
| 7 | `np.int32(bin_n*ang/(2π))` 角度恰为 2π 时 bin=16 越界，特征维度悄悄漂移 | `svm_classifier.preprocess_hog` 里 `np.clip` 修正 |
| 8 | 多车牌场景 `break`，只返回第一个命中 | `cv_engine.predict` 处理全部候选，按合法性评分排序 |
| 9 | 无置信度，只有硬标签；未知标签直接 `chr()` 输出乱码 | SVM 决策值 softmax 出字符级置信度；未知标签丢弃并记日志 |
| 10 | 工作目录耦合：`open('config.js')`、`os.walk("train\\chars2")` 相对路径 | 模型路径由 `MODELS_DIR` 解析，任意 CWD 可运行 |
| 11 | `cv2.findContours` 用 try/except 兼容 2.x 返回值 | OpenCV 4.x 固定两返回值 |
| 12 | `np.int0`（numpy 2.x 已删除）、`Image.ANTIALIAS`（Pillow 10 已删除）| `boxPoints` 直接用 float、`Image.LANCZOS` / Qt 缩放 |
| 13 | 失败原因只有 `print("peak less 1")` 这类碎片输出 | logging 体系 + 结果 `meta` 字段，界面可见运行日志 |

## 3. 新增能力

- **深度学习引擎**（`deep_engine.py`）：RapidOCR（ONNX，CPU 友好）/ PaddleOCR 双后端，OCR 结果按"车牌语法 + 同行聚类"重组为车牌号。缺失依赖时优雅降级，不拖累 CV 引擎
- **引擎注册表**（`registry.py`）：`register_engine` 装饰器扩展第三方引擎
- **统一数据契约**（`types.py`）：`PlateResult` 带 bbox / roi / confidence / 字符级置信度
- **车牌语法后处理**：0↔O、1↔I 混淆修正 + 正则校验 + 候选评分
- **标注图导出**：车牌框 + 中文标签（`cv2.putText` 不支持中文，改用 Pillow）

## 4. 界面重写（surface.py → ui/）

| 原版问题 | v2 |
|---|---|
| 识别期间画面冻结，只能"每 2 秒识别一帧" | 采集与识别线程分离（`CameraWorker`），画面持续流畅 |
| 子线程直接调 Tkinter 控件（非线程安全，偶发崩溃）| Qt 信号槽跨线程通信 |
| 线程退出 `join(2.0)` 超时强杀 | `stop()` + `wait(2000)` 优雅退出，超时才 terminate |
| 一次只能选一张图 | 多选 / 整个文件夹批量，带进度条 |
| 识别结果即用即丢 | 历史表格 + CSV 导出 |
| 无结果可视化 | 车牌框标注、ROI 预览、颜色卡片、逐字符置信度 |
| 高分屏发虚 | Qt DPI 自适应 + QSS 主题 |

## 5. 实测对比（samples/ 两张 4K 样张）

| 指标 | 原版（v2 流水线下运行 SVM）| v2 深度引擎 |
|---|---|---|
| 1.jpg 京E51619 | ✗（定位到但字符全错）| ✓ 置信度 0.78 |
| 1.jpg 背景车牌 京PAP315 | ✗ 未发现 | ✓ 置信度 0.84 |
| 2.jpg 京AD77972 | ✗ | ✓ 置信度 0.87 |
| 单帧推理（已加载，1080p 级别）| ~25 ms | ~300-500 ms（首次 3s 含模型初始化）|

结论：传统 CV 引擎保留了原项目"轻、快、可解释"的教学价值；需要生产精度时切 `deep` 引擎。

## 6. 保留的原版资产

- `models/svm.dat`、`models/svmchinese.dat`：原版权重原样使用（HOG 特征实现逐位对齐，实测 `getVarCount()==64` 一致）
- `samples/1.jpg`、`samples/2.jpg`：原项目 test 图
- 定位算法思想（边缘+形态学 → 长宽比筛选 → HSV 颜色二次定位 → 投影波峰分割）与原版一致，是 EasyPR 一脉的思路

---

# v2.1 升级记录：更精准、更快速、更安全、更全面

v2.0 完成了"能跑起来 + 现代化"，v2.1 解决的是"跑得好、跑得快、跑得稳、跑得广"。

## 1. 更快速 —— 跳过 OCR 检测网络

**问题**：v2.0 的 `deep` 引擎把整张图送进 OCR 流水线，其中文本检测（DBNet）是最大的耗时来源（实测约 317 ms），而 1080p 监控图里真正的车牌只占极小面积。

**关键发现**：我们已经用轻量 CV 定位（约 50 ms）拿到了车牌候选区，**根本没有必要再跑一遍通用文本检测**。RapidOCR 提供了绕过检测网络、直接把整块图当单个文本框送入识别网络的入口。

**改动**：

| 项 | v2.0 | v2.1 |
|---|---|---|
| OCR 路径 | 全图 → DBNet 检测 → 识别 | CV 定位 → 候选区 → **直接识别**（跳过 DBNet）|
| 单张耗时（中位）| ~1600 ms | **194 ms** |
| QPS | 0.6 | **3.8** |

```python
# deep_engine.py — 绕过检测网络
_, crops = self._ocr.get_boxes_img_without_det(img, img.shape[0], img.shape[1])
results, _elapse = self._recognizer(crops)
```

**配套机制**：
- 超分到目标字符高 48px（`_upscale`）——识别网络的输入高度固定，原图车牌字符常只有 20~30px
- USM 锐化（`_sharpen`）增强笔画对比
- 多尺度投票 + **早停**：语法合法性 ≥ 0.6 且票数占比 ≥ 0.5 即结束，典型只跑 1~2 轮
- 兜底：候选区全部失败时回退到全图 OCR 路径

## 2. 更精准 —— 语法规则 + 多尺度投票 + 跨帧跟踪

**新增 `plate_rules.py`（车牌语义规则引擎）**

| 能力 | 说明 |
|---|---|
| `identify_format` | 12 种格式正则匹配（顺序敏感：具体在前） |
| `correct` | 保守纠错：发牌机关位数字→字母（`0`→`D`）、I/O 不合法（`I`→`J`、`O`→`D`）、末位字母降级 |
| `plate_score` | 合法性评分 0~1，用于候选排序与投票加权 |
| `describe` | 输出类型/省份/发牌机关，供审计展示 |

**新增 `tracker.py`（跨帧多车牌跟踪）**

视频场景下用 IoU + 号码一致性把连续帧关联成轨迹，多帧投票纠正单帧偶发错误。合法号码 1 次即确认（几乎不可能是误检），可疑号码需多帧确认。

**修复的精度缺陷**：
- `_dedupe` 只按框重叠去重 → 相邻/重叠的不同车牌被错误合并（漏报）。改为「号码相同才合并，号码不同即使框重叠也保留两者」
- `plate_rules` 三处规则漏洞：军队牌 `京V12345` 被归为普通民用（新增旧式军牌格式）、使领馆 `使123456` 判为非法（新增格式）、`_best_letter('I')` 返回空（补映射）
- 新能源牌与普通牌的 8 位串形歧义：从"能区分"的错误假设改为显式承认歧义，交由 OCR 置信度与多帧投票消解

## 3. 更安全 —— 输入防护 + 端到端脱敏

**新增 `utils/security.py`**

| 类别 | 能力 |
|---|---|
| 资源限制 | `Limits`：文件 32MB / 像素 5000 万 / 单边 20000px / 最小尺寸 / 批量上限 / 扩展名白名单 |
| 输入校验 | `check_file`（读前）、`check_image`（解码后）→ 防解压炸弹与畸形尺寸 |
| 隐私脱敏 | `PrivacyPolicy` + `mask_plate` + **`scrub_value`（递归）** + `MaskingFilter`（日志）|
| 抗滥用 | `RateLimiter` 令牌桶限流 |
| 完整性 | `sha256_file` / `verify_model` 模型哈希白名单 |
| 路径安全 | `safe_output_path` 防路径穿越 |

**关键教训：脱敏必须递归，且要覆盖所有落盘点。**

修复过程中连续发现三处旁路泄露，均已被单元测试守住：

1. `RecognizeOutcome.to_dict()` 只对 `text` 打码，但 `meta.variants` 里含明文候选号码 → 改为对整棵 payload 树过 `scrub_value`
2. `BatchOutcome.save_json()` 的 `unique_plates` 汇总字段未打码 → 等于把 items 里打掉的明文又列一遍
3. `BatchOutcome.save_csv()` 的「来源」列未打码，而文件名常含车牌号（`京AD77972_20260101.jpg`）→ 来源列同样过 `mask_plate`
4. CLI 控制台输出（`✓ 2.jpg: 京AD77972`）未受策略约束 → 统一走 `mask_plate`

```python
def scrub_value(value: Any, policy: PrivacyPolicy) -> Any:
    """递归清洗任意嵌套结构中的车牌号。"""
    if not policy.mask_output:
        return value
    if isinstance(value, str):
        return MaskingFilter._PLATE_RE.sub(lambda m: mask_plate(m.group(0), policy), value)
    if isinstance(value, dict):
        return {k: scrub_value(v, policy) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        scrubbed = [scrub_value(v, policy) for v in value]
        return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
    return value
```

## 4. 更全面 —— 类型 / 输入 / 输出

| 维度 | v2.0 | v2.1 |
|---|---|---|
| 车牌格式 | 隐含在定位逻辑里 | **12 种显式格式定义** + 9 类验证通过 |
| 双行牌 | 不支持 | **自动检测拆行 + 横向拼接**（`_to_single_line`） |
| 输入来源 | 图片 / 目录 | + 递归目录 / 视频文件 / 摄像头流 / numpy 数组 |
| 输出格式 | 控制台 / 标注图 | + JSON / CSV / 标注视频 |
| 高层 API | 引擎级 `get_engine()` | + `Recognizer` / `RecognizeOutcome` / `BatchOutcome` |
| CLI | 3 个子命令 | **7 个**（+ `engines` / `check` / `models` / `config`）|

## 5. 测试与基准

| 项 | v2.0 | v2.1 |
|---|---|---|
| 单元测试 | 23 | **179** |
| 覆盖模块 | types/config/engines/segment | + security / plate_rules / tracker / api / deep 预处理 |
| 基准脚本 | 无 | `tools/benchmark.py`（引擎性能 + 安全逐项检查 + 覆盖统计）|

```bash
pytest tests/ -q
python tools/benchmark.py --images samples --engine deep,cv --repeat 3 --json out/bench.json
```

实测基准（`samples/`，每张重复 3 次）：

| 引擎 | 冷启动 | 中位耗时 | P95 | QPS | 命中率 |
|---|---|---|---|---|---|
| `deep` | 541 ms | 194 ms | 507 ms | 3.8 | 100% |
| `cv` | 143 ms | 28 ms | 31 ms | 3.8→35.5 | 100%（但号码错）|

**重要结论**：`cv` 引擎比 `deep` 快约 7 倍，但其 SVM 对**任何**输入都输出置信度 1.0（严重过自信），5 个输出的号码全部错误。这从数据上确认了：定位其角色为「快速筛候选 + 提供 bbox/颜色」，**精度必须由深度学习引擎承担**。默认引擎因此从 `cv` 改为 `cascade`（CV 快速定位 → 深度精识别）。

## 6. 环境约束（踩坑记录）

| 问题 | 现象 | 解决 |
|---|---|---|
| OpenCV 5.0 移除 `cv2.ml` | `module 'cv2.ml' has no attribute 'SVM_load'` | 锁 `opencv-python-headless>=4.10,<5`（5.0 的 `cv2.ml` 是空壳） |
| rapidocr 连带升级 OpenCV | 装 rapidocr 后 opencv 被顶到 5.0，SVM 又崩 | 卸载两者后重装 headless 4.x |
| `RapidOCR(det_limit_side_len=960)` | `KeyError: 'model_path'` | 该版参数名不同，改为多组参数 try/except，失败退回无参构造 |
| `get_boxes_img_without_det` | 首次调用 `AssertionError` | 返回值是 `(dt_boxes, img_crop_list)` 二元组，需 `_, crops = ...` |
| `rapidocr.text_rec` | `AttributeError` | 实际属性名为 `text_recognizer` |
| `cv2.putText` 不支持中文 | 标注图显示 `???` | 改用 Pillow + `msyh.ttc` 绘制标签 |
