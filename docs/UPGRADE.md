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

---

# v2.1.1 补充：五种底色 + 危险品运输牌

## 1. 五种底色（黑/蓝/绿/白/黄）

| 枚举 | 值 | 中文名 | 适用车辆 |
|---|---|---|---|
| `PlateColor.BLACK` | `black` | 黑色 | 使领馆、境外车辆 |
| `PlateColor.BLUE` | `blue` | 蓝色 | 小型汽车（最常见） |
| `PlateColor.GREEN` | `green` | 绿色 | 新能源、农用 |
| `PlateColor.WHITE` | `white` | 白色 | 警用、军队、特种车辆 |
| `PlateColor.YELLOW` | `yellow` | 黄色 | 大型汽车、教练车 |

配套新增：
- `COLOR_NAMES`：五种底色的规范中文名元组
- `COLOR_ALIASES` + `parse_color()`：接受中文全称/简称/英文（`黄色`/`黄`/`yellow`），供 CLI `--color` 与界面下拉使用
- `COLOR_SWATCH`：界面色块，补齐五种底色

## 2. 危险品运输牌（黑底黄字）

危险品运输车辆悬挂 **黑底黄字** 牌。难点在于它与普通黄牌（**黄底黑字**）的
底色与字色**恰好相反**，仅凭"存在黄色像素"无法区分。

### 判定策略：明暗极性分析

`detect_hazmat()` 同时要求三个条件成立：

| 条件 | 阈值（可配） | 作用 |
|---|---|---|
| 黑色主体占比 ≥ 45% | `hazmat_bg_ratio` | **排除黄底黑字的普通黄牌** |
| 黄色前景占比 ≥ 3% | `hazmat_fg_ratio` | **排除黑底白字的普通黑牌** |
| 黄/(黄+黑) 比值 ≥ 6% | `hazmat_fg_share` | 确保黄字只是少量前景，而非大片黄底 |

第 1 条排除普通黄牌、第 2 条排除普通黑牌，两条合起来才能唯一锁定黑底黄字。

### 实测区分效果

| 合成车牌 | 底色 | 字色 | 判定结果 |
|---|---|---|---|
| 危险品运输 | 黑 | 黄 | **危险品运输牌** ✓ |
| 普通黄牌 | 黄 | 黑 | 黄牌 ✓（不误判为危险品）|
| 普通黑牌 | 黑 | 白 | 黑牌 ✓（不误判为危险品）|
| 蓝牌 | 蓝 | 白 | 蓝牌 ✓ |
| 绿牌 | 绿 | 黑 | 绿牌 ✓ |
| 白牌 | 白 | 黑 | 白牌 ✓ |

### 号码格式

新增 `危险品运输` 格式（省 + 字母 + 4~5 位数字 + `危`），必须在普通民用之前匹配，
否则 `京A1234危` 会被吞成非法长度。

```bash
lpr check 京A1234危      # → ✓ 合法 类型=危险品运输
```

### 已知限制

- **灰度输入无法判定**：灰度图丢失色相，"黑底黄字"与"黑底白字"不可区分。
  `detect_hazmat` 对灰度输入保守返回 `False`（宁可漏判为黑牌，也不凭亮度猜出错误结论）。
- 极端光照下黑底可能被压到纯黑、黄字泛白，此时极性特征减弱，可能退化为黑牌。

## 3. OCR 文本抽取正则修正

原 `_PLATE_TEXT` 正则不认末位中文标识，会把 `京A1234危`、`京A1234警` 截断。
已补充末位标识字符集：

```python
_PLATE_TEXT = re.compile(
    r"[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]"
    r"[A-Z][A-Z0-9]{3,6}[警挂学危港澳]?"
    r"|[军空海北沈兰济南广成使][A-Z][A-Z0-9]{3,6}[警挂学危]?"
)
```

## 4. 按底色筛选

CLI：
```bash
lpr predict ./photos/ --color 危险品          # 只保留危险品运输车
lpr predict ./photos/ --color 黄色,危险品      # 多底色（逗号分隔）
lpr predict ./photos/ --color blue            # 中英文均可
```

API：
```python
batch.keep_colors({parse_color("危险品")})     # 就地过滤，返回 self 可链式调用
batch.color_summary()                         # {"危险品运输牌（黑底黄字）": 2, "黄牌": 5}
```

## 5. 测试与基准

- 新增 `tests/test_plate_color.py`（50 项）：枚举完整性、中英文别名解析、五种底色判定、
  危险品与黄牌/黑牌的区分、灰度与空输入健壮性、号码格式、OCR 正则抽取、批量过滤 API
- 测试总数 179 → **229**
- `tools/benchmark.py` 新增底色覆盖检查：**13 种格式 / 10 类类型 / 6 种底色全部通过**

## 6. 修复：HAZMAT 缺少 HSV 区间导致投影裁剪崩溃

新增 `HAZMAT` 枚举后，`refine_by_color()` 走 `_color_mask(hsv, color)` 时
因 `_COLOR_RANGES` 里没有 `HAZMAT` 键而抛 `KeyError`，**整条识别管线中断**。

修复方式：引入 `_REFINE_BASE` 映射，把 `HAZMAT` 归到 `BLACK` 基准：

```python
_REFINE_BASE = {PlateColor.HAZMAT: PlateColor.BLACK}
```

理由是语义正确的：投影裁剪要剔除的是**非车牌背景**，而危险品牌的"底"是黑色，
黄色字符属于前景（和黑牌的白字一样），不该参与裁剪基准。

同时补充回归测试 `test_refine_by_color_supports_hazmat` 与
`test_refine_by_color_all_colors_do_not_crash`（遍历全部枚举值），
防止将来再加底色时重演同类崩溃。

## 7. 新增工具的完整清单

| 工具 | 用途 |
|---|---|
| `tools/benchmark.py` | 性能 + 安全 + 覆盖统计（13 格式 / 10 类型 / 6 底色）|
| `tools/make_color_samples.py` | 合成六种底色样张，逐张核对判定结果是否与预期一致 |

---

# v2.2.0：12 大类车牌全覆盖 + 版式（单行/双层）维度

用户点名的 12 类：单行蓝牌、单行黄牌、新能源、白色警用、教练车、武警、
双层黄牌、双层白牌、使用馆（使领馆）、港澳粤Z、双层绿牌、民航。

核查结论：前 11 类的**号码格式**均已支持，缺的是两块——

1. **民航车牌完全没有代码路径**（`民航A0001` 判为"未知/不合法"）
2. **"双层"没有表达维度**：双层黄/白/绿牌与单行同号码（`京A12345` 既是
   单行蓝牌也可能是双层大车后牌），仅凭号码与颜色无法区分

## 1. 民航号牌（依据 CCAR-331SB-R1）

* 格式：`民航` + 管理局字母 + 4 位数字，如 `民航A0001`
* 字母表：A=华北局 B=华东局 C=中南局 D=西南局 E=西北局 F=东北局 G=新疆局
* 外观：绿底白字白框线，440×140mm；用于机场摆渡车、专用作业车、登机梯车

代码改动：

```python
# plate_rules.py —— FORMATS 最前新增
CIVIL_AVIATION_HEADS = "ABCDEFG"
PlateFormat("民航", re.compile(rf"民航[{CIVIL_AVIATION_HEADS}][{D}]{{4}}$"), 6, 6),

# deep_engine.py —— _PLATE_TEXT 抽取正则新增分支（"民"不是省份简称，
# 现有规则匹配不到这两个字开头的文本）
r"|民航[A-G][0-9]{3,4}"
```

`describe()` 新增 `is_civil_aviation` 字段。

## 2. 版式维度 PlateLayout

新增与底色**正交**的枚举：

```python
class PlateLayout(str, Enum):
    SINGLE = "single"    # 单行 —— 绝大多数乘用车
    DOUBLE = "double"    # 双层 —— 大车后牌、挂车、双层绿牌
    UNKNOWN = "unknown"
```

* `PlateResult` 新增 `layout` 字段，`to_dict()` 输出 `layout`/`layout_label`
* `parse_layout("双层")` 供 CLI/界面解析用户输入
* `plate_rules.describe()` 新增 `layout_hint`（各格式的版式先验兜底）

## 3. 版式判定：split_double_line + measure_layout

把原来 `_to_single_line` 里的行拆分逻辑提取为独立的 `split_double_line()`，
并加上**宽高比门槛**（`_DOUBLE_MAX_ASPECT = 2.35`）：

* 单行牌 440×140mm ≈ 3.14:1，双行牌 440×220mm ≈ 2.0:1
* 只有「投影切出恰好 2 段行带」**且**「宽高比 ≤ 2.35」才判双层
* 第 2 条很关键：带上下装饰边框的单行牌投影上也会切出两段，
  没有宽高比门槛会被误拆成乱码（真实踩坑，见回归测试）

`measure_layout(roi)` 组合两个证据源：行分割成功 → double；
否则按宽高比先验（≤2.35 → double，≥2.6 → single，之间 → unknown）。
识别管线对每个候选自动回填 `result.layout`。

## 4. 测试与基准

* 新增 `tests/test_twelve_kinds.py`（36 项）：12 类逐项类型判定、民航字母表
  边界（A-G 合法 / H 非法、位数 3/4/5）、版式枚举与解析、`describe` 的
  layout_hint 全格式覆盖、合成单行/双行图的版式判定、带边框单行牌不误判回归
* 修正旧用例 `test_to_single_line_splits_two_bands`：原样图 200×80（2.5:1）
  超出真实双行牌比例，被新的宽高比门槛正确拦截 —— 改为 2:1 真实比例
* 测试总数 232 → **286**，全部通过
* 真实样张回归：与 v2.1.1 基线完全一致（9 张同样 7 命中、号码逐一相同），无性能回退

---

# v2.2.1：借鉴 YOLO26-plate 的双层牌处理

参考项目：[we0091234/Chinese_license_plate_detection_recognition](https://github.com/we0091234/Chinese_license_plate_detection_recognition)
（现已升级为 YOLO26-plate，源码形态见 `yolo26-plate-main`）。

它的 12 类支持清单与我们 v2.2 完全一致，值得借鉴的是**双层牌的处理方式**。

## 1. 参考项目的算法架构

```
输入图
  │
  ├─(1) YOLO26-pose 检测器
  │       ├─ bbox（车牌框）
  │       ├─ keypoints（4 个角点，kpt_shape=[4,2]）
  │       └─ cls（0=single 单层 / 1=double 双层）   ← 关键
  │
  ├─(2) four_point_transform 用 4 角点做透视矫正
  │
  ├─(3) 若 cls==1：get_split_merge 按固定比例切分并拼接
  │       img_upper = img[0 : 5/12*h]     ← 上行 41.7%
  │       img_lower = img[1/3*h : ]       ← 下行 66.7%
  │       上行 resize 到下行尺寸后 hstack
  │
  └─(4) CRNN 识别（myNet_ocr_color）
          输入 168x48，mean=0.588 std=0.193
          双头输出：字符序列 + 颜色分类（5 类）
          字符表 78 类：
            京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新
            学警港澳挂使领民航危 0-9 A-Z(去IO) 险品
```

三个关键设计：

1. **检测阶段就判单/双层**（`cls` 类别），而不是识别后再猜
2. **固定比例切分双层牌**（5/12、1/3），不依赖投影分析
3. **字符识别与颜色分类共享 backbone**，颜色作为辅助头并行输出

## 2. 我们吸收了哪一条，为什么

### 采纳：检测阶段先判版式（对应 v2.2 的 `measure_layout`）

原实现对**每个**候选都尝试双行拆分；我们改成先用 `measure_layout(roi)`
判定版式，只有 `double` 才生成拼接形态：

```python
layout = PlateLayout(measure_layout(roi))
variants = _to_single_line(roi) if layout is PlateLayout.DOUBLE else [roi]
```

实测收益（9 张样张、43 个候选）：

| 版式判定 | 候选数 | 是否走拼接 |
|---|---|---|
| single | 30 | 否（省掉双倍识别开销）|
| double | 7 | 是 |
| unknown | 4 | 否 |

即 **70% 的候选不再做无谓的双行尝试**。

### 不采纳：固定比例切分

参考实现用 `5/12` 和 `1/3` 两个魔数切分。这在它的 YOLO 检测框已经
做过透视矫正、尺寸归一的前提下够用，但对我们**依赖 CV 定位**的管线不合适：
定位框的宽高比与留白每次都不一样，固定比例会把字符切掉半截。

我们改成**裁左右边框 + 行投影 + 宽高比门槛**，不依赖固定比例。

## 3. 由此发现并修复的真实 bug：边框桥接

原 `split_double_line` 只按"投影切出恰好 2 段行带"判断，实测**带边框的
双行牌永远切不出来**：车牌四周的框线（每行仅 ~10px 前景）贡献的投影，
把上下两行字符桥接成了一整段连续行带。

诊断（440x220 合成双行牌，外框 3px）：

```
行带切分结果: [(1, 219)]   ← 只有 1 段，整块连在一起
   5  438  ← 上边框（满行）
  25  206  ← 上行字符
  95   10  ← 左侧框线（把空白桥接掉了！）
 130  280  ← 下行字符
 215  438  ← 下边框（满行）
```

修复三件套：

1. **`_band_profile`**：算投影前先左右各裁 8%（`_BORDER_TRIM`），
   去掉竖直框线的贡献
2. **`_find_bands`**：桥接宽度小于 1.2% 高度的空白（`_BAND_BRIDGE_RATIO`），
   容忍框线的抗锯齿断开
3. **单行牌防线**：某段行带高度占比 > 50%（`_SINGLE_BAND_RATIO`）→
   判定为单行字符，不拆。这是"带上下边框单行牌被误拆"的最后一道防线

另加**行带高度相似性**检查（`_BAND_HEIGHT_SIMILARITY = 0.55`）：
真实双行牌上下行字号一致，上薄下厚多半是单行牌上方的装饰条。

修复前后对比：

| 输入 | 修复前 | 修复后 |
|---|---|---|
| 双行 440x220 + 外框 | 切不出（1 段）| **正确切出 2 段** → double |
| 单行 440x140 + 外框 | single ✓ | single ✓（未回退）|
| 单行 440x140 无框 | single ✓ | single ✓ |

## 4. 踩坑：变量遮蔽导致 layout 被塞进 ndarray

`_recognize_candidates` 里为了采纳"检测阶段判版式"，新增了局部变量
`layout`；而内层已有 `for layout in layouts:` 循环，**把枚举值覆盖成了
图像数组**。后果很隐蔽：

* 识别本身完全正常，控制台输出看不出问题
* 只在**写报告**时 `to_dict()` 处炸掉：`'numpy.ndarray' object has no
  attribute 'value'`
* 而且只在**有识别结果时**才炸 —— 空图不触发

修复：内层循环变量改名 `variant`、`variants`。并新增
`TestResultFieldIntegrity::test_layout_is_enum_on_engine_output`
回归用例，要求引擎产出的每个结果的 `layout` 必须是 `PlateLayout` 枚举
且 `to_dict()` 可序列化。

> 教训：给内部循环变量命名时要检查是否与外层新变量撞名；
> 这类错误不会被类型检查器发现（都是合法的 Python），
> 必须在真实数据上跑端到端才暴露。

## 5. 测试与基准

* 新增 `TestDoubleLineWithFrame` 6 项：带框双行能切分、带框单行不误拆、
  行高归一化、上薄下厚被拒
* 新增 `TestResultFieldIntegrity` 1 项：字段类型完整性
* 测试总数 286 → **293**，全部通过
* 真实样张回归：7/9 命中、8 个号码与基线**逐字一致**，无回退
* 性能：候选版式分布 single 30 / double 7 / unknown 4

---

## v2.3.0 —— 高价值点全面落地：准确率与稳定性

> 关键词：宽松召回 / 严格校验分层、形近字纠错、双层垂直合并、宁缺勿错、输入归一化

### 一、召回分层（借鉴 YOLO26-plate 的正则分层思想）

车牌文本抽取从"一条正则打天下"升级为 **四级流水**（`_extract_plate`）：

| 级 | 策略 | 解决的问题 |
|---|---|---|
| 1 | 严格召回（`_PLATE_TEXT`，首位必须是真省份） | 筛掉绝大多数非车牌文本 |
| 2 | 宽松召回（`loose_match`，首位允许**省份形近字**） | `凉A12345` 此前在抽取阶段就被丢弃，纠错逻辑根本没机会执行 |
| 3 | **先纠正再召回**（`correct()` 后再走一遍 1/2） | `京三A5331` —— "三"是 E 的汉字误读，中间位认错整条作废 |
| 4 | 纯字母数字兜底 + **纠正后必须合法闸门** | 车身贴纸 `JINCBFI`、半截号码 `A5331` 此前被当车牌误报 |

配合的规则层改动：

* `PROVINCE_LOOKALIKES`（约 60 条）：凉→京、泸→沪、律→津、奥→粤、淅→浙……
  端到端救回率 **9/9**（改进前这类号码 100% 判无效丢弃）。
* `_best_letter` 新增 **三 → E**：中文 OCR 把 E 读成"三"极常见（字形几乎相同），
  双层黄牌的上牌行首当其冲。

### 二、双层牌垂直合并（新能力）

此前 `_group_fragments` 只做**同行水平**拼接，双层牌上下两行碎片无能为力：
`double_yellow.jpg` 全图 OCR 得到 `京三` + `A5331` 两个碎片，只能报出半截。

新增 `_group_double_fragments`：识别**上下排布**的碎片对 ——
水平重叠 ≥ 窄方 40%、垂直间隙 0.2~1.2 倍行高、两行高度比 0.5~2.0、
拼接后 `_extract_plate` 能抽出且纠正后合法（四重从严，防止拼凑车厢上无关文字）。

实测：`double_yellow.jpg` → **京EA5331 完整召回**（黄牌/普通民用，置信度 0.83）。

### 三、宁缺勿错：误报治理

`double_yellow.jpg` 此前的 `A5331`、`JINCBFI` 两条误报全部消除 ——
第 4 级闸门让"纠正后仍不合法"的文本自然出局。随之 `double_lv.png`（上半行
OCR 完全无输出，只有下半行 `49885` 半截）也不再硬报半截号码。

### 四、旧式军牌收窄（关键分类修复）

旧规则 `[省][VKBSLJNGC][0-9]{5}` 太宽：**皖K（六安）、粤V（揭阳）等都是
正常民用车牌**，导致 `皖K66666` 被误判为"军队"。收窄为无歧义的 **京V**
（这正是原注释声明的意图）：

* `京V12345` → 军队 ✓（保留）
* `皖K66666` → 普通民用 ✓（修复）
* `粤V12345` → 普通民用 ✓（修复）

### 五、稳定性：输入归一化 `ensure_bgr`

管线函数（版式测量/双层切分/颜色判定/危险品判定）此前对灰度图、四通道图、
float32/float64、含 NaN 数组直接抛 OpenCV 异常。新增
`utils.image.ensure_bgr` 统一归一化：

* `None` / 维度异常 / 单边过小 → `None`（调用方优雅降级）
* 灰度 → 三通道；BGRA → 去 alpha
* 浮点 → NaN/inf 置 0、裁剪 0~255、转 uint8

对抗性暴力测试（11 种病态图像 × 5 个管线函数 + 9 种垃圾文本 × 5 个规则函数）
**零未捕获异常**。`predict()` 对过小/异常图像的 `SecurityError` 受控拒绝保持不变
（这是安全设计的预期行为）。

### 六、真实样张回归对比

| 样张 | v2.2.1 | v2.3.0 | 变化 |
|---|---|---|---|
| double_yellow.jpg | `A5331`(半截误报) + `JINCBFI`(误报) | **京EA5331** 黄牌/普通民用 | ✅ 消除 2 误报 + 完整召回 |
| single_yellow.jpg | 皖K66666 黄牌/**军队** | 皖K66666 黄牌/普通民用 | ✅ 分类修复 |
| police.jpg | 鄂F08888 unknown | 鄂F08888 黑牌 | （v2.2 系已改） |
| 其余 6 张 | 与基线一致 | 与基线一致 | 无回退 |

命中率 7/9（v2.2.1 为"7 个号码但含 2 条误报"），且未识别的 2 张
（double_lv 上半行无 OCR 输出、xue.jpg 丢失省份"琼"）属于 OCR 感知能力边界，
规则层不瞎补 —— 宁缺勿错。

### 七、测试

* 新增 `tests/test_v2_3_quality.py` **49 项**：四级抽取、形近字召回、三→E、
  军牌收窄、垂直合并几何、`ensure_bgr` 病态输入、管线异常防护
* 测试总数 293 → **334**，全部通过

---

# v2.3.1 —— 修掉 4 个真 Bug + 补上工程化配套

这一版没有新功能，做的是**代码体检后兑现的修复**。起因是一次逐文件核查：
文档里承诺的能力，有几处在实现上并不成立。

## 一、Recognizer 静默忽略 config.security

`api.py` 里原本是：

```python
self.limits  = limits  or Limits()          # ← 库默认值
self.privacy = privacy or PrivacyPolicy()   # ← 库默认值
```

从不回落到 `config.security`。后果是 **`config.json` 的 `security` 段、
以及 `LPR_MASK_OUTPUT` 环境变量，只对 CLI 生效，走 Python API 时被完全忽略**。
实测（修复前）：

```
config 要求打码: True        → Recognizer 实际: False   ← 被忽略
config 要求上限 1MB          → Recognizer 实际: 32MB    ← 被忽略
```

修复后优先级为「显式传参 > `config.security` > 库默认值」，且 `mask_logs`
会在构造时幂等地装上日志脱敏过滤器。

## 二、GUI 完全没有接入安全层

`ui/window.py` / `ui/worker.py` 里**零** security 引用：

* `imread(path)` 裸调 —— 没有 `check_file`（不校验大小/像素/扩展名）
* CSV 导出是裸 `open(path, "w")`，直接写明文车牌号
* `SecurityContext.setup()` 从未调用 → 日志脱敏过滤器在 GUI 下没装
* 摄像头路径 `engine.predict(frame)` 也没有 `check_image`

也就是说 README 的「更安全：完整的输入防护与隐私脱敏」**在 GUI 路径上不成立**。
现在三条入口（CLI / Python API / GUI）共用同一份 `config.security`。

## 三、摄像头"采集与识别解耦"是假的

`CameraWorker.run()` 里：

```python
if now - last_predict >= self.interval and not self._recognizing:
    results = engine.predict(frame)   # ← 同步阻塞采集循环
```

`_recognizing` 只在 `__init__` 赋值过一次、**从此再没被读取过**（死逻辑）。
识别就在采集循环里同步跑，deep 引擎约 200ms/张 → 采集帧率从 ~30fps 掉到 ~5fps，
与 docstring 写的「采集循环里绝不进行识别」和 README 的「画面流畅不掉帧」都不符。

现在拆成两个线程：`CameraWorker` 只采集（循环里没有任何推理），
`_RecognizeThread` 只识别，且用「序号 + 最新帧引用」保证永远在**最新画面**上推理，
识别慢于采集时不积压帧。另外补了 `INFERENCE_LOCK` 把推理串行化 ——
引擎是跨线程共享的单例，批量识别与摄像头识别并发跑同一份模型并不安全。

## 四、模型哈希白名单形同虚设

`lpr models --verify` 原本输出：

```
✓ 未配置哈希白名单，跳过完整性校验
```

* `SecurityConfig` 里**根本没有 `model_hashes` 字段** —— 照它自己的建议也配不了
* `verify_model` 对未校验项返回 `True`，CLI 于是打出一个「✓」：
  **"没检查"显示成了"检查通过"**

现在：新增 `ModelStatus` 三态（`ok` / `skipped` / `mismatch` / `missing`）与
`ModelCheck`，跳过校验显示为「—」并明确说明；`SecurityConfig.model_hashes`
开箱内置随仓库分发的两个 SVM 模型哈希，`lpr models --verify` 立刻能真做校验。

## 五、顺带修掉的一致性 / 真错问题

| 问题 | 位置 | 说明 |
|---|---|---|
| `mode` 是死配置 | `config.py` | 只在 `validate` 里出现，从未被读取。现在赋予真实语义：`mode="cascade"` 时主引擎未命中会跨引擎兜底，并把 `default` 与 `mode` 的正交关系写清楚 |
| 引擎单例忽略新配置 | `registry.py` | 原本只按 name 缓存，第二次传入的新配置被静默忽略（必须手动 `clear_instances()`）。现在按 `(name, 配置指纹)` 缓存，配置变化自动重建并先 `close()` 旧实例 |
| **搜索网格量纲错了** | `eval_locate.py` | `morph_kernel_ratio` 是"相对图宽的比例"，网格却给到 `(0.025, 0.60)` —— 0.60 = 图宽 60% 的形态学核，会把整幅图糊成一团；且网格整体偏大、**没包住默认值** `(0.005, 0.021)`，选出来的"最优"是网格边缘的退化解。已修正范围并加 `TestSearchGridSanity` 守住 |
| 参数量纲注释写错 | `config.py` | `open_kernel_ratio` / `morph_kernel_ratio` 注释写"万分比/千分比"，实际就是比例（`0.02` = 图宽 2%） |
| 未使用导入 | 多处 | `cli.py`、`window.py`、`security.py`、`cv_engine.py` 等 13 处；因为这次要给 CI 开 ruff，必须清掉 |
| `Track.to_result()` 伪造逐字符置信度 | `tracker.py` | 用 `[best_conf] * len(text)` 填 `char_confidences`，与 deep 引擎真实 per-char 概率语义冲突。**本次未改动**（会牵动跨帧投票的加权），列为待办 |

## 六、批量识别多进程（`--workers`）

新增 `lpr/batch.py` + `Recognizer.recognize_many(workers=N)` + CLI `--workers`：

* 分片数 = `workers * 4`（多于进程数，便于负载均衡）
* 每个子进程在 `initializer` 里只建一次识别器（模型冷启动是固定成本）
* 结果**按原始顺序**拼回，与串行路径一致
* 单文件异常被隔离成一条 warning，不拖垮整批
* `workers > 1` 时 `roi` 恒为 `None`（跨进程传图像的开销远大于收益），已写进 docstring
* 文件数不足以摊平固定成本时**自动退回串行**（`MIN_FILES_PER_WORKER`），
  避免"开了并行反而更慢"
* `spawn` 而非 `fork`：避免 Linux 上 fork 出持有 `INFERENCE_LOCK` 的副本导致死锁

### 如实说明：收益没有被验证

在开发/CI 环境实测**没有加速**：

| 场景 | 串行 | 4 进程 | 加速比 |
|---|---|---|---|
| cv 引擎 120 张 | 2.83s | 2.34s | 1.21x |
| deep 引擎 40 张 | 29.87s | 43.58s | **0.69x** |
| deep 引擎 160 张 | 172.68s | 196.29s | **0.88x** |

为排除"是实现的锅"，做了一组**纯 CPU 计算对照**：

```
cpu_count=20   4x 工作量：串行 0.50s / 4 进程 1.02s → 0.49x
```

环境报 20 核，但纯 CPU 计算用 4 进程也只有 0.49x —— **是 CPU 配额受限，
不是并行实现有问题**。这个结论写进了 README 的「已知限制」，并提供了
`tools/bench_workers.py --cpu-control` 让使用者在自己的机器上量。

之所以把这套代码留下来：功能本身完整、有测试（顺序 / 隔离 / ROI 取舍 /
小批量退回串行都有断言），在真实多核机器上应当有收益；但**我们没有证到**，
所以不写"提速 N 倍"这种话。

## 七、工程化配套

* **CI**（`.github/workflows/ci.yml`）：ruff + mypy；Ubuntu 上 Python 3.10~3.13、
  Windows 3.12（中文路径逻辑强相关）；Qt 用例走 `QT_QPA_PLATFORM=offscreen`
* **pyproject**：ruff（E/W/F/I/B/UP/C4/SIM/RUF）、mypy、`py.typed`、`Typing :: Typed`
  classifier、`deep-paddle` / `gui` / `dev` extras 整理
* **回归基线**（`tests/test_regression_baseline.py` + `tests/baselines/real_samples.json`）：
  用确定性 `cv` 引擎跑仓库自带 11 张真实样张，对照候选数 / top 候选框（IoU ≥ 0.85）/
  top 底色，任一漂移即失败
* `tools/eval_locate.py` 新增 `--save-baseline` / `--baseline` / `--tolerance`，
  可把定位指标做成卡口（fitness 跌幅超阈值即非零退出；样本数不一致会明确拒绝比较）
* `tools/make_regression_baseline.py`：生成/更新行为快照

### 关于"合成图当准确率基线"——试过，放弃了

为了在 CI 里有准确率门槛，写过一版程序化生成的合成样张（精确真值、可复现、
不依赖下载）。实测不可行：

```
ratio=0.10  真值=[214,195,314,226]  候选=0
ratio=0.20  真值=[558, 11,758, 74]  候选=0
ratio=0.35  真值=[549,336,899,447]  候选=0
ratio=0.50  真值=[179,446,679,605]  候选=2  ← 找到的是字符子区域，不是车牌
```

合成牌只有放大到占画面一半宽才勉强被定位到，且找到的是字符区而非车牌区。
拿它算准确率是自欺欺人，所以那套生成器**已删除**，改为"行为快照"基线，
并在文件里明确写出它**不能**回答"准确率是多少"。

## 八、测试

* 新增 `tests/test_v2_3_1_fixes.py`（37 项）、`tests/test_batch_parallel.py`（30 项）、
  `tests/test_regression_baseline.py`（22 项）、`TestSearchGridSanity`（5 项）
* 测试总数 366 → **472**，全部通过
* 关键回归断言举例：`Recognizer` 必须回落 `config.security`、
  `CameraWorker` 源码里不得出现 `predict(`、`check_model` 的"跳过"不得显示成 ✓、
  多进程结果顺序与串行一致、网格必须包住默认值

## 九、阈值调参复核：结论是"不落库"

上一版留下的待办"把调参结果 apply 到 `config.json`"，复核结论是
**维持默认参数不改** —— 因为上轮那份"最优值"本身就是网格量纲错误产出的
退化解（见第五节），而重新评测没有找到任何站得住的改进。

### 依据：弱监督评测（`tools/eval_weak.py`）

手上大量真实 CCPD 图没有真值框（标注在重命名时被抹掉），带标注集又下载
不全，于是用弱监督信号：真实车牌号码必须满足国标语法，所以
"**能否产出语法合法的号码**"是不需要人工标注的有效指标。基线（deep 引擎，
60 张真实样张）：

```
合法率 0.5167   候选率 0.5667   过检率 0.0000
```

命中的候选里 31/34 语法合法（质量很高），瓶颈是**召回**（四成图两条
定位/兜底路径都颗粒无收）。

### 三组候选参数 × 留出集验证

| 候选 | 调参段（前 60 张） | 留出集（第 61~160 张，100 张） | 结论 |
|---|---|---|---|
| A `min_area_ratio 0.0008` + `color_ratio 0.35` | 完全持平 | —（未进入验证） | 淘汰 |
| B `min_aspect 1.6` + `max_aspect 7.0` | 合法率 +1.7pp | **完全持平**（0.4900 → 0.4900） | 淘汰 |
| C `max_candidates 8` + `min_chars 5` | 完全持平 | —（未进入验证） | 淘汰 |

候选 B 在调参段看着有 +2 张的改善，**在留出集上没有复现** —— 是切片噪声。
上一轮 `morph_kernel_ratio=(0.025, 0.60)` 的教训就是"没有验证集的调参
不可信"，这次同样的问题在 60 张图的小样本上差点重演，被留出集拦下了。

### 顺带修掉的工具 bug

`eval_weak.py` 的 `--skip/--limit` 组合原本是"先取前 limit 张、再跳过
skip"，`--skip 60 --limit 100` 实际只评 40 张且与调参段重叠。已把切片
抽成 `select()` 并钉死回归测试（第 61~160 张必须恰好 100 张）。

### 留下的资产

* `tools/eval_weak.py`：无标注真实图的弱监督评测，`--override` 一行 JSON
  即可做参数对照，`--skip/--limit` 支持调参段 / 验证段分离
* `tests/test_eval_weak.py`（18 项）：指标口径 / 异常隔离 / 覆盖参数 /
  切片口径全部有测试
* 以后真要调参：先在调参段看到明确改善，再过留出集，最后过
  `tests/test_regression_baseline.py` 的行为快照 —— 三关都过才动 `config.json`

## 十、特种号牌（军/警/白牌）专项提升

> 起因：`img/police.jpg`（警车 鄂F8888警）实测识别为 `鄂F08888 [黑牌]`——
> 号码错一位、后缀丢失、白底判黑，三处全错。特种牌是全链路最弱的一环。

### 10.1 实证定位的问题

| # | 问题 | 证据 |
|---|---|---|
| 1 | 严格抽取正则末位可选集缺「使/领」，使领馆牌被截成非法残串 | `京123领` → 抽出 `京123`（6 位残串） |
| 2 | 严格层没有武警分支（`WJ…`）与「使+数字」分支 | `WJ京1234X`、`使123456` 严格层抽不到 |
| 3 | 武警格式强制要求末位字母，且没有旧式 | `WJ京1234` 判非法 |
| 4 | `correct()` 对军/使/WJ 字头**整体早退** | `军A1O345` 的 O 从未被纠正 |
| 5 | 特殊后缀（警/挂/学/危/使/领/港/澳）与字头（军/空/海…）**没有任何形近字纠错** | `敬→警`、`顿→领` 直接判无效 |
| 6 | `correct()` 会把**已合法**的号码纠坏 | `京123使`（合法）→ `京J23使`（非法） |
| 7 | 阴影下白底警牌被判黑牌 | `img/police.jpg` 真牌区 V 中位仅 ~120，进不了 V≥200 的白区间 |
| 8 | cv 引擎混淆修正破坏特种字头 | `使123456` 第 2 位数字被强改字母 |

### 10.2 判色层的取舍记录

白底警牌（阴影）曾尝试两种方案，均**被实测否决**后才有现方案：

* **放宽白色 HSV 区间**（V≥200→150）：阴影警牌命中率仍只有 0.7%，无济于事；
* **边缘环带极性判定**（用边框亮度仲裁白/黑）：实测反转——阴影白牌边缘中位
  灰度 59，比真黑牌使领馆（88）还暗。废弃。

最终方案：**红色字符信号**。军牌字头（军/空/海…）、警牌「警」字、武警「WJ」
按规范均为红色，这是特种白牌唯一的法定物理特征：

* 红色占比 ∈ [0.03, 0.15] 且最大红色连通域高度 ≥ 45% ROI 高（字符级红块，
  实测警牌「警」高占 0.81，尾灯/招牌反射 <0.45）→ 判白，置信度抬到 0.5
  （与危险品极性判定同款先例，避免被大面积黑底占比压过）；
* 触发时机在彩色判定**之后**：`hongkang1.jpg` 的黄牌候选含红色招牌反射，
  黄底先胜出，顺序本身就是防线；
* 阈值仅由仓库内 3 张样张标定（样本量小，如实记录），后续有更多特种牌
  样张应再校准。

效果（`img/police.jpg` 端到端）：`鄂F08888 [黑牌]` → **`鄂F8888 [白牌]`**，
其余 7 张样张判色零变化，行为基线无需重生成。

### 10.3 语法层修复清单

* `_PLATE_TEXT`：末位可选集补「使/领」；新增 `使+4~6位数字`、
  `省+3位数字+使/领`、`WJ+省+4位数字+可选尾位`、`WJ+5~7位数字`（旧式）分支；
* `FORMATS`：武警末位字母改可选（`WJ京1234` 合法）；新增旧式武警
  `WJ+6~7位数字`（WJ01-12345 清洗后形态）；
* `correct()`：
  * 开头新增**合法即返回**短路（修复问题 6，同时对已合法号码零开销）；
  * 新增规则 0b：字头形近字（`HEAD_LOOKALIKES`：罕→军、每→海、便→使…），
    替换后整体合法才采纳、采纳即返回；
  * 新增规则 0c：末位后缀形近字（`SUFFIX_LOOKALIKES`：敬→警、桂→挂、
    字→学、免→危、顿→领…），闸门同上；「字/宇」在首位归军、末位归学，
    按位置取表；
  * 军/使/WJ/民航字头的早退分支改为**仍执行序号位 I/O 纠正**（修复问题 4）；
* cv 引擎：`_fix_confusions` 的"第 2 位数字→字母"仅对省份牌与军队牌执行；
  `_score_text` 改用统一的 `identify_format`，特种牌不再被系统性压低。

### 10.4 如实记录的边界

* **末位红字的 OCR 漏检未解决**：`鄂F8888警` 仍可能读成 `鄂F8888`（小字号
  红色字符是通用 OCR 模型的弱区）。语法层可以在"结构+白牌+纯数字结尾"时
  提示疑似缺字，但**不能反向补字**（普通蓝牌同形），未做臆改；
* 使领馆牌第 2 位按现行口径为数字，`京A123使` 类号码仍判非法（无国标依据
  放宽，避免误吞民用牌）；
* cv 引擎（SVM）的字符集**结构上不含** 军/空/海/警 等特种字符（省份分类器
  只认 31 省份、字母数字分类器无汉字），特种牌语法层修复对 cv 引擎只在
  "不破坏、不误罚"层面生效；要真正识别需要重训模型，样本不可得。

### 10.5 测试

* 新增 `tests/test_special_plates.py`（49 项）：严格校验/抽取/纠错/打分/
  红色信号（合成 ROI 四分支）/cv 混淆修正，全部通过；
* 测试总数 478 → **527**；行为基线（11 张真实样张）零变化。
