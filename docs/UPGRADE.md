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

## 十一、启动入口与系统兼容性（v2.3.3）

> 起因：用户问「怎么启动 / 兼容什么系统 / Win7 8 10 11 XP 支持吗」。
> 三个问题都答不上来，说明**文档缺了一节**；顺着查还揪出三处与"启动"直接相关的实际缺口。

### 11.1 查出并修复的问题

| # | 问题 | 影响 |
|---|---|---|
| 1 | 没有 `src/lpr/__main__.py` | `python -m lpr` 报 "No module named lpr.__main__"，只能写 `python -m lpr.cli`；不装 console script 时没有干净入口 |
| 2 | 没有 Windows 启动脚本 | 用户得自己解决"用哪个 python""依赖装没装""PYTHONPATH 怎么设" |
| 3 | `CameraWorker.run()` 无条件传 `cv2.CAP_DSHOW` | DirectShow 是 **Windows 专有**后端，Linux/macOS 上必然打不开摄像头——而项目其余部分与 CI 都是跨平台的，等于把摄像头悄悄钉死在 Windows |
| 4 | README 缺「环境与系统要求」 | 系统兼容性只能靠猜 |

### 11.2 改动

* 新增 `src/lpr/__main__.py`：无参数 → 注入 `gui`（"clone 下来就想看效果"的最短路径）；
  带参数 → 原样透传给 CLI 子命令。于是 `python -m lpr` 开界面、
  `python -m lpr predict a.jpg --engine deep` 走命令行。
* 新增 `start.bat`（Windows 一键启动）：按 `LPR_PYTHON` → `.venv` → `py -3` → `python`
  找解释器，校验 ≥ 3.10，缺 PySide6 就装，设好 `PYTHONPATH` 后启动；带参数透传给 CLI。
* `ui/worker.py` 新增 `camera_backend()`：Windows 保留 DirectShow（对部分 USB 摄像头
  比 MSMF 稳），其它平台交回 `CAP_ANY`（Linux → V4L2，macOS → AVFoundation）。
* README 新增「环境与系统要求」：最低要求表、**系统兼容矩阵**、
  「为什么 Win7 / 8 / XP 不行」、四种启动方式、引擎↔依赖对应表。

### 11.3 关于 Windows 7 / 8 / XP（如实记录）

不支持的结论建立在两道**互相独立**的硬门槛上：

1. **CPython 侧**：官方只在 Microsoft 仍处延长支持期的 Windows 上支持。
   最后一个能装到 Win7 的是 **Python 3.8**（官方原话："If you require Windows 7
   support, please install Python 3.8"）；XP 止步 **Python 3.4**。
   而本项目 `requires-python >= 3.10`。
2. **Qt 侧**：PySide6（Qt 6）官方支持 **64 位 Windows 10 1809+**，明确放弃
   Win7 / 8 / 8.1 与**全部 32 位 Windows**。

Win8.1 的特殊性：Python 3.13 官方文档仍声明支持 8.1，但界面依赖的 Qt 6 已放弃
→ **命令行理论可行（未验证）、图形界面一定不行**。

想支持 Win7 属于独立移植（Python 3.8 + PySide2/Qt5 + 旧版 numpy/opencv +
放宽 `requires-python`），本仓库未做也未承诺。

### 11.4 踩到的坑

**批处理文件不能写非 ASCII。** 第一版 `start.bat` 用中文写注释，`cmd` 按系统
ANSI 代码页（GBK）解析 UTF-8 字节，中文被拆碎后**当成命令执行**，实测报出
`'器（优先级最高），例如：' 不是内部或外部命令`。改为纯 ASCII + CRLF 后正常，
并加护栏测试锁死。

### 11.5 测试

* 新增 `tests/test_launch_env.py`（11 项）：
  * `python -m lpr` 入口存在、无参数注入 `gui`、参数透传（**真起子进程**验证）、
    非法参数仍按 CLI 语义报错；
  * `camera_backend()` 三平台分支（win32 / linux / darwin）；
  * `worker.py` 源码不得出现无条件的 `VideoCapture(..., cv2.CAP_DSHOW)`；
  * `start.bat` 必须存在、**纯 ASCII**、含 `PYTHONPATH=%~dp0src`、含版本下限校验、行尾 CRLF；
  * README / UPGRADE 必须写明 Python ≥ 3.10。
* 测试总数 527 → **538**，全部通过。

实测（Windows 11 + Python 3.13.14）：`python -m lpr engines`、`start.bat engines`
（含中文输出的子进程路径）均正常。

## 十二、角点排序旋转鲁棒性（P0）与号牌规范口径核对（P1/P2）

> 落实 `docs/REFERENCE_REVIEW.md` 第 6 节的 P0/P1/P2 三条建议。

### 12.1 P0：`_order_quad` 在侧倾 >45° 时把车牌"立"起来

**缺陷**（几何实证 + 端到端实证）：

旧实现用「x+y 最小/最大 + y-x 最小/最大」两两配对，该判据只在侧倾 <45° 时
恰好把"上边"配到长边上。对 220×70 的矩形逐角度实测：

| 侧倾 | 44° | **46°** | 50° | 60° | 75° | 89° | −50° |
|---|---|---|---|---|---|---|---|
| 旧实现上边 | 220 | **70** | 70 | 70 | 70 | 70 | 70 |

46° 起上边被配到**短边**上，透视矫正产物变成 90° 竖图，字符分割必然失效。
而当时的 docstring 写着"用叉积判断"——代码里根本没有叉积，文档与实现不符
让这个缺陷一直没被怀疑。

**修复**（`locate._order_quad`，约 40 行）：

1. **极角环形排序**：按绕质心的极角排序，得到与旋转角无关的稳定环形顺序
   （图像坐标系里是顺时针，因此映射到目标矩形不会引入镜像）；
2. **挑起点**：4 个环上起点中只有两个能让"上边"是长边（另一个正是它的
   180° 反转）。再要求上边**指向右侧**（`tr.x > tl.x`）——车牌从左往右
   阅读，只要侧倾不到 90°，真正的左上角必然满足这一条。

**实测结果**：±89° 逐度扫描（共 179 个角度，输入顺序随机打乱），角点归位
误差全部为 0.000；矫正图恒为横向。合成蓝底车牌端到端 0°→89° 全部读出
`WA12345`。

**如实修正一个旧结论**：早先评估时说 `single_blue.jpg` 旋转 50° "定位已
命中车牌、只是被 warp 成竖图"——这次把那个 2.53 比例的 ROI 导出来看过，
**是一根树枝**。真实情况是定位层根本没找到侧倾 50° 的车牌；几何缺陷是真实
的（纯几何实证成立），但"修好它就能救回旋转样张"这个推断当时并不成立。
侧倾车牌的**定位召回**是下一个真短板，已写进 README 已知限制。

### 12.2 P1：武警旧式 + 「危险品」多字后缀

* **武警旧式**（`WJ + 2 位机构码 + 4~5 位序号`）在 v2.3.2 第十节已加入
  `FORMATS`（`WJ[{D}]{6,7}`），并有 `WJ0112345` / `WJ012345` 回归用例——本次核对确认无需再改；
* **危险品后缀**：允许 OCR 把末位「危」多认成「危险/危险品」
  （`危(?:险品?)?`）。GA 36-2018 的分类表其实**没有**单列危险品运输号牌
  （危险货物车辆另按 GB 13392 挂标志牌/标志灯），"末位标危"属行业通行口径，
  已在注释中如实记录。放宽只会提高召回——这类串必须前缀已成立才可能匹配。

### 12.3 P2：三处口径核对（均有权威出处）

| # | 疑点 | 核对结论 | 依据 | 处理 |
|---|---|---|---|---|
| 1 | 使领馆排列 | **参考实现是对的**：现行式"使"在号码**末尾**（六位代码+使；领馆为省简称+国家代码+领）。`使[0-9]{6}` 是 2017-05-01 废止的旧式 | 《外交车辆管理办法》(2017-01-09)、GA 36-2018 式样修订、维基百科"中华人民共和国机动车号牌" | 新增 `[字母数字]{6}使`（使馆代码可含字母，如 X59/3XX 为国际组织）；旧式保留一条兜住老影像 |
| 2 | 警用第 3 位 | **参考实现是对的**：第三位为 A/B/C/D 或阿拉伯数字（法院/检察院/国安/司法行政/公安），我们只认数字会把 `京AA006警`（法院）判非法 | 《警车管理规定》附件 2《警车号牌式样》（公安部令，mps.gov.cn 原文） | 第三位放宽为 `[0-9A-D]`，并钉死"E 之后不收" |
| 3 | 军队第二位 | **不采纳参考实现的受限字母表**。其 `[A-DJ-PR-TVY]` 对应大军区时代二级机构代号表（A 司令部…Y 野战部队），出处是问答类网站，非权威标准；军车号牌**不在 GA 36 管辖范围**，军队不公开完整字头表，收紧只会制造漏检 | 同上 + 解放军报/北京交管局对 2012 式的报道 | 保持全字母集 |

**P2 核对中的额外发现**（比疑点本身更重要）：我们原有的军队格式
`汉字字头 + 字母 + 5 位数字`（军A12345）其实是 **2004 式**，2013-05-01 已
废止；现行 **2012 式** 是「双字母 + 5 位数字」（如 LB12345，字头取大单位
拼音首字母），整串不含汉字。已新增该格式，旧式保留。

*如实的代价说明*：`双字母+5 位数字` 没有汉字锚点，`GB12345` 这类串也会
被判成军牌。60 张真实样张弱监督评测显示**过检率仍为 0.0**、候选率/合法率
与改动前完全一致（0.5667 / 0.5167），故采纳；样本量小，后续有军牌样张应再校准。

### 12.4 实现中踩到并当场修掉的坑

* 想匹配"危/危险/危险品"写成了 `危险品?`——实际上 `?` 只作用于 `品`，
  该正则要求"险"必须出现，导致 `京A1234危`（最常见的形态）反而判非法。
  正确写法是 `危(?:险品?)?`；
* 军牌 2012 式分支（纯字母数字）放在 `_PLATE_TEXT` 交替组**前面**，会把
  `WJ012345` 抢着切成 `WJ01234`（`{5}` 先匹配满就停）——已移到最后，
  并加用例钉死"不得吞武警"。

### 12.5 测试

* 新增 `tests/test_quad_ordering.py`（179 项）：±89° 逐度角点归位、
  输入顺序无关性、矫正图恒横向、车牌比例保持、退化输入、斜拍梯形；
* `tests/test_special_plates.py` 新增 4 个口径类（30 项）：使领馆现行式/旧式、
  警用第三位 A–D、军队 2012 式/2004 式、危险品后缀变体，以及
  "纯字母数字分支不得吞武警"的顺序守卫；
* 测试总数 538 → **776**，全部通过；行为基线（11 张真实样张）零变化。

---

# v2.3.5 —— 修 `start.bat` 的依赖误判与环境污染

起因是一份真实的首次启动日志：在一台**只装了官方 CPython 3.13**、
从未装过本项目依赖的机器上双击 `start.bat`，脚本宣布
`[2/3] PySide6 missing, installing...`，开始下载 82 MB 的
`PySide6` wheel。三个问题叠在一起：

## 一、依赖探测不完整：只查了 PySide6（会必然崩溃）

旧实现只跑 `import PySide6` 作为唯一判据。但干净解释器里
`numpy` / `cv2` / `PIL` **本来就没有**。实测：

```text
$ python -c "import importlib.util as u; print(u.find_spec('numpy'))"
None
$ PYTHONPATH=src python -m lpr --help
  File ".../src/lpr/api.py", line 28, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

于是老脚本的结局是：**先花几分钟装完 220 MB，再在最后一步
`ModuleNotFoundError: numpy` 崩掉**。检查项与真实依赖集不一致，
是所有"一键脚本"最常见的失效模式。

修法：探测表达式改为 `import numpy, cv2, PIL, PySide6`，四项缺一不可；
并且**安装完成后复检一次**——"装成功但装进别的解释器 / 包名写错"
这类问题不允许静默通过。

## 二、往用户的全局解释器里装包

`.venv` 不存在时，旧脚本回落到 `py -3` / `python`，即用户的**全局**
Python，然后 `pip install` 直接污染它（本次实测就是
`C:\Users\hongliwang\AppData\Local\Programs\Python\Python313`）。
项目自己的 README 并未承诺要改用户的系统环境，这属于越界。

修法：缺依赖时**创建项目内 `.venv`** 并把之后所有命令切到它的解释器
（创建失败才回落到原解释器，且明确告警）。已有 `.venv` 则直接复用。
另外保留 `LPR_NO_VENV=1` 作为逃生门，给确实想装进现有环境的人。

顺带一个"不折腾"的设计：**先探测、后建 venv**。如果找到的解释器本来
就齐全，就直接用它，不为建而建、不重复下载。

## 三、GUI 依赖装的是元包，白下 130 MB

界面只 `import` `QtCore` / `QtGui` / `QtWidgets` 三个模块，全在
**PySide6-Essentials** 里；而元包 `PySide6` 会连带拉
`PySide6-Addons`（WebEngine / Charts / Multimedia / 3D…，约 130 MB），
本项目一个都不用。

修法：`pyproject.toml` 的 `gui` extra 由 `PySide6>=6.5` 改为
`PySide6-Essentials>=6.5`，`start.bat` 的兜底清单同步。
（已用合成冒烟测试确认 Essentials 足以 import 整个 `lpr.ui.window`。）

## 四、安装目标改为 `.[deep,gui]`

旧脚本装完 PySide6 就启动，结果是"界面能用、但推荐引擎不可用"。
`deep`（`rapidocr-onnxruntime` + `onnxruntime`）实测安装后仅占 62 MB，
是 README 明确标注的**推荐引擎**，一键启动理应交付一个完整可用的程序。
故首次安装统一为 `pip install -e ".[deep,gui]"`，与 README「四种启动方式 ②」
保持一致。

## 五、顺带补上：pip 不得继承外部 `PYTHONPATH`

验证过程中真实撞到一次：在带环境注入的 shell 里跑安装，pip 解包时的
`os.unlink` 被**外部 `sitecustomize.py`**（经 `PYTHONPATH` 注入的策略模块）
劫持并抛出 `SystemExit: 1`，报错文本与依赖毫无关系：

```text
pip/_internal/operations/install/wheel.py, in save
    os.unlink(self.dest_path)
.../sitecustomize.py, in _safe_remove
    _try_trash(abs_path)
SystemExit: 1
```

这跟本项目无关，但它说明一件该防的事：**安装器进程不该被外部
`PYTHONPATH` 影响**——路径上任何目录里的 `sitecustomize.py` 都会在 pip
自己的进程里被 import 并执行。因此 `start.bat` 现在会先存下用户的
`PYTHONPATH`、清空后才去装依赖，只在最后启动时还原；还原用 `if defined`
判断，避免空值拼出尾随分号（`sys.path` 里的空条目等价于把当前工作目录
加进去）。

## 六、验证方式（不靠"看起来对"）

批处理是典型的"改完不跑必出事"，因此分两级验证：

1. **干跑**：把两行 `pip install` 替换成 `echo`，用真实 `cmd` 执行，
   确认控制流、标签跳转、**`%PY%` 在 `goto` 之后确实切到了新解释器**
   （日志里 `[3/4] installing into .venv\Scripts\python.exe` 就是证据）、
   以及装后复检的报错分支；
2. **从零实跑**：把 `start.bat` + `pyproject.toml` + `README.md` + `src/`
   复制到一个临时目录（等价于全新检出），在**不带任何依赖、且继承着外部
   `PYTHONPATH`（shim 在位，最坏情况）** 的环境里完整跑一遍：建 `.venv` →
   装 `.[deep,gui]` → 复检 → 启动 `engines`。实测 35 秒装完
   （`Successfully installed ... PySide6-Essentials-6.12.0 ... lpr-v2-2.3.5`），
   两个引擎均报"可用"。这一跑同时证明了 PYTHONPATH 隔离确实挡住了注入；
3. **GUI 冒烟**：用只装了 Essentials 的 `.venv` 在 offscreen 下真实构造主窗口，
   确认窗口标题为 `车牌识别系统 v2.3.5` —— 这是"Essentials 够用、
   不需要 Addons"的直接证据（不是靠推断）。

同时新增 9 项静态回归（见下），把这些策略钉在测试里，防止回退。

## 七、测试

`tests/test_launch_env.py` 新增 `TestLauncherDependencyPolicy`（9 项）：

* 探测表达式必须覆盖 `numpy` / `cv2` / `PIL`（防退回"只查 PySide6"）；
* 探测表达式必须是合法且可执行的 Python，在依赖齐全的环境里静默通过、
  在缺任一模块时非零退出（正反双向对照）；
* 缺依赖必须建项目内 `.venv`，且存在 `LPR_NO_VENV` 逃生门；
* 安装后必须复检；
* `pyproject.toml` 的 extras 与 `start.bat` 的兜底包名必须一致
  （防止两处各说各话）；
* 安装步骤必须先保存并清空 `PYTHONPATH`、启动时用 `if defined` 还原，
  且不得再出现 `PYTHONPATH=%~dp0src;%PYTHONPATH%` 这种空值留分号的写法。

测试总数 776 → **785**，全部通过；行为基线（11 张真实样张）零变化。

---

# v2.3.6 —— 启动器改为「纯 ASCII 引导壳 + Python 打印中文」

诉求很朴素：把启动脚本的英文提示改成中文。第一版直接在 `start.bat` 里写中文，
**实测当场炸了**，于是变成了一次结构性重构。

## 一、实测结论：批处理文件里不能出现汉字（不是风格问题）

第一版写法是 `chcp 65001` + 文件存成 UTF-8（无 BOM），小探针测试通过。
换成真实脚本后，第 20 行左右出现：

```text
       完成 —— CV 引擎、深度引擎、图形界面均已就绪。
'，未改动你系统里的' 不是内部或外部命令，也不是可运行的程序。
```

一行提示被从中间截断，**残片被当成命令执行**。

根因：cmd.exe 计算批处理「下一行」的偏移时，拿上一行的**字符数**当**字节偏移**
用。文件里只要有一个多字节字符，下一行就会被超前 ``(字节数 - 字符数)`` 个字节
开始读。核对得上：上一行 `echo ...完成 —— CV 引擎、深度引擎、图形界面均已就绪。`
含 20 个多字节字符，正好吞掉下一行开头约 40 字节。

**为什么小探针会骗过我们**：短文件里超前量往往刚好落在下一行的缩进空格里，
输出看上去完全正常。我们那个 6 行探针就是这么"通过"的 —— 这是本次最值得记的
一条经验：**验证要跑真实规模，探针通过 ≠ 成立**。

顺带确认：把中文放在 `if (...)` 块里、以及 `goto` 之后，同样会错位；与
`(` `)` 转义无关，与"特殊字节"无关，纯粹是偏移数学。

## 二、新结构：两层

| 文件 | 职责 | 编码 |
|---|---|---|
| `start.bat` | 纯 ASCII 引导壳：找解释器 → 转交 → 无解释器兜底 | **ASCII（硬约束）** |
| `tools/launcher.py` | 版本校验、依赖探测、建 `.venv`、pip 安装、启动；**全部中文提示** | UTF-8 |
| `tools/no-python.txt` | 找不到任何 Python 时的中文提示，由 `type` 原样输出 | UTF-8 |

中文改由 Python 打印：Python 在 Windows 经 `WriteConsoleW` 输出，与控制台代码页
（65001 / 936）无关 —— 用户日志里 `lpr engines` 的中文一直显示正常，就是这条路径。

`:no_python` 分支没法用 Python（鸡生蛋问题），改用 `type` 输出 UTF-8 文本文件：
`type` 是字节拷贝，批处理解析器全程不接触多字节字符，命令本身仍是 ASCII；
`chcp 65001` 只服务于这一条路径的渲染。

## 三、顺带修掉的三件小事

1. **重定向时输出顺序错乱**：`start.bat > out.txt` 时 Python 的 stdout 变块缓冲，
   本模块的提示整段压在子进程输出之后。改成行缓冲，控制台与管道下顺序一致。
   （这条是写端到端测试时被自己的断言抓出来的。）
2. **`%~dp0` 在切代码页之前捕获**：路径里含中文时（`D:/我的项目/lpr-v2`）
   更稳。
3. **解释器可用性校验**：失效的 `LPR_PYTHON`、坏掉的 `.venv` 会先被清理掉，
   再回落到 `py -3` / `python`，而不是直接报一句 cmd 的
   `'xxx' 不是内部或外部命令`。

## 四、批处理的逻辑搬进 Python 之后的额外收益

`choose_target()` 现在是一个纯函数，可以拿假 probe 直接测四种分支
（复用 / 用现有 venv / 建 venv / `LPR_NO_VENV` 逃生门）——
这在批处理里是做不到的。安装环境与启动环境的构造也抽成了
`install_env()` / `build_env()` 两个纯函数，于是"pip 不继承外部 PYTHONPATH"
和"`src` 注入不留尾随分隔符"这两条约束第一次有了**行为级**测试，
而不再只是对脚本做字符串匹配。

## 五、验证（三条路径都真跑）

1. **正常启动**：`start.bat engines` → `[1/4] 解释器正常` … `[4/4] 正在启动...`
   + 引擎表，中文完好、顺序正确；
2. **从零全路径**：脚本复制到临时目录（等价全新检出）→ 建 `.venv` →
   装 `.[deep,gui]` → 复检 → 启动，全程中文；
3. **无 Python 兜底**：把 PATH 收窄到只剩 `System32`（`py` / `python` 都不可见、
   无 `.venv`）→ `type` 输出的中文正常，没有任何残片被当命令执行。

另外反向验证：把同一份输出按 GBK 解码得到乱码，证明它确实是 UTF-8 字节
（即 `chcp` 生效、不是碰巧）。

## 六、测试

`tests/test_launch_env.py` 重写为 5 个测试类（32 项）：

* `TestWindowsLauncher`：`start.bat` 必须是纯 ASCII、无 BOM、CRLF；
  逻辑不许长回批处理（`set "PROBE=` / `-m venv` / `-m pip` / `sys.version_info`
  都不许出现）；必须转交 `tools/launcher.py`；
* `TestPythonLauncher`：**用 AST 断言只导入标准库**（启动器必须能在
  "什么都没装"的解释器上跑）；版本号与 pyproject / cli.py 互相钉住；
  `choose_target` 四种分支；`probe` 对坏路径返回 False 而不抛异常；
  `install_env` 剥掉 PYTHONPATH；`build_env` 不留尾随分隔符；
* `TestLauncherMessages`：提示文件是 UTF-8 中文且无 BOM；
  `type` 原样输出该文件（Windows 专有）；
  **真跑一次 `start.bat engines` 并断言中文没被当命令执行**（这是唯一能抓住
  本版本那个 bug 的测试）。

测试总数 785 → 797（v2.3.6）；v2.3.7 后见下一节，全部通过；行为基线（11 张真实样张）零变化。

## 七、附带发现（本次未改）：CI 的 lint 任务其实是红的

在验证过程中装了 ruff / mypy 真跑了一遍，发现两个 **既有**、与本次改动无关的
问题 —— 也就是说 `main` 上的 `lint` job 一直没过：

* `ruff check src tools tests` → **142 个错误**、`ruff format --check` →
  **46 个文件需要重排**（例如 `B905` 要求 `zip(..., strict=)`、
  `SIM105` 要求 `contextlib.suppress`，这些规则在本项目动工之后才加入）；
* `mypy` → **33 errors in 10 files**（如 `cli.py:143` 需要类型标注、
  `api.py:540` 的 `VideoWriter_fourcc`）。

根因是 `dev` extra 里写的是 `ruff>=0.6`（不封顶），CI 每次装到的都是**当时最新**的
ruff/mypy，而代码是按 0.6 时代写的。两种修法各有取舍（把工具版本钉死，
或把代码升到最新规则），需要单独决策，故本版本只保证**新增/改动的文件是干净的**：

```text
$ ruff check tools/launcher.py tests/test_launch_env.py
All checks passed!
$ ruff format --check tools/launcher.py tests/test_launch_env.py
2 files already formatted
```

---

# v2.3.7 —— 摄像头打不开时，说人话

用户在一台**没有插任何摄像头**的机器上点「启动摄像头」，弹窗只有一句
「无法打开摄像头 0」——分不清是没插线、被别的程序占用、还是索引选错；
弹窗点掉之后画面永远停在「等待识别」。

## 一、先诊断：是代码问题还是环境问题

在用户机器上实测（OpenCV 三种后端逐个试）：

```text
DSHOW: opened=False   MSMF: opened=False   ANY: opened=False
[ERROR] obsensor: Camera index out of range
```

再从 Windows 侧枚举设备：

```text
USB Camera          Present=False
Chicony USB2.0 Camera  Present=False
WXSJ 4K CAMERA      Present=False
```

三个摄像头条目全是 `Present=False` 的**幽灵设备**（历史连接残留，当前不在线）。
结论：GUI 代码行为正确，是机器上真的没有可用摄像头。旧弹窗的问题不是"报错"，
而是**报得不可行动**——用户拿到截图来找我们，本身就是这个 UX 失败的证据。

## 二、改进

1. **打不开时先扫描**：`CameraWorker.run()` 打不开设备时，先调用新加的
   `list_cameras()` 逐个探测 0~7 号索引，再按结果分两种话：
   - 一个都没有 → 完整排查清单（插线/设备管理器/独占程序/隐私开关）；
   - 有但不是当前索引 → 直接列出可用索引，指向工具栏下拉框。
   探测在摄像头线程里做（失败很快，实测全扫 <1s），不堵 UI。
2. **工具栏新增「摄像头」下拉框**：窗口启动时在后台线程探测一次并填充
   （跨线程用 Qt 信号队列投递）。多摄像头机器上索引 0 未必是想要的那个；
   什么都没探到也保留一个禁用的「未检测到摄像头」项，按钮可点、提示可触发。
3. **`list_cameras()` 判定从紧**：`isOpened()` 之外还要 `grab()` 一帧才算数
   （部分虚拟设备能通过 isOpened 却出不了帧）；单个设备抛异常跳过不中断；
   每个探过的设备都立即 `release()`。

## 三、验证

- 单元：`tests/test_camera_probe.py` 8 项——假开设备对象测探测五种情形
  （正常/幽灵设备/句柄释放/驱动异常不中断/全空）与提示文案三分支；
- **真机冒烟**：就在这台没有摄像头的机器上完整走一遍——下拉框显示
  「未检测到摄像头」并禁用，点「启动摄像头」弹出排查清单，按钮正确复位，
  全程无崩溃；
- 测试 797 → **805**，全过；行为基线（11 张真实样张）零变化。

---

# v2.3.8 —— 界面美化：统一视觉 + 修掉两个真实的样式 Bug

对 GUI 做了一轮视觉整理，顺带修掉两个一直存在的渲染问题。为了「改前改后
有据可查」，先给项目加了 `tools/shot.py`：offscreen 渲染主窗口到 PNG
（自动指定 `QT_QPA_FONTDIR` 解决无头截图的中文豆腐块；可用 `--sample` 让它
真实跑一张样张、填上结果与历史再截图）。

## 一、修掉的两个样式 Bug

1. **颜色徽章渲染成暗红色块**：动态样式写的是
   `background:#e6a70022` —— Qt QSS **不支持八位 `#RRGGBBAA`**，解析失败后
   落到默认色。改为 `rgba(r, g, b, 0.14)` 函数形式（hex 拆成三分量）。
2. **「未识别到车牌」的提示色是无效十六进制** `#b0621`（5 位）。改为
   `#e5484d`。

## 二、视觉统一

* **布局**：工具条从画面**下方**移到上方——操作按钮在内容之上才符合
  自上而下的使用顺序；左侧「图像 + 工具条」、右侧「引擎 / 结果 / 历史 /
  日志」四张卡片。
* **设计令牌**：全站颜色收敛进 `_T` 字典（底色 / 卡片 / 描边 / 正文 /
  次要 / 主色 / 危险色），QSS 用 f-string 展开——改主题只动一处。
* **车牌展示卡**：车牌号原来是大字裸在白底上（黄牌黄字几乎看不清），
  现在放在深色展示卡上，字色跟随牌色（黄牌黄字、蓝牌蓝字，深底上都清楚），
  并加了 4px 字距。
* **颜色徽章**：从色块改为浅底胶囊（14% 透明底 + 55% 描边 + 同色文字）。
* **表格**：去掉行号列与网格线，行高 30 / 26、隔行浅底、表头灰蓝；
  日志改 Consolas 等宽。
* **控件**：按钮统一 32px 高、圆角 8，secondary 带描边有 hover 反馈；
  下拉框与按钮等高；进度条、菜单、滚动条、ToolTip 全部纳入同一套样式。

## 三、验证

`tests/` 全量 805 项通过；真实样张与空状态两张 offscreen 截图人工核对
（见 `docs/screenshot.png`）；改动文件 ruff check + format 全绿。

---

# v2.3.9 —— 界面二次打磨：车牌皮肤、状态栏，和一个被它揪出来的老 Bug

在 v2.3.8 的基础上继续打磨视觉，同时状态栏一上线就揪出了一个一直存在的
隐性不一致。

## 一、新视觉

1. **结果区渲染成"真车牌"**：号码不再是"深色卡片上染个字色"，而是按
   国标配色画成车牌本来的样子——蓝牌蓝底白字、黄牌黄底黑字、新能源
   绿底黑字、白牌白底黑字、黑牌黑底白字（危险品牌黑字变红）。一眼即知
   识别出了什么类型。皮肤表在 `_PLATE_SKIN`，全部用 `qlineargradient`
   渐变 + 同系描边。
2. **图像区改深色画布**（`#171b22`）：照片在深底上更突出，专业看图器
   的观感；空状态提示文字也换成了浅灰。
3. **底部状态栏**：`引擎 ｜ 隐私策略 ｜ 摄像头设备 ｜ 版本`，引擎随下拉框
   实时刷新、设备数随探测结果填充。
4. **卡片投影**：五张卡片统一 `QGraphicsDropShadowEffect`（Qt 的 QSS 没有
   box-shadow，只能用图形效果实现）。
5. **工具条卡片化** + 批量进度旁显示 `3 / 12` 文本。
6. **历史表空态**：空表格放一行居中的灰色提示「暂无记录」，首次识别自动
   让位（`setSpan` 合并整行）。

## 二、状态栏揪出的老 Bug：界面跑的引擎和配置各说各话

首次渲染截图里，状态栏写「引擎：cascade」，下拉框却选着「传统 CV 引擎」。
追查发现：配置默认引擎 `cascade` **不可用**时，`_refresh_engine_list` 会把
下拉框回落到第一项，但 `config.engine.default` 仍是 cascade——界面实际跑
的引擎与配置不一致，这个问题从引擎下拉框诞生起就存在，只是一直没人看见。
修法：回落时把配置同步成实际选中的项，并写日志说明；状态栏永远显示下拉框
的当前值。

## 三、顺带修掉：`lpr.__version__` 停在 2.3.1

写状态栏要显示版本号时发现 `lpr.__version__` 还是 2.3.1——与
pyproject / cli / 窗口标题各说各话。收敛为**单一来源**：

* `lpr/__init__.py` 的 `__version__` 是唯一字面量；
* `cli.VERSION = __version__`（`from . import __version__`）；
* 启动器、pyproject、窗口标题与它对齐；
* 版本一致性测试从"grep 字面量"升级为**值比较**（tomllib 读 pyproject，
  import 后比对），任何一处漂移都会红。

## 四、验证

805 项测试全过（含升级后的版本一致性测试）；真实样张与空状态两张
offscreen 截图人工核对（`docs/screenshot.png`）；改动文件 ruff 全绿。

---

# v2.4.0 —— 识别率：早停加严 + ROVER 风格读数融合

## 真实故障（samples/1.jpg，用户截图复现）

远距离黄牌 **京E·51619** 被识别成 **京S1619**。逐变体回放定位到三个缺陷叠加：

1. **OCR 丢字**：细笔画的 `E` 在 6 个预处理变体中被 5 个读丢（读成 `京51619`），
   仅第 3 轮（2.0x，无锐化）读对（置信度 0.721，还低于错读的 0.815）。
2. **纠错火上浇油**：`correct('京51619')` 的"第 2 位必须是字母"规则把数字
   `5` 按"字母被误读成数字"修成 `S` → 制造出**合法短牌** `京S1619`，
   真数字 5 反而被毁。纠错把 1 个字符错变成了 2 个字符错。
3. **早停太松**：旧版第 1 轮就允许停（`plate_score ≥ 0.6 且 share ≥ 0.5`），
   错读锁定时**读对的变体根本没跑**。

整串投票救不了它：同一错读重复多轮、每轮单独计票，5 票错读必然碾压
1 票正确读数（实测 4.88 : 1.08）。

## 修复（src/lpr/engines/deep_engine.py）

### 1. 早停三道闸门（全部满足才停）

* 至少 `_MIN_STOP_ROUNDS=2` 轮——单轮读数没有跨变体一致性证据；
* 领手必须是 **7/8 位合法车牌**——6 位"短牌"读数常是丢字后被纠错
  "修"出来的假合法串，必须跑满计划给正确读数出场机会；
* 票份过半。领手存在弱置信位（逐字符 < 0.7）时继续收集证据，
  封顶 `_MAX_WEAK_ROUNDS=4` 轮控制耗时。

### 2. ROVER 风格逐列读数融合（`_fuse_reads`）

把同一候选区所有变体的**原始读数**（清洗后、未抽取纠错）用
`difflib.SequenceMatcher` 对齐到最高分参考串，逐列加权投票：

* **多数列**：取加权多数字符（一致时无操作）；
* **插入列**：参考串缺失、其他读数在该位置稳定多出的字符
  （如被丢掉的 E），权重 ≥ `_FUSED_INSERT_RATIO×参考串分数` 才补回；
* **删除列**：不删（OCR 丢字远比多字常见）。

融合串经 `_extract_plate` 闸门滤噪后作为候选加入投票；与整串投票
领手不一致时，**结构分（`plate_score`）更高者胜**——逐列融合代表
所有变体的真实共识，优先级高于可被重复错读带偏的整串票数。

## 效果（11 张真实样张新旧对照）

| 样张 | 旧版 | v2.4.0 |
|---|---|---|
| 1.jpg | 京S1619 ✗ | **京E51619** ✓ |
| police.jpg | 鄂F8888（丢红字「警」）| **鄂F8888警** ✓（顺带修复）|
| 其余 9 张 | — | 完全一致（零回归）|

定位引导路径的最坏耗时从 1 轮早停变为跑满 6 轮计划
（约 +100~300ms/候选），典型清晰车牌仍为 2 轮早停。

## 测试

新增 `tests/test_deep_accuracy.py`（13 项）：`_fuse_reads` 纯函数
6 项 + 投票行为 7 项（真实故障回放、早停最少轮数、短牌禁早停、
弱位续跑与封顶、融合候选入票审计）。全部用假识别器回放，不依赖模型。

全量 818 项测试通过；改动文件 ruff check + format 全绿。

---

# v2.4.1 —— 修复：引擎下拉弹层条目「隐形」

## 故障（用户截图）

识别引擎下拉框展开后弹层是一片空白——条目其实在，但看不见。

## 根因

v2.3.9 的 QSS 给 `QComboBox QAbstractItemView` 设了**白底**，却没给条目
设**字色**（`::item` 无规则）。系统深色模式下默认调色板文字是浅色，
于是白底白字、条目全部隐形；下拉框本体因为单独设了 `color` 所以正常。

## 修复（src/lpr/ui/window.py）

补三条 `::item` 规则并显式上色：普通态深色文字 + 30px 行高 +
圆角；选中态浅蓝底；禁用态（不可用引擎）灰字。顺带把摄像头下拉
「未检测到摄像头」的行为确认为符合预期（用户机器 3 个摄像头条目
全是幽灵设备，Present=False，OpenCV 打不开任何一个）。

## 验证

offscreen 展开 showPopup() 截图核对：两个条目（传统 CV / 深度学习）
均清晰可见，选中项浅蓝高亮。

---

# v2.4.2 —— 分叉收口：合并后的静态检查补齐

`main` 与 `ci-fixes` 分叉过一段时间，合并时对 3 个冲突文件保留了 `main`
版本，于是 v2.3.6~v2.4.2 期间新引入的代码没经过静态检查。本次补齐：

- `ensure_bgr` 返回 Optional 的变量重绑定改为新变量名（`bgr`），
  避免 mypy 认定「可能为 None」；
- `np.float32(list)` 改为 `np.array(..., np.float32)`（mypy `arg-type`）；
- `m1 | m2` 改为 `cv2.bitwise_or`（mypy 不支持 ndarray 的 `|` 运算类型）；
- `locate_plates` 的 `logger` 参数标注为 `logging.Logger | None`；
- `DeepEngine._ocr` 标注为 `Any`（Optional 不可直接调用）；
- ruff format 统一 7 个文件。

无功能变更。ruff check / format / mypy 全量通过，fast 测试 813 项通过。

---

# v2.4.3 —— 历史行回显、版本号对齐，以及 CI 从 5 红到全绿

## 一、点识别历史行回显该条结果（GUI）

此前历史表只读：点中一行后上方结果区仍停在最后一次识别，看起来「历史与
结果不同步」。现在 `cellClicked` 取回行内绑定的 `PlateResult`，回显号码
皮肤、颜色徽章、ROI 与元信息（只更新展示，不再追加历史）。

顺带修掉一个隐患：`_history` 超过 `max_history` 截断时表格行未同步删除，
行号会与 `_history` 下标错位；现在截断时同步 `removeRow(0)`，且结果对象
改为直接绑在单元格 `UserRole` 上，不再依赖行号对齐。

## 二、版本号对齐

`v2.4.2` 被提交序列跳过（收口提交是 `style(ci)`，无版本语义），版本号直接
从 2.4.1 对齐到 2.4.3。此前提交标题已写 v2.4.3，但三处版本字面量
（`src/lpr/__init__.py`、`tools/launcher.py`、`pyproject.toml`）仍停在
2.4.1，导致 tag 无法诚实落地。补打 v2.3.1 / v2.3.9 / v2.4.0 / v2.4.1 /
v2.4.3 五个标签，逐一校验标签名与代码内版本号一致。

## 三、CI：三处根因，两轮修复

`main` 上 6 项检查一度挂了 5 项（另有一条 Windows 3.12 随后暴露），
根因是三类：

**1) Python 3.10 没有 `tomllib`** —— 它是 3.11 才进标准库的，而测试矩阵
覆盖 3.10。版本断言改为按 `[project]` 段解析 `pyproject.toml`，不引入
TOML 解析器，也就无需 `tomli` 回退依赖。

**2) mypy 报「未使用的 `type: ignore`」（Linux）** ——
`os.sched_getaffinity` 在 Linux 上存在，`# type: ignore[attr-defined]` 被
判为多余（在 Windows 开发机上恰好看不到）。ignore 代码列表补上
`unused-ignore`，两平台都成立；已用 `mypy --platform linux` 对照验证。

**3) 依赖版本漂移让准确率门槛变成抽奖** —— 两个层面：

- `rapidocr-onnxruntime` **1.4.x** 移除了 `text_recognizer` /
  `get_boxes_img_without_det`，`deep` 引擎的「纯识别」快路径失效，每张图
  退化成全图检测（0.3~1s+），对抗样张门槛测试在 CI 上超时；先把上界收到
  `<1.3`，并把 77 张对抗样张的单测按退化类型拆开，避免撞 300s 单测上限。
- 收到 `<1.3` 后仍有 Python 3.10 / 3.11 判红：`img/single_blue.jpg` 的
  `V` 被读成 `9`。证据是**同一个 onnxruntime 1.31.0** 下，3.12 / 3.13
  装到的 **1.2.3** 读对，3.10 / 3.11 装到的 **1.2.13** 读错，判别因子就是
  rapidocr 自带模型的漂移。最终把 `deep` 依赖**钉死**为 `==1.2.3`
  （`tools/launcher.py` 兜底清单同步）。

**4) Windows 管道下的中文崩溃** —— CI 的 Windows runner 在 stdout 走管道
时编码是 cp1252，`cli.py` 与 `tools/launcher.py` 里 `print` 中文直接抛
`UnicodeEncodeError`，进程以 1 退出（`python -m lpr engines` 与
`start.bat` 两条路径都中）。两个模块各加一道 `_make_pipes_utf8()`：
stdout / stderr 不是控制台（管道或重定向）且当前编码不是 UTF-8 时改为
UTF-8；**控制台本身不动** —— 中文 Windows 控制台是 cp936，保留原编码
才不会花屏。本机用 `PYTHONIOENCODING=cp1252` 复现了原报错，修复后两条
路径退出码均为 0。

## 验证

本地 906 项测试全绿（约 2 分钟），ruff check / `ruff format --check` /
`mypy --platform linux` 全过。CI 上 6 项检查（3.10 / 3.11 / 3.12 / 3.13 /
Windows 3.12 / 静态检查）全部 success。

> 说明：GitHub 上历史提交旁边仍是红叉，那是**它被推上去那一刻**的快照，
> 不会随修复自动重跑；仓库健康状态只看最新提交。

---

# v2.4.4 —— 许可切换到 GPL-3.0，并新增 `yolo` 检测识别引擎骨架

## 一、许可：MIT → GPL-3.0-only

引入 YOLO 系权重（Ultralytics 血统）会带来 **AGPL-3.0 / GPL-3.0** 的传染性
义务，MIT 与之不兼容。因此在接入第三个引擎之前先把项目许可整体切到
**GPL-3.0-only**（单独一次提交，便于回溯）：

- `LICENSE` 换成 GPL-3.0 全文，并在文件头写下中文说明；
- 文件尾部**保留原始 MIT 许可全文** —— MIT 允许再许可，但要求保留原版权
  声明与许可文本，这一条不能省；
- 另存一份 `LICENSE.MIT.original` 作为证据快照；
- `pyproject.toml` 的 `license` 改成 `{ text = "GPL-3.0-only" }`，并补上
  `License :: OSI Approved :: GNU General Public License v3 (GPLv3)` 分类器；
- README 许可章节重写，明确「**v2.4.3 及更早版本以 MIT 发布，其许可状态
  不受本次变更影响**」——已经拿到的旧版本不回溯；
- 新增 `docs/THIRD_PARTY.md`，逐个列出运行时依赖的许可证（从已安装包的
  元数据里读出来的，不是凭印象写的）。

> 注意这不是「换成更严的许可」这么简单：GPL-3.0 要求**衍生作品整体以
> GPL-3.0 分发**。如果将来要接 Ultralytics 官方权重，须一并接受其 AGPL
> 条款；本项目的做法是把权重放在 `models/yolo/` 由用户自行准备，仓库
> **不分发**权重，从而不与上游许可产生直接冲突。

## 二、新增第三个引擎：`yolo`

前两个引擎（`cv` / `deep`）的定位都依赖**轮廓/颜色先验**，对大幅侧倾、
透视变形、运动模糊的车牌召回偏弱。第三个引擎改走「检测网络」路线：

| 维度 | 设计 |
|---|---|
| 检测头 | 一次前向同时输出 bbox + **四个角点** + 单/双层分类（YOLO-pose 式） |
| 优势 | 角点直接给出，省掉「先轮廓后矫正」的几何假设，侧倾/透视场景更稳 |
| 推理 | **纯 ONNX**（`onnxruntime`），不引入 PyTorch —— 打包体积与启动时间都可控 |
| 权重 | 仓库不分发，放 `models/yolo/plate_det.onnx` + `plate_rec.onnx` 即自动启用 |
| 依赖 | 独立 extra：`pip install -e ".[yolo]"`（`onnxruntime>=1.16`） |

本次落地的是**骨架**，已完成的部分：

- `src/lpr/engines/yolo_engine.py`：`BaseEngine` 实现、注册表接入、
  `is_available()` 双条件探测（onnxruntime **且** 权重文件在），不可用时
  返回**可行动**的原因（含缺失文件的绝对路径建议位置）；
- `Detection` 数据类：`corners`(4,2) / `box` / `layout` / `score`，
  带 `width` / `height` 便捷属性；
- 权重目录解析：从当前工作目录向上走到项目根，找 `models/yolo/`；
- `models/yolo/README.md`：权重放置位置、许可义务、CCPD 训练与导出指引；
- 自动被发现：`lpr engines` 会列出第四个引擎并标注「不可用」，
  GUI 引擎下拉同样出现（禁用态 + 原因提示）；
- `tests/test_yolo_engine.py`：17 项用例覆盖注册 / 可用性探测 / 路径解析 /
  默认参数 / `Detection` 语义 / 未实现路径；
- `pyproject.toml` 新增 `yolo` extra，README 同步更新引擎表、依赖表、
  安装说明与项目结构。

**尚未实现**（明确写在 `Known limitations` 里，不做假象）：

- `_detect()`：读 ONNX 输出 → 解码 bbox / 角点 / 分类 → NMS；
- `_recognize_all()`：按四角点做透视矫正 → 送入识别网络 → 语法纠错。

两者当前都会抛 `NotImplementedError` 并附实现指引，**不会静默返回空结果**。

## 三、验证

- 本机 940 项 pytest 全绿（约 2 分 13 秒）；
  > **勘误（v2.5.0 补）**：这个 940 是当时写下的估计值，**并不准确**。
  > v2.5.0 用 `pytest --co -q` 实测了 `87464a4` 这个提交的收集数，是 **851** 项。
  > README 里的同名数字已一并改为实测值。保留此行是为了不掩盖这个差错。
- `ruff check` / `ruff format --check` / `mypy --platform linux` 全过；
- `lpr engines` 实测输出四个引擎，`yolo` 正确显示为不可用并给出原因；
- GUI 引擎下拉离屏实测：3 个可选 + 1 个禁用（`enabled = False`）。

---

# v2.5.0 —— `yolo` 引擎补成可用：ONNX 推理链路全实装

v2.4.4 交付的是**只有外壳的引擎**：注册、探测、路径解析、CLI/GUI 接入都通了，
但两个核心方法一调就抛 `NotImplementedError`。v2.5.0 把这条路补完。

## 一、实装了什么

`_detect()` + `_recognize_all()` 合起来是一条**七步链路**，每一步都有对应断言：

| 步 | 做什么 | 关键约定 |
|---|---|---|
| 1 | 读 ONNX 会话，取第一个输入名 | 不写死 `"images"`，从 `session.get_inputs()` 读 |
| 2 | **letterbox** 到 `640×640` | Ultralytics 同款：`round(x-0.1)/round(x+0.1)` 分配 padding，填充值 114，`ratio = min(S/h, S/w)` |
| 3 | 前向推理 | 输出可能是 `(1, C, N)` 也可能是 `(1, N, C)`，靠「属性数 ≪ 锚点数」判朝向 |
| 4 | **输出解码** | 置信度 = `obj × max(cls)`（有 objectness）或 `max(cls)`（无）；box 是 `cx,cy,w,h` |
| 5 | **类别感知 NMS** | 单层牌与双层牌框重合时**不能互相压制**，所以按类别分开做 |
| 6 | **逆 letterbox** | 减 padding、除 ratio，把框还原回原图坐标 |
| 7 | **四点透视矫正** → 识别 | 用 `locate._order_quad` 把角点归到「左上→右上→右下→左下」，再 `_warp_plate` 拉正 |

识别侧**刻意不重写**：复用 `deep` 引擎已有的 OCR 与语法纠错。为此给
`deep_engine.py` 加了公开入口 `recognize_regions(regions, layout_hint=...)`，
并让内部 `_recognize_candidates()` 接受版式提示 —— `yolo` 已经从检测头
拿到了单/双层分类，没必要让识别侧再去猜一遍行分割。

> 设计取舍：把识别做成「可选后端」而不是硬依赖。只装 `.[yolo]` 时引擎仍然
> 能跑，只是退化成**纯检测模式**（给出框 + 单/双层，文本为空）；装了
> `.[yolo,deep]` 才是完整链路。这让「只想用检测定位」的用户不必背上 OCR 依赖。

## 二、三条取权重的路（都实装了）

| 路 | 做法 | 工具 |
|---|---|---|
| **A. 取现成权重** | 公开可下载的检测权重：带 SHA256 校验下载 + 加载体检 | `tools/fetch_yolo_weights.py --list / --url / --github / --inspect` |
| **B. 自己训练** | CCPD → YOLO-pose 数据集 → 训练 → 导出 ONNX | `tools/ccpd_to_yolo.py` + `tools/train_plate_yolo.py` |
| **C. 纯检测模式** | 只要框和单/双层分类，不接识别 | `YoloEngine(..., recognize_backend="none")` |

**关于 A 的现实**：动手查过之后发现，主流权重要么在 Git LFS 指针后面
（`curl` 下来是几十字节的文本，`onnxruntime` 报 "Protobuf parsing failed"），
要么在百度网盘。所以这个工具**不是"一键可用"的自动装配器**，定位是
**校验与体检**：`--list` 列出已知来源及其许可/布局，`--url` 强制要求
`--sha256`（除非显式 `--insecure-no-hash`），`--inspect` 会真正建会话、
跑一次空输入、打印输入名与输出形状。把「拿到的文件到底能不能用」这件事
变成一条命令。

**关于 B 的一个细节**：CCPD 文件名里的四角点顺序是
「右下→左下→左上→右上」，而 YOLO-pose 要求「左上→右上→右下→左下」，
中间差一次置换 `[2, 3, 0, 1]`。转换脚本实测验证了这一点（用坐标级断言钉住，
不是"看上去对"）。另外 `flip_idx` 必须是 `[1, 0, 3, 2]` —— 水平翻转时
左上/右上互换、右下/左下互换，写成顺时针轮转 `[1, 2, 3, 0]` 是错的
（测试里专门放了一个反例证明它错）。

## 三、三个值得记下来的坑

### 1. 输出布局存在**形状上无法消解**的歧义

YOLO-pose 的输出属性序列有两种主流写法：

- Ultralytics 系：`box(4) + cls(nc) + kpts(8)` —— 类在前
- `we0091234` 系：`box(4) + obj(1) + kpts(8) + cls(nc)` —— **关键点在前**

当 `nc = 2` 时两者属性数**都是 15**。也就是说，拿到一个 `(1, 15, N)` 的张量，
**只看形状永远分不出是哪种**。之前的骨架代码用「属性数」硬套模板，遇到
第二种就会把关键点的 x 坐标当成分数读。

处理方式：不猜，把它变成显式参数。引擎新增 `attr_order`（`"cls_first"` 默认 /
`"kpt_first"`），并补了一组专门测试 —— 同一个张量按两种顺序解，断言结果
**必须不同**，且只有正确的那种能与手算的角点对上。

> 这类「两种合法解释在观测上等价」的问题不能靠启发式修，只能靠显式声明。
> 代码里给错的症状是「框看着对、角点乱飞」，非常容易误判成模型坏了。

### 2. CTC 解码的转置判据：用字符表长度，不要比大小

识别网络输出可能是 `(T, C)` 也可能是 `(C, T)`（T = 时间步，C = 字符表大小）。
顺手写成 `if arr.shape[0] < arr.shape[1]: arr = arr.T` 在多数情况下对，
但**当时间步少于字符表大小时会翻错**，输出一串乱码。正确判据是
`arr.shape[1] == len(charset)`，两者都不等时直接报错而不是猜。测试里
`test_orientation_uses_charset_length_not_size_comparison` 构造了
`T < C` 的情形来钉死它。

### 3. 类方法 `is_available()` 不能拿来在实例里判断可用性

`describe_engines()` 要求 `is_available()` 是**类方法**（它要能在没有实例时
回答"这个引擎能不能用"），所以它只看默认目录 `models/yolo/`。但 `load()`
如果图省事直接调它，那么**把权重放在别处的用户**就会遇到"文件明明在、
却报找不到" —— 这个 bug 在联调时真实出现过。修法是拆成三件：
`_probe_path()`（静态，判单个文件）、`_check_available()`（实例，用自己的
`models_dir`）、`_missing_model_message()`（拼报错文案）。测试
`test_instance_check_honours_custom_models_dir` 守住这条。

## 四、没有第三方权重也能验证整条管道

这是本次**最关键的工程决定**。真权重有体积与许可两个问题，不能进仓库；
但"管道通不通"必须能**无条件**验证，否则这条链路就等于没有 CI 覆盖。

做法是 `tools/make_stub_onnx.py`：生成**常量输出**的桩 ONNX 模型
（一个 `Constant` 节点，直接吐出写死的张量）。它覆盖 5 种布局：

| 布局 | 形状 | 含义 |
|---|---|---|
| `v8-pose` | `(1, 4+nc+8, N)` | Ultralytics pose，类在前 |
| `v5-pose` | `(1, N, 4+1+nc+8)` | Ultralytics pose，带 objectness |
| `kpt-first` | `(1, N, 4+1+kc+8)` | `we0091234` 系，关键点在前 |
| `v8-det` | `(1, 4+nc, N)` | 纯检测 |
| `v5-det` | `(1, N, 4+1+nc)` | 纯检测，带 objectness |

因为桩模型的框是**已知常量**（在 640 的 letterbox 坐标系里写死
`cx=320, cy=320, w=200, h=64`），端到端测试可以**逐位对齐**坐标：

```
800×480 输入 → ratio 0.8、上边 padding 128
桩框 (220,288)-(420,352) → 期望还原为 (275,200)-(525,280)
```

5 种布局解出来的框**必须完全一致** —— 这比"跑通就行"强得多：它同时验证了
朝向判定、置信度取法、逆 letterbox 的算术三件事。此外还用它验证了
NMS 去重（第二条与第一条 IoU ≈ 0.87，必须被压掉）、阈值过滤、
几何合理性过滤（宽高比越界丢框）、`attr_order` 给错必须解不出正确结果、
纯检测模式返回空文本、以及**旋转车牌**的角点矫正。

`onnx` 因此进了 `dev` extra（仅测试/造桩用，运行时不需要）。

## 五、顺手修掉的两处不一致

- `yolo` 之前**注册了但选不了**：`config.EngineConfig.validate()` 的白名单、
  CLI 两处 `--engine` 的 `choices`、`api` 的文档都没带上它。现在收敛成
  `config.ENGINE_CHOICES` 单一事实源，四处引用同一份；并补了
  `test_engine_is_selectable_from_config_and_cli` 直接从 `_make_parser()`
  拿 choices 来断言。
- `bench_workers.py` 的 `--engine` 同样跟着 `ENGINE_CHOICES` 走。

## 五之二、联调时抓到的第四个坑：复用识别后端会**丢候选**

这个坑是跑端到端时发现的，值得单独记 —— 它是**静默错误**，不报错、不抛异常。

`deep` 的 `recognize_regions()` 对「多尺度全轮都读不出东西」的候选是
**直接 `continue`**（这本身合理：没有读数的候选不该产生结果）。于是它的
**返回列表会比入参短**。而 `yolo` 侧原本这样配对：

```python
for result, (_roi, _bbox, _layout, score) in zip(results, prepared, strict=False):
```

按**位置**配对。两个后果：

1. **错位**：只要中间丢一个，后面全部挪位 —— A 的号码会挂上 B 的框和检测分。
   `strict=False` 让 `zip` 静默截断，连异常都不会有。
2. **丢框**：被丢掉的候选**连同检测框一起消失**。用户看到"未识别到车牌"，
   但检测其实成功了 —— 这正好与"纯检测模式保框留白"的既定口径自相矛盾。

修法是两条：

- 改成**按 bbox 配对**（bbox 是我们自己传进去的，`deep` 原样写回，可精确匹配）；
- 识别侧没产出结果的候选，**按纯检测口径补回**：保留框与 ROI、`text=""`、
  置信度用检测分，并在 `meta` 里打 `ocr_failed: True` —— 这样"没检到"与
  "检到了没认出"在数据上可区分。

同时补了 4 项测试（`TestReusedRecognizerAlignment`）。为了确认测试**真的**
能抓住老代码，我把旧实现临时贴回去跑了一遍：

```
E  AssertionError: 丢掉的候选被静默吃掉了
E  assert 1 == 2
E  assert 0.855 == 0.4275 ± 1.0e-04     ← 恰好是错位：B 的结果挂着 A 的检测分
FAILED ...test_lost_candidate_keeps_its_box_with_empty_text
FAILED ...test_results_are_matched_by_bbox_not_by_position
```

> 有个细节差点让测试变成摆设：第一版让两条桩框共用同一个检测分，
> 于是"错位"这个错误**在数值上不可见**（0.855 贴给谁都是 0.855）。
> 改成第二条分数减半后才测得出来。写回归测试时要盯的正是这个 ——
> 断言必须落在**会随错误改变**的量上。

## 六、验证

- 本机全量 pytest **930 + 4 = 934 项全绿**（用 `--junitxml` 读数，
  避免终端宽度把汇总行截掉）；
  - `tests/test_yolo_engine.py` **71** 项（原 17 项 → 71 项）、
    `tests/test_ccpd_to_yolo.py` **29** 项（新增）；
  - 对账：`87464a4` 收集 **851** 项 → 851 − 17 + 71 + 29 = **934** ✓
- `ruff check` / `ruff format --check` / `mypy` 全过（mypy: 25 个源文件 0 问题）；
- 桩模型端到端：5 种布局解出的框坐标逐位一致，且与手算的
  `(275.0, 200.0, 525.0, 280.0)` 相同；
- `tools/make_stub_onnx.py --selftest` 可在无第三方权重的机器上独立复现。

> 顺带修正一处历史数据：README 与 v2.4.4 条目的「940 项」是估计值，
> 实测为 851 项（见上）。这条留痕是为了让后来的读者知道数字是怎么来的。

## 七、仍未做到的（不做假象）

- **没有"一键可用"的公开权重**，原因见第二节；用户仍需自行取权重或训练；
- `attr_order` 目前只能在 Python 里传（`YoloEngine(attr_order="kpt_first")`），
  CLI/GUI 尚未暴露该开关 —— 加开关需要先想清楚"给错就静默出错"的 GUI
  提示怎么做，留待后续；
- 自训练路线需要 PyTorch + Ultralytics，**不在任何 extras 里**（体积原因），
  也没有在本机跑通完整训练（数据集体积 12 GB），只验证到"数据集格式正确 +
  `data.yaml` 自检通过 + 导出脚本能调用"；
- 与 `cv`/`deep` 的**准确率对比没有做**：手里没有带标注的侧倾车牌评测集，
  拿不到可信数字，所以 README 里只写"定位召回优于轮廓法"这个**机制性**判断，
  不写具体百分比。


---

## v2.5.1 —— yolo 引擎真实权重跑通 + `attr_order` 进配置

日期：2026-10-15

### 一、这次解决的是什么

v2.5.0 交付时 `yolo` 引擎"代码可用但实际不可用"——手里没有能落盘的
`plate_det.onnx`，GUI 下拉框里它一直显示**（不可用）**。本次把这条
"最后一公里"走通了：**真实 GPL 权重已导出为 ONNX 并本地验证通过**。

### 二、权重获取实录（可复现）

1. `we0091234/Chinese_license_plate_detection_recognition` 的
   `weights/plate_detect.pt`（1.17 MB）**可以下载了**（v2.5.0 时记录的
   "LFS 指针/网盘" 状况已变化，`main` 分支直接挂普通文件）。
   raw.githubusercontent 不通时，可走 `api.github.com` 的 Git Blobs API
   （base64）拿文件；仓库源码用 `.../tarball` 端点拿。
2. 用**官方仓库自带的 `export.py`** 导出 ONNX（不是 Ultralytics CLI）：
   - 依赖：`torch` / `torchvision` / `onnx` / `onnxscript` / `PIL` /
     `scipy` / `thop`（装在临时 venv，不进项目依赖）；
   - **坑 1**：torch ≥ 2.6 默认 `weights_only=True`，旧检查点加载直接炸，
     给 `attempt_load` 的 `torch.load` 补 `weights_only=False`；
   - **坑 2**：新导出器把权重写成**外部数据文件** `plate_detect.onnx.data`，
     引擎加载会报 `External data path validation failed`。用
     `onnx.load(...)`（自动带上外部数据）+ `onnx.save_model(...,
     save_as_external_data=False)` 内嵌成**单文件** 3.19 MB；
   - **坑 3**：codeload 的 tar 端点 404，改用
     `api.github.com/repos/<repo>/tarball/<ref>`；下载中途被杀会留下
     截断的 tar.gz，`tar -tzf` 能查出来。仓库整体 33 MB，走代理太慢时
     可以按 Git Trees API 逐文件抓 `models/` + `utils/` 的 .py（共 42 个）。
3. 体检：`tools/fetch_yolo_weights.py --inspect` 报
   `output [1, 25200, 15]`，与预期布局一致（`kpt_first`，has_objectness）。

### 三、端到端实测（真实权重，非桩模型）

| 样张 | 结果 |
|---|---|
| `samples/1.jpg` | `京E51619`（黄牌）+ `京PAP315`（蓝牌），置信度 1.00，检测分 0.89/0.72 |
| `samples/2.jpg` | `京AD77972`（绿牌新能源），置信度 0.98 |

识别复用 deep 引擎的 OCR（`recognize_backend="auto"`），底色分类一并给出。
对比一个反面数据：同样的图走错列序（cls_first）时输出乱码、
"置信度" 1.40（合法值 ≤ 1）——列顺序给错的破坏性可见一斑。

### 四、`attr_order` 进配置（修掉 v2.5.0 留下的缺口）

`EngineConfig` 新增三个字段，GUI/CLI 无需任何开关即随配置生效：

- `yolo_attr_order`：**默认 `"kpt_first"`**（对齐路径 A 的 we0091234 权重，
  绝大多数用户取权重就走这条路；Ultralytics 导出的请改 `"cls_first"`）；
- `yolo_recognize_backend`：`auto` / `crnn` / `deep` / `none`，默认 `auto`；
- `yolo_models_dir`：权重目录，空 = 默认 `models/yolo`。

优先级：**显式实参 > config > 引擎默认值**；非法值在 `validate()` 拒绝。
此前"只能在 Python 里传"的缺口由此关闭。

### 五、验证

- 本机全量 pytest **941 项全绿**（`941 passed, 126.1s`，`--junitxml` 读数；
  新增 `TestConfigPlumbing` 7 项：4 个优先级用例 + 1 个字段透传 +
  2 个参数化校验）；
- `ruff check` / `ruff format --check` / `mypy` 全过；
- CLI `lpr predict --engine yolo` 与引擎状态页均走默认配置实测。

### 六、仍未做到的（不做假象）

- **权重文件本身仍不进仓库**（`.gitignore` 里 `models/yolo/*.onnx` 不变），
  本节只保证"照着做 10 分钟内能落盘"；
- 自训练路线仍只验证到数据集格式与导出脚本可调用，没跑完整训练；
- 与 `cv`/`deep` 的准确率对比仍缺带标注的评测集，维持机制性表述。

---

## v2.6.0 —— 第四引擎 `fusion`：三合一（cv + deep + yolo）

日期：2026-10-15

### 一、为什么要有它

三个引擎的强项与失误模式是**互补**的：`cv` 在清晰正视角下很快但字符错误率高、
且在复杂大图上大量误检；`deep` 中文与模糊字符识别强，但靠形态学定位，
远景小牌容易漏；`yolo` 用检测网络找车牌，侧倾/透视/远景召回好。
既然三者的错误不重合，"让它们同时上、再投票"就是最直接的收益点。

### 二、三种策略（`engine.fusion_strategy`）

| 策略 | 做什么 | 代价 |
|---|---|---|
| `ensemble`（默认） | 三个成员各跑一次 → 按空间重叠分组 → 组内按 `成员权重 × 置信度` 投票 | 耗时≈三者之和 |
| `chain` | 依次尝试，拿到 `confidence ≥ fusion_chain_accept`（默认 0.75）就停 | ≈最快命中的那个引擎 |
| `auto` | 按长边分辨率 + 拉普拉斯方差排序，再按序试、命中即止 | 单引擎 |

配套旋钮：`fusion_members`（成员与顺序）、`fusion_engine_weights`（票权）、
`fusion_group_iou`（分组阈值）、`fusion_min_support` / `fusion_trusted_weight`
（准入，见下）。

### 三、三条设计决定，以及它们各自的数据依据

**1. 融合结果的 `confidence` 仍是"胜出成员的原始置信度"，不是票占比。**
下游有太多地方按 `confidence` 设阈值（安全层、跟踪器投票、报告里的置信度列），
把语义换成"一致度"会让这些逻辑全部失准。融合特有的信息进 `meta["fusion"]`：
`votes`（号码→票数）、`agreement`（一致度）、`support`（几个成员投了它）、
`contributors`（每个成员读了什么）、`elapsed_ms`（各成员耗时）。

**2. 加了"准入规则"：孤立候选要么凑到第二个成员的票，要么票权够高。**
这一条是被实测逼出来的，不是设计洁癖。同一张 `samples/1.jpg`（4096×2304）：

| 引擎 | 输出 |
|---|---|
| `cv` | **9 条**候选（`川J111WE`、`赣A8B111`、`浙S1EB81`…其中 7 条是误检） |
| `deep` | 1 条（`京E51619`） |
| `yolo` | 2 条（`京E51619`、`京PAP315`）—— 全中 |

若融合只做并集，输出会是 13 条，最弱成员的误检把结果淹没。于是默认
`fusion_min_support=2`：孤立候选需要第二个成员背书，除非它的票权达到
`fusion_trusted_weight=1.0`（`yolo`/`deep` 默认 1.0，可以独存；`cv` 默认 0.6，
独有结果会被滤掉）。**单成员在岗时自动放宽到 1**，否则会全滤成空。
滤掉的候选**不静默**：`FusionEngine.last_dropped` 可查，日志会列出号码。

实测收敛效果：

| 样张 | 融合前（并集） | 融合后 |
|---|---|---|
| `samples/1.jpg` | 9 + 1 + 2 = 12 条（含 7 条 cv 误检） | **2 条全对**（`京E51619` support=2、`京PAP315` yolo 独存），滤掉 6 条 |
| `samples/2.jpg` | 7 + 1 + 1 = 9 条 | **1 条全对**（`京AD77972` support=2），滤掉 6 条 |

`京E51619` 那组的三方投票明细：`yolo` 与 `deep` 都读 `京E51619`（合计 2.0 票），
`cv` 读 `苏A7Z119`（0.6 票）—— 少数派意见完整保留在 `contributors` 里。

**3. `auto` 如实承认是单引擎，并且默认偏好 `deep` 在大图上会漏远景牌。**
自动选优的规则只有两条判据（长边、模糊度），没有玄学。实测代价：
`samples/1.jpg` 是"大图 + 清晰"→ 选 `deep` → **只报 1 块**车牌（漏了远景的
`京PAP315`），而 `yolo` 能报出 2 块。这条写进了 README 已知限制：
要召回就用 `ensemble` / `chain`，别指望 `auto`。

### 四、工程细节

- **成员是共享单例**（走 `get_engine`），模型只加载一次；`FusionEngine.close()`
  只丢引用**不关成员** —— 否则会顺手把别的引擎砸了（有测试盯住）。
- **成员退席不牵连他人**：加载失败（缺权重/缺依赖）或单次调用抛异常，都只记进
  `member_errors` 并跳过；**全部成员都不行**才抛 `EngineUnavailable`，异常里
  逐个列出原因。
- **成员可用性用"试错"而非"预判"**：不去看 `is_available()` 的类方法探测结果
  （它只看默认权重目录），而是直接建实例 —— 用户把 `yolo` 权重放到自定义目录时，
  探测会说不可用，真加载反而是成的。
- 新增 `tests/test_fusion_engine.py` **77 项**：纯函数（分组/投票/准入/模糊度/
  选优）、编排（三策略路由、成员退席、异常隔离、生命周期、`member_factory`
  注入假成员）、注册与配置校验。

### 五、验证

- 本机全量 pytest **1018 项全绿**（941 + 77），`1018 passed, 124.2s`；
- `ruff check` / `ruff format --check` / `mypy`(26 files) 全过；
- CLI 实测：`lpr predict samples/ --engine fusion` → 2 张全中，平均 1.7 s/张；
- API 实测：`Recognizer(engine="fusion")` 走通；
- GUI：下拉框自动出现「三合一引擎」（`describe_engines()` 驱动，无界面改动）。

### 六、仍未做到的（不做假象）

- **没有做多引擎准确率的量化对比**：样本只有 2 张真实样张（各 1~2 块车牌），
  不足以给出百分比。上面写的是**逐张的实测条数与号码**，不是"提升了 X%"。
  要做准得先建带标注的侧倾/小目标评测集；
- `fusion` 的耗时是三者之和（2.2 s/张），**没有做并行**：底层推理后端共享
  会话，多线程只会互相抢 CPU（`INFERENCE_LOCK` 就是为这件事存在的）；
- **GUI 里没有"融合明细"面板**：`meta["fusion"]` 已经带了票数、一致度、
  各成员读数，但界面暂时只显示融合后的号码，谁投了什么要看日志或导出 JSON。

# v2.7.0 —— 准确率实测：六个配置横向对比、失败按环节拆解，揪出一个路由 bug

> 起因是用户一句"看看各个模型的准确率还有提升办法"。量完发现：
> 有几个结论与直觉相反，还有一个真 bug 被数据揪了出来。
> 完整数据与逐条依据在 **[docs/ACCURACY.md](ACCURACY.md)**，这里只记结论与改动。

## 一、量了什么

两套互为补充的实测集：

| 集 | 张数 | 特点 |
|---|---|---|
| CCPD 标注集（`datasets/ccpd_labeled/`，720×1160 原始分辨率） | 100 | 常规牌为主（base 67），含 weather/blur/tilt/challenge/fn/db/rotate |
| 自有样张（`tests/baselines/ground_truth.json` 的人工真值） | 10（有效） | 特殊牌：双层、黄色、港/警/领/学 尾字 |

指标分**识别**（图级严格命中为主口径）与**定位**（以文件名 bbox 为真值）两组，
另加耗时与误检数。关键设计：失败不是笼统的"没识别对"，而是按
**卡在哪一环** 拆成四类互斥主类 —— `无框（定位失败）` / `有框无字（读不出来）` /
`形近字（差 1 字符）` / `严重错读（差 ≥2）`，因为四类的"药方"完全不同。

## 二、CCPD 100 张的实测

| 配置 | 严格命中 | 定位命中 | 定位精确率 | 平均IoU | 中位ms | 误检 |
|---|---|---|---|---|---|---|
| cv | **0/100** | 76.0% | 16.6% | 0.643 | 36 | **691** |
| deep | 79/100 | 82.0% | 88.5% | 0.670 | 716 | 14 |
| **yolo** | **84/100** | **100.0%** | 98.0% | **0.878** | **125** | 7 |
| fusion(ensemble) | 84/100 | 92.0% | 97.9% | 0.805 | 1737 | **5** |
| fusion(chain) | 84/100 | 96.0% | 76.8% | 0.840 | 190 | 38 |
| fusion(auto) 改前 | 80/100 | 92.0% | 89.6% | 0.764 | 749 | 15 |
| fusion(auto)@yolo 改后 | **84/100** | **100.0%** | 98.0% | 0.878 | 146 | 7 |

四个反直觉的结论：

1. **`cv` 是 0/100** —— 不是差一点，是一张都没读对。单图对照显示
   它**定位到了真牌区**（bbox 与 deep 完全相同），但把 `皖A·U3Z28`
   逐字符读成 `黑B158T8`，且置信度饱和在 1.000。SVM 对分布外样本
   过度自信 + 691 条误检，说明它的识别端在这两套数据上完全失效；
2. **`yolo` 全面领先 deep**：识别 84% vs 79%、定位 100% vs 82%、
   速度快 5.7 倍（125 vs 716 ms）；
3. **但排名在自有样张上反转**：特殊牌（港/警/领/学/双层）10 张里
   `deep` 10/10 全对，`yolo` 只有 8/10。**没有全能引擎**；
4. **`deep` 的命中集合是 `yolo` 的子集**（CCPD 上 deep 独占 0 张），
   所以"至少一个成员命中"的完美路由上界 = 84% = yolo 单独。
   **成员之间没有互补，融合救不回任何图** —— 这解释了为什么三种
   融合策略的严格命中都等于最强单成员。

## 三、失败按环节拆解（这是本文最有用的表）

| 配置 | 未命中 | 无框<br>（定位失败） | 有框无字 | 形近字 | 严重错读 |
|---|---|---|---|---|---|
| cv | 100 | 1 | 0 | 1 | **98** |
| deep | 21 | **10** | 0 | 8 | 3 |
| yolo | 16 | 0 | **8** | 7 | 1 |

读法：

* `deep` 的 10 张"无框"就是它输给 yolo 的主因 —— **定位/召回问题**，
  换识别模型没用；
* `yolo` 的 16 张失败里 **0 张是定位失败**（定位 100%），8 张是"有框无字"
  （blur 4 + db 2 + fn 1 + challenge 1），7 张形近字 —— **全是识别端的事**；
* 形近字混淆对抓到了具体字符：`Z→2`、`X→4`、`D→0`、`C→0`、`H→N`，
  以及**两个引擎都把省份读成 `皖`**（真值 `赣`/`苏`）—— 那是 CCPD 的
  数据偏差，不是模型缺陷，**不要据此在生产上加省份白名单**。

## 四、发现"同一个识别器"，它改写了提升路线的排序

`yolo` 引擎的 `recognize_backend="auto"` 在没有 `models/yolo/plate_rec.onnx`
时会**回退复用 `deep` 引擎的识别管线**，而本机恰好没有那个文件。

也就是说：**yolo 84% 与 deep 79% 的差距，全部来自定位与透视矫正的质量**
（平均最佳 IoU 0.878 vs 0.670），字符识别模型是同一个。
所以"换更强的识别模型"对两边同时生效，而"调定位"只对 deep 有效 ——
提升办法的性价比排序因此重排（见 ACCURACY.md §7）。

## 五、两个附加实验

### 5.1 第二识别模型：阴性结果（维持默认关闭，现在有数据背书）

仓库里躺着一份 10.8 MB 的 `models/ch_PP-OCRv4_rec_mobile.onnx`（未被 git 追踪），
`EngineConfig.secondary_rec_model` 已实装但默认关闭。开着它重跑：

| 配置 | 严格命中 | 中位ms |
|---|---|---|
| deep | 79.0% | 716 |
| deep + 第二识别器 | **79.0%（一張未变）** | **2042（×2.9）** |
| yolo | 84.0% | 125 |
| yolo + 第二识别器 | **84.0%（一張未变）** | 157 |

两引擎的失败集合**逐张完全相同**，deep 只有 1 张图的候选列表多了一条垃圾。
机制设计没错（只在该失败的路径上激活，所以零回归符合预期），
问题是它救不回任何东西 —— 这些失败不是"低置信"，是"高置信地错"。

### 5.2 auto 路由：一个被数据揪出来的 bug（已修）

`fusion(auto)` 默认 80%，**比它自己最强的成员 yolo（84%）还低 4 张**，
且一张都没多救。路由决策分布说明了原因：

```
auto 的选择：deep × 85，yolo × 15   ← 85% 的图走了"清晰大图 → deep"分支
丢掉而 yolo 命中的 4 张，全是被路由到 deep 的
```

根因：`select_single_engine` 清晰大图分支的偏好是**硬编码**的
`("deep", "yolo", "cv")`，用户配置的 `fusion_members` 顺序被悄悄无视。

**修复**（`src/lpr/engines/fusion_engine.py` + `src/lpr/config.py`）：

* 除"小图/糊图固定检测网络优先"外，清晰大图的首选成员改为可配置：
  新增 `EngineConfig.fusion_auto_clear_first`（默认 `"deep"`，**行为不变**）；
* 其余成员一律跟随 `fusion_members` 的配置顺序，不再用硬编码清单；
* 校验：`fusion_auto_clear_first` 必须是合法成员名。

实测切到 `yolo`：80% → **84%**（+4 张），定位 92% → 100%，中位 749 → 146 ms。

**为什么默认值仍留 `deep`**：两套数据指向相反（CCPD 要 yolo，
特殊牌样张要 deep 10/10 vs yolo 8/10），写死任何一边都会在另一边倒退。
所以交给用户，依据都写进了 ACCURACY.md。

## 六、工具链（都在仓库里，可复现）

* `tools/eval_ccpd.py` 泛化为多引擎评测器：
  * `--engines cv,deep,yolo,fusion:ensemble,fusion:chain,fusion:auto`
    （`fusion:策略` 语法，一条命令跑完全部对比）；
  * `--taxonomy` 输出失败分类 + 形近字混淆对；
  * `--secondary-rec PATH` / `--fusion-clear-first NAME` 两个实验开关；
  * 失败分类的键是稳定 ASCII（`no_box` / `no_text` / `near` / `gross`），
    中文标签由 `FAIL_LABELS` 单点定义；
* `tools/eval_report.py`（新增）：把评测 JSON 排成 markdown 表 ——
  总览 / 分子集 / 失败分类 / 互补性（含完美路由上界）/ 硬样本清单；
* `tools/ccpd_labels.py` 修了一个**真 bug**：`parse()` 早先只认 `&`
  坐标分隔符，而新版导出用 `,` —— 导致对整个新格式数据集**静默**返回
  `None`（评测拿到 0/100 的解析率而不报错）。现在两种都认，
  并新增 6 项测试钉住（含"未知分隔符必须返回 None 而不是错解析"）。

## 七、验证

* 新增 **59 项**测试（`tests/test_eval_ccpd.py` 47 + `TestCommaSeparatorFormat` 6
  + fusion `clear_first` 6：4 个用例 + 2 个参数化校验），
  本机全量 pytest **1077 项全绿**；
* `fusion(auto)@yolo` 的 +4 张是在真实 100 张上实测复验的，不是推算。

# v2.7.1 —— 启动器补上模型权重自检（附 `doctor` 只读体检）

> 起因：用户说「把启动文件完善下」。翻代码时发现一个**反馈链断点**：
> 启动器只探测 numpy / OpenCV / Pillow / PySide6，装齐就打印「均已就绪」，
> 而权重根本不在它的视野里 —— 于是 yolo 权重缺失时它照样报「就绪」，
> 用户看到的却是界面下拉框里那行「（不可用）」。

## 一、问题到底是什么

`models/` 下的资产分三类，而启动器原先一类都不看：

| 资产 | 归属 | 谁需要它 |
|---|---|---|
| `svm.dat` / `svmchinese.dat` | `cv` | 随仓库分发；缺了 `load()` 直接抛 |
| `rapidocr_onnxruntime` / `paddleocr` | `deep` | **二选一**；模型 13.7 MB 在 wheel 里 |
| `models/yolo/plate_det.onnx` + `onnxruntime` | `yolo` | **都不进仓库**：默认安装规格 `.[deep,gui]` 根本不含 `onnxruntime` |

所以「装完依赖」和「引擎能用」之间隔着一整层。更糟的是这层失败是**静默**的：
启动器没有报错，界面只是把那个引擎标成「（不可用）」，用户拿不到下一步该做什么。

## 二、改了什么

### 2.1 启动流程：四步 → 五步

新增第 4 步，它只陈述事实，**不阻断启动**：

```text
[1/5] 解释器正常
[2/5] 依赖齐全
[3/5] 无需安装
[4/5] 模型权重与引擎
      可用：传统 CV 引擎 / 深度学习引擎 / 三合一引擎
      不可用：检测识别引擎 —— 未安装 onnxruntime（检测推理后端）；缺少模型权重：models/yolo/plate_det.onnx
              修复：pip install -e ".[yolo]"
                    python tools/fetch_yolo_weights.py --list          # 已知来源与许可
                    python tools/fetch_yolo_weights.py --github <owner/repo> --asset "*det*.onnx"
      失去的只是对应引擎，界面与其他引擎照常可用；
      要看完整报告执行：start.bat doctor
[5/5] 正在启动...
```

顺手改掉了一句**不实的成功提示**：原来第 3 步结束后打印「完成 —— CV 引擎、
深度引擎、图形界面均已就绪」。它当时就只保证 Python 依赖，现在改成
「Python 运行库与图形界面已就绪」，引擎的结论交给第 4 步。

### 2.2 新增 `start.bat doctor`（别名 `--doctor` / `--check`）

只体检不启动。四条刻意的设计：

1. **只读**：不建 `.venv`、不装包、不改任何文件 —— 环境已经坏掉时才最需要它，
   所以它不能在诊断过程中继续动环境。用测试钉住了（把 `create_venv` /
   `install_deps` / `launch` 换成抛异常的桩，再比对运行前后目录快照）。
2. **退出码即结论**：四个引擎全可用 → 0，否则 1，脚本里可以直接判。
3. **不制造假警报**：只装了 `rapidocr` 没装 `paddleocr`（二者其一即可）标 `[--]`，
   只有真正拖垮某个引擎才标 `[!!]`。假警报比不报还糟 —— 用户会去修一个
   本来没问题的地方。
4. **一次报全**：`yolo` 同时缺 `onnxruntime` 与权重是最常见的新手形态，
   两条一起给，免得「装一个 → 重启 → 发现还缺一个」。
   原因里**必须写出模块名**（「未安装 onnxruntime」而不是「缺少后端」），
   否则用户没法照着修 —— 这条是实现时就写错、被测试抓出来的。

### 2.3 判定口径与引擎对齐（这是最需要小心的地方）

启动器**只允许依赖标准库**（有守卫测试盯着它不许 `import lpr`，否则它会在
最需要它工作的那台机器上 `ImportError`），所以它不可能去问引擎「你需要什么」，
只能自带一份清单。清单一旦漂移，启动器就成了**第二个「假真相」**：
它说齐全、引擎说不可用，用户不知道该信谁。

因此：

* 清单与引擎常量**逐条比对**，写成了测试（`TestAssetManifest`）：`cv` 的两个
  `.dat` 走 `lpr.config.MODELS_DIR`，`yolo` 的三个文件名直接取
  `yolo_engine.DETECT_MODEL_NAME` / `RECOGNIZE_MODEL_NAME` / `CHARSET_NAME`
  与 `locate_model_dir()`；引擎中文名与 `describe_engines()` 的
  `display_name` 也必须一致。
* **可用性判定不许比引擎更乐观**（`TestVerdictAgainstEngines`）：启动器说
  「可用」的，`is_available()` 必须也说可用；环境齐全时两者要**逐引擎完全一致**。
* 探测**真去 `__import__`**，不用 `importlib.util.find_spec`：装入一半
  （DLL 缺失、依赖不全）的包 `find_spec` 照样返回路径，于是启动器报「可用」、
  引擎报「不可用」—— 正是本节要消灭的那种分歧。
* 多个模块**合并到一个子进程**探完（Windows 上解释器冷启动本身 ~0.5 s，
  逐个起进程会让每次启动多等好几秒）；`probe_modules` 的结果用带标记的前缀
  打印再解析，避免被测模块自己往 stdout 写日志时污染 JSON。

### 2.4 踩到并绕开的一个坑：`@dataclass` 在按路径加载时炸

第一版用的是 `@dataclass(frozen=True)`，结果**测试直接红了**：

```text
AttributeError: 'NoneType' object has no attribute '__dict__'
  dataclasses.py, in _is_type:  ns = sys.modules.get(cls.__module__).__dict__
```

`start.bat` 与 `tests/test_launch_env.py::_load_launcher` 都是
`spec_from_file_location` + `exec_module` 这种**按文件路径**的加载方式，
它不把模块注册进 `sys.modules`；而 `@dataclass` 在装饰期要读
`sys.modules[cls.__module__]`。实测确认 `NamedTuple` 与
`collections.namedtuple` 都没这个问题，于是三个值对象全部改用 `NamedTuple`
—— 一个连自己都起不来的启动器，比不完善的启动器糟糕得多。
（同时把测试的加载器按标准写法补上了 `sys.modules` 注册。）

### 2.5 顺带修的小事

* `start.bat` 头部注释里写着 `v2.3.6`，而项目已经到 2.7.x。现在**不写版本号**了，
  真实版本由 Python 在启动时打印 —— 一个放在「用户会双击的文件」里的过期版本号，
  比没有更糟。头部同时补上了 `doctor` 的说明。
* 引擎表的对齐：`lpr engines` 用的是朴素的 `f"{name:<16}"`，中文按字符数算宽度，
  所以列是歪的。启动器这边用 `unicodedata.east_asian_width` 算了**显示宽度**，
  新增的表就是齐的。

## 三、验证

* `tools/launcher.py` 仍然只依赖标准库（`json` / `unicodedata` / `typing`
  都是标准库，守卫测试新增前后都通过）。
* 真实环境端到端跑过：`start.bat doctor` → **4 个引擎全部可用**，退出码 0；
  `start.bat engines` → 第 4 步打印「全部可用：传统 CV 引擎 / 深度学习引擎 /
  检测识别引擎 / 三合一引擎」，随后正常启动。
* 四种降级场景用**假项目根**验证（不动真实 `models/`）：无 yolo 权重 /
  有权重但缺 `onnxruntime` / 运行库全缺 / `cv` 的 `svm.dat` 丢失 ——
  判定与修复命令都符合预期。
* 新增 **33 项**测试（`tests/test_launch_env.py` 31 → 64 项），全量 pytest
  **1110 项**全绿。

## 四、仍然没做

* **没有 Windows 之外的启动脚本**。`start.sh`（Linux / macOS 一键启动）这次没做 ——
  用户只要求补模型自检。`tools/launcher.py` 本身是跨平台的（`venv_python()`
  已经区分 `Scripts/` 与 `bin/`），缺的只是一个 shell 引导壳。
* **不做「缺权重就自动下载」**。权重有许可与哈希校验的要求，
  自动下载会把「从哪来、可不可信」这件事藏起来。现在的做法是给出确切命令，
  让用户显式执行 `tools/fetch_yolo_weights.py`。
* 判定只看默认权重路径（`models/`）。用 `engine.yolo_models_dir` 把权重放别处时，
  体检仍会报「不可用」。这不是启动器独有的 —— `yolo_engine.is_available()` 这个
  类方法（`lpr engines` 也走它）同样只认默认目录，是注册表视角的固有边界。

---

# v2.7.2 —— 准确率提升（补救预处理）与界面美化

两个目标一次做完：识别侧把「有框无字」这类失败多救回一些；界面侧把
`UIConfig.theme` 这个**声明了却从未实现**的字段真正落地，并补上几个
日常使用里缺的东西。

## 一、准确率：补救预处理（换增强方式，而不是换尺度）

### 1.1 从诊断开始，而不是从"加功能"开始

上一版 [ACCURACY.md](ACCURACY.md) 已经把 `yolo` 的 16 张失败拆成了
8 张「有框无字」+ 7 张形近字 + 1 张严重错读。这次先写了一个诊断脚本，
把 8 张「有框无字」的透视矫正图分别用 7 种预处理（原图 / 外扩 8% /
CLAHE / CLAHE+3x / 直方图均衡 / OTSU / 去噪+3x）送进**同一个**识别器：

* `0808_019.jpg`（GT `皖AUF027`）：7 种里 6 种读成 `皖ANF027` 或空，
  只有「非局部均值去噪 + 3x 放大」读对 —— **噪点把 U 的底部糊成了 N**，
  这是"高置信度地读错"，把同一张图放大 2/3/4 倍再锐化并不改变它；
* `0139_058.jpg`（GT `皖AAF725`）：OTSU 二值化后读出 `皖AF725`
  （仍丢 1 字，但方向对了）；
* 其余 6 张：全部无效（blur/challenge 子集，人眼也难认）。

结论：主识别计划的 6 轮只换**尺度**（`_UPSCALE_STEPS × _SHARPEN_LEVELS`），
从不换**增强方式**。v2.7.2 据此新增 `_rescue_preprocess`：
主计划全部落空时，依次用「去噪 / CLAHE / OTSU」各配一档 3x 放大重试。

### 1.2 三条设计决定（都有实测背书）

1. **触发条件是"该候选一个合法读数都没有"** —— 与第二识别模型的
   交叉验证同一条火线，对已命中路径零回归（有专门的测试钉住：
   主计划有产出时，`_rescue_preprocess` 一次都不许被调用）。
2. **每图预算 `_RESCUE_BUDGET = 2`**。第一版没有预算，deep 的候选池
   （上限 5 个）让单图中位耗时从 716ms 涨到 1344ms 而**命中一张没多** ——
   触发补救的多是那些本来就该读不出的垃圾候选。卡到 2 之后收益不变、
   误检不变，耗时只涨三成（716 → 946ms）。
3. **二值化变体用最近邻放大**。三次插值会在笔画边缘拉出灰过渡，
   把"只有黑白"破坏掉 —— 这是被自己的测试抓出来的
   （断言 otsu 输出只有两档灰度时实测多出 80/175 两档）。

### 1.3 实测（100 张 CCPD，同一份代码开/关该路径）

| 配置 | 严格命中 | 字符级 | 有框无字 | 误检 | 中位ms |
|---|---|---|---|---|---|
| yolo 改前 | 84.0% | 90.7% | 8 | 7 | 125 |
| **yolo 改后** | **85.0%** | **93.6%** | **4** | 9 | 109 |
| deep 改后 | 79.0% | 87.9% | 0 | 14 | 946（+32%） |

诚实声明：**净收益是 +1 张**，不是 8 张 —— 8 张里 1 张救回、2 张从
"空白"变成"读错了"、4 张仍读不出。误检 7 → 9 是多读一次的固有代价。
详见 [ACCURACY.md §5.3](ACCURACY.md)。

## 二、界面：把声明过却没做的事做掉

* **双主题落地**：`UIConfig.theme` 字段从 v2.4.x 起就躺在配置里，
  但界面永远浅色 —— 没有任何代码读过它。现在 `_THEMES` 里浅色/深色
  两套令牌共用同一个 `build_qss()`，加一套主题 = 加一个 dict；
  视图菜单可切换（Ctrl+1/2），选择写回配置，下次启动沿用。
  卡片投影、斑马纹、菜单、滚动条、tooltip 全部跟着令牌走；
  车牌展示卡刻意**不**跟主题走（国标配色是物理属性，蓝牌到哪都是蓝的）。
* **候选列表**：此前一张图检出多块车牌时，界面上只显示最高分的那块，
  其余只写进日志。现在结果区多一张候选表（号码/颜色/置信度），
  点行回显 —— 与历史行点击共用同一段渲染代码（此前那两处是两份
  几乎相同的拷贝，改一处漏一处）。
* **拖拽打开**：图片或文件夹直接拖进窗口即开始识别；
  `dragEnterEvent` 只判断"有本地路径"，不预读目录（它会被高频调用）。
* **一键复制号码**：识别结果旁一个小复制按钮（Ctrl+Shift+C），
  识别之后最常见的下一步就是把号码粘到别处去。
* **窗口缩放后图片跟随重绘**：`imageArea` 刻意没开 `scaledContents`
  （自己拉伸会糊），代价是缩放后 pixmap 不跟 —— 补一个去抖 200ms 的重绘。
* **顺带修**：`UIConfig.window_title` 默认值里的「v2.3.9」—— 版本号
  已在状态栏显示（`__version__`），写进标题的版本号必然漂移（此处停了
  四个版本）；ROI 缩略图的硬编码浅色底改成由主题令牌控制，深色主题下
  不再刺眼。

## 三、验证

* 新增 **16 项**测试（`tests/test_rescue_preprocess.py`：纯函数契约 +
  接线零回归 + 预算），相关既有套件全绿。
* 两套主题用离屏渲染各截一张图人工核验（布局/配色/车牌皮肤正常；
  离屏环境缺中文字体，文字显示为方块是环境问题不是代码问题）。
* 全量 pytest、ruff、mypy 全绿（计数见 README）。

---

# v2.7.3 —— 运行日志落盘，以及一个被日志揪出来的测试脆弱点

用户一句话反馈：「没有运行日志」。查下去发现是**两个**独立的问题，
而且第二个比第一个严重 —— 界面里那个「运行日志」面板**从来没工作过**。

## 一、运行日志：从"只打 stderr"到"三个出口"

### 1.1 原来的样子

`setup_logging()` 只装两个 handler：`StreamHandler(stderr)` 和一个给界面用的
内存环形缓冲。于是：

* **程序一关，日志就没了**。用户报障时手上没有任何证据 —— 而"权重缺失导致
  yolo 不可用"这类问题恰恰需要看历史日志才知道发生过；
* **界面「运行日志」面板永远空白**。`setup_logging` 只在 `cli.py` 里被调过
  （而且只有 `predict` / `video` 两条路径），**GUI 入口 `run_ui()` 从没调过它**。
  后果不只是面板空：`logger.propagate` 也一直是默认的 `True`，日志顺着根
  logger 走，INFO 级被直接丢掉（只有 WARNING 以上会经 `lastResort` 打到 stderr）；
* **多进程子进程的日志完全丢失**。`batch.py` 用 `spawn` 起子进程，新解释器上
  没有 handler，子进程里 `logger.error("子进程初始化失败: ...")` 这类信息
  没人接。

### 1.2 现在

`src/lpr/utils/logging_setup.py` 增加文件出口，并补齐三个入口的接线：

* **文件出口**：`logs/lpr-YYYYMMDD.log`，`RotatingFileHandler` 按 `log_max_mb`
  （默认 5MB）轮转、保留 `log_backups`（默认 5）份；
* **幂等语义明确**：`setup_logging` 第二次调用直接返回（此前会**重设 level**，
  于是 `main()` 用 `--log-level DEBUG` 配好之后，`cmd_predict` 里那句
  `setup_logging(cfg.log_level)` 会把它悄悄改回配置文件里的 INFO）；
  要重建用 `force=True`，`close_logging()` 负责关句柄（测试收尾与运行期换日志用）；
* **写不进去就降级**：目录只读 / 磁盘满 / 被占用 → 打一行警告、只保留控制台，
  **绝不抛异常**。真正需要日志的时刻，恰恰是环境已经不正常的时候；
* **第一行告诉你日志在哪**：`运行日志 → <绝对路径>`。用 `absolute()` 而不是
  `resolve()` —— 后者在 Windows 上会返回 `C:/Users/HONGLI~1/...` 这种 8.3 短名；
* **子进程按 pid 分文件**：`-p<pid>.log`。不是洁癖：多个进程轮流轮转同一个
  文件时，Windows 上重命名一个仍被别的进程打开的文件会抛 `PermissionError`，
  跑量大的批次里会随机炸；顺带还解决了"哪个进程读了哪张图"。

### 1.3 配置与命令行

`config.json` 顶层新增 `log_to_file` / `log_dir` / `log_max_mb` / `log_backups`，
并新增 `Config.logging_kwargs()` 统一转成 `setup_logging` 的入参（三个入口共用，
避免各自记一遍字段名）。手改 JSON 的宽松解析：`"log_to_file": "false"` 当假、
`log_max_mb: 0` 夹到 1、`log_backups: "abc"` 保持默认 —— 日志设置不值得让配置加载失败。

新增 `--log-level` / `--log-dir` / `--no-log-file`，**同时挂在主命令和每个子命令上**：

```
lpr --log-level DEBUG predict a.jpg     # 选项写在子命令前
lpr predict a.jpg --log-level DEBUG     # 选项写在子命令后
```

两种写法都生效，靠的是子命令那份用 `argparse.SUPPRESS` 当默认值 —— 否则
子命令解析时会用 `None` 把主命令填好的值覆盖掉（argparse 的经典坑）。

### 1.4 界面

* `run_ui()` 里补上 `setup_logging`（就是上面那个"从没调过"的洞）；
* 「运行日志」面板**修掉一个真 bug**：原来刷新判据是"行数没变就不再重绘"，
  而缓冲只有最近 20 行 —— 攒够 20 行之后行数恒定，面板**永久冻住**。
  改成用内容快照判重，并把显示上限提到 200 行；
* 面板下面增加一行**日志文件路径**（可选中复制）与「打开日志目录」按钮；
  `_open_directory()` 在打不开文件管理器时只降级为一句说明（把路径显示出来），
  不抛异常。

### 1.5 启动器

`start.bat` 的第 5 步会打印日志目录。目录是从 `config.json` 的 `log_dir` 读的，
读不出来才回落到默认 `logs/`（与 `Config.log_dir` 的默认值有测试交叉校验）；
配置里关掉了文件日志就**什么都不打印** —— 指一个不存在的地方比不说更糟。

## 二、顺手修掉：回归基线测试的"按位置配对"

跑全量时 `tests/test_regression_baseline.py` 红了 6 条，一度怀疑是这次改动
动了定位。逐张按**路径**复算后证明是虚惊：11 张基线样张的 IoU 全是 **1.000**，
候选数与底色一字不差。

真正的原因是测试自己：三条"逐张对照"用的是

```python
for base, now in zip(baseline["records"], current["records"], strict=False):
```

**按位置**配对。只要 `img/` 下多出文件，就变成"图 A 的基线"比"图 B 的现状"，
于是报出一屏 IoU≈0 的假漂移，把真信号彻底埋掉。本次环境里 `img/` 多出了
192 张本地图片（见下），正好踩中。

改成按路径索引配对，并让 `test_locate_still_finds_candidates` 只查基线覆盖过的
那批样张（`img/` 里后来多的随手拍车流照定位不到车牌不算回归）。
`test_baseline_covers_repo_samples` 这条**故意保持会失败** —— 它是"样张清单
变了就该重新生成基线"的绊线，只把报错信息改成直接列出多了/缺了哪些。

> 附带发现：`img/` 目录里当时有 202 个文件（其中 192 个不在基线内、也不在 git 里），
> 总体积 61.4MB，超过了"样张总体积 < 40MB"这条 CI 预算。数据是否保留由使用者决定，
> 这里只保证**测试能说清问题**，不擅自删文件。
> → 使用者的决定与随后的处理见下面的「四」，那两条红已按"保留入库、改门槛"解决。

## 三、验证

* 新增 **60 项**测试（`tests/test_logging_setup.py`：落盘与轮转、写失败降级、
  幂等与 force 重建、级别解析、配置与命令行接线、脱敏贯通到文件、
  界面日志面板回归），`test_launch_env.py` +8、`test_batch_parallel.py` +1，
  总数 1126 → **1195**；
* `ruff` / `mypy` 全绿；全量 pytest 当时是 **1193 / 1195** 通过 —— 剩的 2 条红是
  `img/` 被塞了 192 张本地图导致的（样张清单与 40MB 预算），属**数据问题**，
  不是代码问题：11 张基线样张的定位行为逐张复算 IoU 全为 1.000
  （→ 这 2 条已在「四」里解决，现在全量 **1195 / 1195**）；
* 实机抓了一段真实日志贴进 README §3.8（含 `--workers 2` 产生的子进程文件）。

## 四、v2.7.3 之后的收尾（同一天）

v2.7.3 推送后连着发现两件事，都与"运行日志"无关，但都出在同一次推送之后。

### 4.1 CI 上一处真实的红：界面用例缺 libEGL

5 个 ubuntu 作业全红，而 Windows 作业通过，失败点全在
`tests/test_logging_setup.py::TestGuiLogPanel`：

```
ERROR  ...::test_panel_keeps_refreshing_after_it_is_full
ERROR  ...::test_panel_shows_the_log_file_path
ERROR  ...::test_panel_says_so_when_file_logging_is_off
FAILED ...::test_open_directory_failure_is_reported_not_raised
E  ImportError: libEGL.so.1: cannot open shared object file
```

原因是 `pytest.importorskip("PySide6")` **拦不住**这种失败：PySide6 顶层 import
并不加载 Qt 平台库，`QtWidgets` 才加载 —— importorskip 放行，下一行照样炸。
本机 Windows 上 Qt 库齐全，完全复现不出来；静态检查作业（ruff + mypy）反而是绿的，
所以光看作业名很容易误判成"CI 环境问题"。

两条一起修：

* **测试侧**：`qt_app` 改成 `pytest.importorskip("PySide6.QtWidgets")`，环境缺平台库时
  整组用例 **skip 而不是 error**；`_open_directory` 那条也挂上 `qt_app`（它 import
  `lpr.ui.window`，同样依赖平台库）；
* **CI 侧**：Linux 作业补装 `libegl1 / libgl1 / libfontconfig1 / libdbus-1-3`，
  让这些用例在 Linux 上**真跑**而不是静默跳过。`ci.yml` 的注释本来就写着"Qt 用例在
  无显示环境下跑（CI 没有 X）"，但此前**没有任何用例会建 `QApplication`**，
  这个前提从未被验证过。

修完 ubuntu 作业的通过数 1187 → **1189**（那 4 条从 error/failed 变成真跑并通过）。

### 4.2 样张目录治理：按使用者决定"保留入库、改门槛"

推送当天 `img/` 又进了三批下载样本（17:18、17:26 各约 100 张，18:25 再 200 张
`sample_*.jpg`）。命名来自 `tools/fetch_ccpd.py` —— 它本身下到 `datasets/ccpd_raw`
（已 gitignore），之后被搬进了 `img/`。使用者的决定是**全部保留在仓库、按现实改门槛**：

* `tests/baselines/real_samples.json` 重新生成：11 条 → **403 条**
  （`img/` 401 + `samples/` 2；`img/` 里的 `A1_Z拖链.3mf` 不是图片，被按扩展名滤掉）；
* 体积预算 `40MB → 150MB`，并提成常量 `SAMPLE_SIZE_BUDGET_MB`、把取舍写进注释：
  它现在只是**失控增长护栏**，不再是"体积小 / checkout 很快"的承诺；
* 文档同步：README §3.7 / §13.4 / §15.4、ACCURACY §7.3、REFERENCE_REVIEW §5、
  ROADMAP 第 4 项 —— "11 张真样张"改成"403 张（**带人工真值的仍只有 11 张**）"，
  并如实记下"`img/` 同时是下载落图目录"这个**根因未除**的事实与三条根治思路。
* 判据也跟着改对了一处：原来的 `test_locate_still_finds_candidates` 断言"基线里
  每张样张都能定位到候选"，那是 11 张精选样张时代的前提 —— 403 张里有 2 张
  （`img/0315_067.jpg`、`img/sample_1095.jpg`）cv 引擎本来就给不出候选。
  已改成 `test_candidate_presence_matches_baseline`：**逐张比较"有候选/没候选"是否
  与基线一致**。既不再误判那 2 张，又顺带补上了 `test_top_bbox_not_drifted` 的缺口
  （那条在"基线有框、现在没框"时会 `continue` 跳过）。

执行时实测：`img/` 402 个文件 / 68.8MB，连同 `samples/` 共 **75.7MB**
（`datasets/ccpd_raw` 里另躺着 654 个未入库文件）。

### 4.3 推送与发布

| 提交 | 内容 |
|---|---|
| `2cf1076` | v2.7.3 运行日志落盘（本文档主体） |
| `37788b2` | `img/` 纳入 192 张本地样张 + 1 个 `.3mf` |
| `9b1c49b` | `img/` 再纳入 200 张 CCPD 样张 |
| `1bf7273` | 修 CI：界面用例在 Ubuntu 上缺 libEGL |
| 最后一个 | 样张治理：基线 11 → 403 条、体积预算 40 → 150MB、文档同步 |

* tag 与 Release：`v2.7.3` 已打并推送，`v2.7.2` 此前只有提交、漏了 tag 与 Release，
  本次一并补齐（两个 Release 都建了）。
* 注意 **`v2.7.3` tag 指向 `2cf1076`**，不含之后那三个"测试 / 数据 / CI"性质的提交 ——
  发出去的**代码**与 tag 内容一致，后面动的只有测试、CI 配置与样张数据。

---

# v2.7.4 —— 界面把默认引擎悄悄换掉了（`cascade` → `cv`）

这次是从界面「运行日志」面板里的一条记录查起的：

```
19:53:05 [INFO] lpr.ui.window: 配置的默认引擎 cascade 不可用，界面实际使用 cv
```

先给结论：这句话**两处都错**，而且它掩盖了一次真实的准确率退化。

## 一、这句话错在哪

### 1.1 `cascade` 不是"不可用"，它是 `deep` 的别名

`config.ENGINE_CHOICES` 允许五个取值：`cv` / `deep` / `cascade` / `yolo` / `fusion`。
但引擎注册表（`engines/registry.py` 的 `_REGISTRY`）里**只有四个真引擎**：

```
已注册引擎: ['cv', 'deep', 'fusion', 'yolo']
get_engine("cascade") -> KeyError: 未知引擎 'cascade'，可用: cv, deep, fusion, yolo
```

`cascade` 是"用哪个引擎"的**规则**（`Recognizer._resolve_engine` 把它解析成
`deep`，因为 deep 内部本就是"CV 定位 → 纯识别网络"的级联），不是一类实现。
所以它**不该**被注册 —— 真注册了，`FusionEngine` 的成员过滤
（`NON_MEMBER_NAMES = {fusion, cascade, auto}`）会把别名当成可投票成员。

问题在于：界面下拉框是用 `describe_engines()` 填的，而它遍历的正是 `_REGISTRY`，
于是 `cascade` **不是不可用，是压根不在选项里**。而 `_refresh_engine_list` 找默认项的
写法是：

```python
elif info["name"] == self.config.engine.default:   # config 写的是 "cascade"
    default_idx = i                                 # 永远匹配不到
```

匹配不到 → `default_idx` 保持初始值 `0` → 选中第一项 `cv`。接着还顺手把配置改了：

```python
if selected != self.config.engine.default:
    logger.info("配置的默认引擎 %s 不可用，界面实际使用 %s", ...)
    self.config.engine.default = selected      # cascade → cv，静默
```

### 1.2 代价：11 张人工真值样张的命中从 10/11 掉到 0/11

用 `tests/baselines/ground_truth.json`（唯一带人工真值的 11 张）实测三种引擎：

| 引擎 | 解析成 | 命中 | 加载耗时 |
|---|---|---|---|
| `cv` | `cv` | **0 / 11** | 159ms |
| `cascade` | `deep` | **10 / 11** | 1ms |
| `deep` | `deep` | 10 / 11 | 2ms |

也就是说：配置里写着 `cascade`（= deep），界面每次启动都悄悄换成 `cv`，
**准确率从 90% 变成 0%**。逐张看更直观（真值 vs 界面实际读出的）：

```
img/single_blue.jpg     真值=辽DU4356    cv=渝J1151Z      cascade=辽DU4356
img/police.jpg          真值=鄂F8888警   cv=赣EE888Z      cascade=鄂F8888警
img/xue.jpg             真值=琼A6716学   cv=川J117SS      cascade=琼A6716学
samples/2.jpg           真值=京AD77972   cv=黑J11111      cascade=京AD77972
```

`cv` 的 SVM 对任何输入都输出置信度 1.0，读数全是"合法但不对"的号码 ——
这正是 [ACCURACY.md](ACCURACY.md) 里记录过的过自信问题，一个**看起来像正常工作**的
错误答案，比报错更难发现。

这条日志是 `INFO` 级、只在界面里出现、且被后续的正常运行掩盖，所以从 v2.3.1
（`cascade` 被定为默认引擎）起一直存在，直到有人去翻运行日志才暴露。

## 二、改了什么

### 2.1 单一事实来源：`config.CASCADE_RESOLVE_ORDER`

```python
CASCADE_RESOLVE_ORDER = ("deep", "cv")
```

`api.Recognizer._resolve_engine` 里 `auto` 与 `cascade` 两个分支原先各写一份
`("deep", "cv")`，现在都读这个常量。界面算 `cascade` 可用性时读的也是它 ——
"界面说会跑谁"与"运行时真的跑谁"从此不可能分叉。

（别与 `api.CASCADE_ORDER = ("cv", "deep")` 混淆：那是 `engine.mode="cascade"`
时挑**兜底**引擎的顺序，与"`cascade` 这个引擎名等于谁"是两件事，注释里写明了。）

### 2.2 `describe_engine_choices()`：真引擎 + 策略别名

```python
ALIAS_ENGINES = {
    "cascade": {
        "display_name": "级联引擎",
        "description": "CV 快速定位 → 深度精确识别（等同 deep 引擎）",
        "resolve_order": CASCADE_RESOLVE_ORDER,
        "anchor": CASCADE_RESOLVE_ORDER[0],   # 紧跟 deep，而不是按字母序落地
    },
}
```

`describe_engine_choices()` 在 `describe_engines()` 的结果里插入别名行（`strategy: True`），
可用性取自它解析出的引擎。三处行为：

| 情况 | `available` | `description` |
|---|---|---|
| deep 可用 | `True` | 原文 |
| deep 不可用、cv 可用 | `True` | 追加「；deep 不可用，本次退化为 CV」 |
| 两者都不可用 | `False` | `reason` 里两个都点名 |

第二条尤其重要：界面提示语直接读 `description`，不写明退化就会让用户以为跑的仍是 deep。

### 2.3 界面

* 下拉框改用 `describe_engine_choices()`，`cascade` 成为可选中项，位置紧跟「深度学习引擎」；
* **不再静默改写配置**：选中项与配置一致时不碰 `config.engine.default`；
* 真遇到不可用引擎（例如 `yolo` 没放权重）仍会回落，但日志从 `INFO` 升为 `WARNING`，
  内容写明**原因**与**回落目标**，并注明 `config.json 未改动`；
* 状态栏改显示中文展示名（「引擎：级联引擎」），此前显示内部名 `cascade`。

### 2.4 顺手修掉的一个隐藏 bug

写测试时发现：`default_idx` 初始值是硬编码的 `0`，而**第 0 项自己也可能不可用**
（且已被置灰）。`QComboBox.setCurrentIndex` 并不理会单项的 enabled 状态，
于是"回落"会把一个禁用项选成当前项 —— 界面看着选好了，实际跑不了。

现在兜底索引取**第一个可用项**：

```python
default_idx = next((i for i, info in enumerate(infos) if info["available"]), 0)
```

### 2.5 `lpr engines` 也能回答这个问题了

原先它只列真引擎，"我在 config 里写的 `cascade` 到底行不行"没人回答。现在：

```
传统 CV 引擎         可用       ...
深度学习引擎           可用       ...
级联引擎*            可用       CV 快速定位 → 深度精确识别（等同 deep 引擎）
三合一引擎            可用       ...
检测识别引擎           可用       ...

* 策略名：非独立引擎实现，运行时按说明里的顺序解析成具体引擎。
当前 config.json 的 engine.default = cascade → 可用
```

## 三、顺手修掉的：主题/引擎选择根本没落盘

查这条日志时发现 `on_theme_changed` 的 docstring 与 README §6.1 都写着"选择写回
`config.json`，下次启动沿用"，但代码只改了 `self.config.ui.theme`（内存）——
实测切到深色后 `config.json` 一个字节都没变，重开界面又回到浅色。`Config.save()`
此前只有 CLI 的 `config` 子命令在调。

补上一个 `_persist_config(what)`，写失败只降级为警告（与日志落盘同一个取舍：
为了记住一个偏好把界面弄崩不值得）。同时**刻意区分两条路径**：

* **用户主动切换**（点主题、点引擎下拉框）→ 写回 `config.json`；
* **自动回落**（配置里的引擎不可用）→ **只改内存，绝不落盘**。

后者是这次修复的核心：原先那个"静默改写"如果发生在主题落盘之后，就会把用户的
`cascade` **永久**改成 `cv` —— 从"本次运行跑错"升级成"以后每次都跑错"。

## 四、验证

* 新增 `tests/test_engine_choices.py` **22 项**，分四组：
  * `TestAliasCatalog`（9）：别名仍不进注册表（`get_engine("cascade")` 仍 `KeyError`）、
    位置紧跟 deep、可用性跟随解析结果、退化时说明写明、界面用的顺序与
    `Recognizer._resolve_engine` 一致、别名表引用的是同一个常量；
  * `TestGuiEngineCombo`（7）：**核心回归** —— 配置写 `cascade` 时下拉框选中它
    且配置不被改写、不再打假警报、状态栏显示中文名、切引擎后状态栏跟进、
    真不可用时 warning 里含原因与退回目标、回落目标必须是可用项；
  * `TestCliEngines`（2）：`lpr engines` 含别名与脚注、能报出当前默认值；
  * `TestConfigPersistence`（3）：主题与引擎选择**确实写回文件**、自动回落
    **一个字都不写**。
* 写测试时踩到一个坑值得记：界面开始落盘后，`test_switching_engine_updates_statusbar`
  一次 `setCurrentIndex` 就把**仓库的** `config.json` 重写了（引擎被改成 `deep`、
  文件里多出 40 行默认值）。于是把"把 `Config.save()` 目标挪到临时目录"提到
  **fixture 层**（`config_path`），所有建窗口的用例自动隔离，而不是逐个记得加。
* 全局静态检查：`ruff check` / `ruff format --check` / `mypy` 全绿。
* 实机验证：界面下拉框 5 项、默认选中 `cascade`、`config.engine.default` 保持
  `cascade` 不变；`lpr engines` 输出如上；`lpr predict` 走 `cascade` 结果不变。

