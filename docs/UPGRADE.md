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
