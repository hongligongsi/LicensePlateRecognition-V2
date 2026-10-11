# 车牌识别系统 v2（License-Plate-Recognition 升级版）

用 Python + OpenCV + 深度学习做的**中国车牌识别**，支持蓝牌 / 黄牌 / 绿牌（新能源）/ 白牌 / 黑牌。

本项目是 [hongligongsi/License-Plate-Recognition](https://github.com/hongligongsi/License-Plate-Recognition) 的现代化升级版。原项目最后一次提交停留在 2021 年，其依赖（Python 3.4 / OpenCV 3.4 / numpy 1.14）在如今的机器上已无法安装。v2 在保留原算法思想的基础上全面重构，详见 [UPGRADE.md](docs/UPGRADE.md)。

## 特性

- **双引擎可切换**
  - `cv` 传统 CV 引擎：OpenCV 形态学定位 + 投影分割 + SVM 字符识别（兼容原版模型），纯 CPU、毫秒级、零额外依赖
  - `deep` 深度学习引擎：基于 RapidOCR（ONNX）/ PaddleOCR，中文省份简称与模糊车牌的准确率远高于 SVM
- **现代化 GUI**（PySide6）：单图 / 批量 / 摄像头实时识别，识别不再冻结画面，历史记录可导出 CSV
- **命令行工具**：`lpr predict`、`lpr video`、`lpr gui`
- **参数自适应**：定位参数以比例表达，同一套配置适配 640px～4000px 输入（原版换分辨率就失效）
- **多车牌**：一张图里的多个车牌全部识别（原版只返回第一个）
- **置信度**：字符级 + 整体置信度，标注图可视化
- **中英文路径**：图片读写全兼容中文路径，自动纠正 EXIF 旋转

## 实测效果

| 样张 | 原版 SVM | v2 深度引擎 |
|---|---|---|
| `samples/1.jpg` 京E·51619（黄牌）| ✗ | ✓ 京E51619（另识别出背景 京PAP315）|
| `samples/2.jpg` 京A·D77972（绿牌）| ✗ | ✓ 京AD77972 |

（原版算法在 v2 流水线下可定位到车牌区域，但字符识别错误——训练样本不足是原项目自己承认的硬伤。）

## 安装

```bash
# 基础（传统 CV 引擎）
pip install -e .

# 推荐：+ 深度学习引擎 + 界面
pip install -e ".[deep,gui]"

# 可选：PaddleOCR 后端（更大更慢但部分场景更准）
pip install -e ".[deep-paddle]"
```

SVM 模型（`models/svm.dat`、`models/svmchinese.dat`）已随仓库附带，来自原项目（EasyPR 样本训练）。

## 使用

### 命令行

```bash
lpr engines                      # 查看引擎状态
lpr predict car.jpg --engine deep
lpr predict ./photos/ --engine deep --save-annotated   # 目录批量 + 输出标注图
lpr video parking.mp4 --stride 10                      # 视频识别
lpr gui                          # 图形界面
```

### Python API

```python
from lpr import get_engine, setup_logging

setup_logging("INFO")
engine = get_engine("deep")       # 或 "cv"
results = engine.predict("car.jpg")
for r in results:
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
├── config.py           # 集中配置（JSON + 环境变量，比例参数）
├── cli.py              # 命令行入口
├── engines/
│   ├── base.py         # 引擎抽象接口
│   ├── registry.py     # 引擎注册表（可扩展第三方引擎）
│   ├── locate.py       # 车牌定位（向量化、透视矫正、颜色判定）
│   ├── segment.py      # 字符分割（投影波峰 + 动态对齐到 7 字符）
│   ├── svm_classifier.py  # SVM 字符分类（兼容原版模型 + 置信度）
│   ├── cv_engine.py    # 传统 CV 引擎
│   └── deep_engine.py  # 深度学习引擎（RapidOCR / PaddleOCR）
├── ui/
│   ├── window.py       # PySide6 主窗口
│   └── worker.py       # 识别/摄像头线程
└── utils/              # 图像 IO、日志
models/                 # 预训练 SVM 模型（原版权重）
samples/                # 测试样张
tests/                  # pytest 单元 + 端到端测试
docs/UPGRADE.md         # 与原项目的详细对比
```

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 已知限制

- 传统 CV 引擎的 SVM 字符识别精度有限（原版模型训练样本太少，中文省尤其差）——需要精度请用 `deep` 引擎
- 深度引擎首次运行需下载 OCR 模型（约 10MB），之后离线可用
- 摩托车车牌、双行牌（挂车）暂不支持
- 军牌 / 警牌（白牌特殊字符）识别率一般

## 许可

MIT（与原项目一致）
