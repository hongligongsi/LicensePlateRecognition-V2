"""车牌专用检测 + 识别引擎（ONNX，第三引擎）。

**定位**：补上「传统 CV 定位」的短板。``cv`` / ``deep`` 两个引擎的车牌*定位*
都依赖 OpenCV 形态学轮廓，对**侧倾、模糊、低对比度**车牌召回不足
（见 ``docs/REFERENCE_REVIEW.md`` 第 231 行：「侧倾车牌的定位召回是下一个短板」）。
本引擎用**可训练的检测网络**（YOLO-pose 思路：一次前向给出 bbox + 4 角点 +
单/双层类别）替代轮廓定位，角点直接供透视矫正使用。

**与 ``deep`` 的关系**：两者互补而非替代。
``deep`` = CV 定位引导 + 纯识别（快，依赖定位成功）；
``yolo`` = 学习式检测 + 识别（慢，但定位鲁棒）。

**为什么只依赖 onnxruntime**：不引入 torch。权重由用户按需下载或以
Release 附件形式提供，仓库不捆绑（见 ``docs/THIRD_PARTY.md`` 与 README 许可一节）。
这样安装体积保持在「几十 MB 的 onnxruntime」，而不是「2GB+ 的 PyTorch」。

**许可证提示**：若使用基于 Ultralytics YOLO 训练的权重，该权重受
**AGPL-3.0** 约束。本项目自身为 GPL-3.0；以此为第三引擎时请确认你的
使用场景是否满足 AGPL 的源码提供义务（尤其是「作为网络服务提供」的情形）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..types import PlateLayout, PlateResult
from .base import BaseEngine, EngineUnavailable

logger = logging.getLogger(__name__)

#: 本引擎默认查找权重的相对目录（相对项目根）
DEFAULT_MODEL_DIR = Path("models") / "yolo"

#: 检测模型默认文件名
DETECT_MODEL_NAME = "plate_det.onnx"
#: 识别模型默认文件名（可选；缺省时复用 RapidOCR 做识别）
RECOGNIZE_MODEL_NAME = "plate_rec.onnx"

#: 输入尺寸：YOLO 系惯用的 640；改这个必须与导出时的 imgsz 一致
DEFAULT_INPUT_SIZE = 640

#: 检测置信度阈值
DEFAULT_CONF_THRESHOLD = 0.25
#: NMS IoU 阈值
DEFAULT_IOU_THRESHOLD = 0.45


def _probe_onnxruntime() -> tuple[bool, str]:
    """检查 onnxruntime 是否可用（本引擎唯一的硬依赖）。"""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "未安装 onnxruntime: pip install onnxruntime"
    return True, ""


def locate_model_dir(models_dir: Path | None = None) -> Path:
    """定位权重目录。

    优先用调用方显式传入的 ``models_dir``；否则从本文件上溯到项目根，
    再拼 ``models/yolo``。
    """
    if models_dir is not None:
        return models_dir
    # src/lpr/engines/yolo_engine.py -> 上溯 4 层到项目根
    root = Path(__file__).resolve().parents[3]
    return root / DEFAULT_MODEL_DIR


@dataclass(frozen=True)
class Detection:
    """一次检测输出：车牌框、四角点、版式类别、置信度。"""

    #: 四角点，顺序为 左上 / 右上 / 右下 / 左下，形状 (4, 2)，float32
    corners: np.ndarray
    #: 轴对齐外接框 (x1, y1, x2, y2)
    box: tuple[float, float, float, float]
    #: 单层 / 双层
    layout: PlateLayout
    #: 检测置信度
    score: float

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class YoloEngine(BaseEngine):
    """车牌专用检测 + 识别引擎（ONNX 推理）。

    当前为**骨架实现**：接口、注册、可用性探测、权重定位与错误提示已完备；
    ``_infer`` 与后处理待接入具体权重后补全（见 ``docs`` 中的说明）。
    """

    name = "yolo"
    display_name = "检测识别引擎"
    description = (
        "车牌专用检测网络（bbox + 四角点 + 单/双层）+ 识别，"
        "对侧倾/模糊车牌的定位召回优于轮廓法；需自行下载 ONNX 权重"
    )

    def __init__(
        self,
        config: Any = None,
        models_dir: Path | None = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        providers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.models_dir = locate_model_dir(models_dir)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        #: None = 交给 onnxruntime 自动选（有 CUDA 会用 CUDAExecutionProvider）
        self.providers = providers

        self._detect_session: Any = None
        self._recognize_session: Any = None
        self._input_name = ""
        self._provider_used = ""

    # -------------------------------------------------------------- 路径

    @property
    def detect_model_path(self) -> Path:
        return self.models_dir / DETECT_MODEL_NAME

    @property
    def recognize_model_path(self) -> Path:
        return self.models_dir / RECOGNIZE_MODEL_NAME

    # -------------------------------------------------------------- 可用性

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """仅在 onnxruntime 与权重**都**具备时才算可用。

        权重缺失时给出确切的放置路径与下载指引 —— 静默不可用会让用户
        以为是代码问题（v2.3.5 在 start.bat 上吃过这个亏）。
        """
        ok, msg = _probe_onnxruntime()
        if not ok:
            return False, msg

        model = locate_model_dir() / DETECT_MODEL_NAME
        if not model.is_file():
            return False, (
                f"未找到检测模型 {model}；"
                '下载权重后放到该路径即可启用（安装 extras: pip install -e ".[yolo]"）'
            )
        return True, ""

    # -------------------------------------------------------------- 加载

    def load(self) -> None:
        ok, reason = self.is_available()
        if not ok:
            raise EngineUnavailable(reason)

        import onnxruntime as ort

        opts = ort.SessionOptions()
        #: 与 CI / 批量识别共存：限制线程数，避免与多进程批量互相抢核
        opts.intra_op_num_threads = 0  # 0 = 交给 onnxruntime 自行决定
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self.providers or ort.get_available_providers()
        try:
            self._detect_session = ort.InferenceSession(
                str(self.detect_model_path), sess_options=opts, providers=providers
            )
        except Exception as exc:
            raise EngineUnavailable(
                f"检测模型加载失败: {self.detect_model_path}（{type(exc).__name__}: {exc}）"
            ) from exc

        self._input_name = self._detect_session.get_inputs()[0].name
        self._provider_used = self._detect_session.get_providers()[0]

        # 识别模型可选：缺失时复用 deep 引擎的 RapidOCR 做字符识别
        self._recognize_session = None
        if self.recognize_model_path.is_file():
            try:
                self._recognize_session = ort.InferenceSession(
                    str(self.recognize_model_path), sess_options=opts, providers=providers
                )
            except Exception as exc:
                logger.warning(
                    "识别模型加载失败，将回退到 RapidOCR: %s（%s）", self.recognize_model_path, exc
                )

        logger.info(
            "检测识别引擎就绪 | 检测=%s | 识别=%s | 后端=%s",
            self.detect_model_path.name,
            self.recognize_model_path.name if self._recognize_session else "RapidOCR",
            self._provider_used,
        )
        self._ready = True

    def close(self) -> None:
        self._detect_session = None
        self._recognize_session = None
        self._ready = False

    # -------------------------------------------------------------- 推理

    def predict(self, image: np.ndarray | str | Path) -> list[PlateResult]:
        """识别一张图中的所有车牌。

        .. note::
           骨架实现：检测前处理与后处理尚未接入具体权重，故当前返回空列表。
           接入权重后，此处将依次完成
           前处理 → ``_infer`` → 解码 → NMS → 透视矫正 → 字符识别。
        """
        self.ensure_ready()

        img = self._to_bgr(image)
        if img is None:
            return []

        detections = self._detect(img)
        if not detections:
            logger.debug("未检测到车牌候选")
            return []

        return self._recognize_all(img, detections)

    @staticmethod
    def _to_bgr(image: np.ndarray | str | Path) -> np.ndarray | None:
        """统一成 BGR ndarray；读取失败返回 None（并留日志，不抛异常）。"""
        if isinstance(image, np.ndarray):
            return image
        from ..utils.image import imread

        img = imread(str(image))
        if img is None:
            logger.warning("图像读取失败: %s", image)
        return img

    def _detect(self, img: np.ndarray) -> list[Detection]:
        """跑检测网络并解码成 :class:`Detection` 列表。

        TODO(权重): 接入具体权重时补全：
        letterbox 预处理 → session.run → 解码 bbox/kpt/类别 → NMS。
        """
        raise NotImplementedError(
            "检测后处理尚未实现：需要先确定训练/导出时使用的输出张量布局"
            "（YOLO-pose 的 output0 形状随版本变化）。"
            "请在接入权重后实现本方法，并按 docs/THIRD_PARTY.md 登记权重来源与许可。"
        )

    def _recognize_all(self, img: np.ndarray, detections: list[Detection]) -> list[PlateResult]:
        """对每个检测框做透视矫正后识别字符。

        TODO(权重): 识别优先走 ``self._recognize_session``；无识别模型时
        回退到 ``deep`` 引擎的 RapidOCR（复用其 ``text_recognizer`` 快路径）。
        """
        raise NotImplementedError("识别后处理尚未实现：需先补全 _detect")


__all__ = [
    "DEFAULT_MODEL_DIR",
    "DETECT_MODEL_NAME",
    "RECOGNIZE_MODEL_NAME",
    "Detection",
    "YoloEngine",
    "locate_model_dir",
]
