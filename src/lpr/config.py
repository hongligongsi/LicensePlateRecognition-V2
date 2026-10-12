"""集中配置。

原项目把参数放在 ``config.js``（其实是 JSON）里，并且是"前两套硬编码参数
二选一"的模式，改参数必须重启进程。这里改成：

1. 参数来自 dataclass，有默认值、有类型检查、有文档；
2. 支持从 ``config.json`` 覆盖，也支持环境变量覆盖；
3. 参数按**相对比例**而非绝对像素表达，从而对分辨率免疫
   （原项目最大的痛点就是换个像素的车牌就定位不到）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

#: 项目根目录（``lpr/config.py`` 往上三级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass
class LocateConfig:
    """车牌定位相关参数。

    原版 ``config.js`` 里的 ``morphologyr/col_num_limit/row_num_limit`` 等
    都是**写死的像素值**，只在 1000px 宽、车牌约占画面 1/8 的图上有效。
    这里改成比例，使同一套参数能在 640px 到 4000px 的图上通用。
    """

    #: 送进算法前把长边缩放到不超过该值，兼顾速度与精度
    max_side: int = 1200
    #: 高斯模糊核（奇数，0 表示跳过）
    blur_kernel: int = 3
    #: 开运算核大小，用于削弱非车牌纹理。
    #:
    #: 量纲是**相对图宽的比例**（``0.02`` = 图宽的 2%），不是千分比/万分比。
    #: 实际计算见 ``locate.py``：``round(图像宽 * open_kernel_ratio)``。
    open_kernel_ratio: float = 0.02
    #: 闭/开运算核 (高, 宽)。
    #:
    #: 量纲同样是**相对图高 / 图宽的比例**（``(0.005, 0.021)`` 表示
    #: 核高 = 图高的 0.5%、核宽 = 图宽的 2.1%），不是千分比。
    #: 实际计算见 ``locate.py``：``round(图高 * [0])`` × ``round(图宽 * [1])``。
    morph_kernel_ratio: tuple[float, float] = (0.005, 0.021)
    #: Canny 双阈值
    canny_low: int = 100
    canny_high: int = 200
    #: 候选矩形的最小面积（占全图面积比例）
    min_area_ratio: float = 0.0015
    #: 车牌候选的长宽比范围
    min_aspect: float = 2.0
    max_aspect: float = 5.5
    #: 颜色判定：某颜色像素占比超过该值即认定为此色。
    #:
    #: 注意它**不再**是"绝对过半"门槛（v2.3 起）：真实车牌上字符占了大量
    #: 面积，底色占比通常只有 10%~30%，用 0.5 做门槛会导致大量车牌被判成
    #: 未知。现在它只作为**自适应下限的基准**，取值范围在 (0, 1]。
    color_ratio: float = 0.5
    #: 颜色判定：胜出颜色与第二名的最小**占比差**，用于"相对胜出"判定。
    #:
    #: 调大 → 更难判出彩色（更多回落到黑白/未知），适合误判率优先；
    #: 调小 → 更容易判出彩色，适合召回优先。默认 0.04。
    color_margin: float = 0.04
    #: 精定位时某行/列颜色像素占比阈值
    color_row_ratio: float = 0.8
    #: 绿牌渐变严重，横向阈值单独放宽
    color_row_ratio_green: float = 0.5
    #: 是否启用「危险品运输车辆」牌识别
    #:
    #: 危险品运输车悬挂的是**黑底黄字**牌，与普通黄牌（黄底黑字）底色相反。
    #: 仅凭底面颜色无法区分两者，必须借助字符与底色的**明暗反转**关系判定，
    #: 因此需先做二值化极性分析。关闭可省去一次小图统计（默认开启，开销极低）。
    detect_hazmat: bool = True
    #: 危险品牌判定：主体颜色为黑（V 低）的像素占比下限
    hazmat_bg_ratio: float = 0.45
    #: 危险品牌判定：黄色前景（文字）像素占比下限
    hazmat_fg_ratio: float = 0.03
    #: 危险品牌判定：黄色前景占比较「黑底与黄字之和」的下限
    #: （即黄字在黑底上足够醒目，而非大片黄底上零星黑点）
    hazmat_fg_share: float = 0.06
    #: 候选矩形外扩像素，防止车牌边缘被裁掉
    box_padding: int = 5
    #: 字符分割后要求的最少字符数（含省份简称共 7 位，留出容错）
    min_chars: int = 6
    #: 最多保留多少个候选车牌进入字符识别
    max_candidates: int = 5

    def validate(self) -> None:
        if self.blur_kernel and self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel 必须是奇数或 0")
        if not 0 < self.color_ratio <= 1:
            raise ValueError("color_ratio 必须在 (0, 1]")
        if self.min_aspect >= self.max_aspect:
            raise ValueError("min_aspect 必须小于 max_aspect")
        if self.min_chars < 1:
            raise ValueError("min_chars 必须为正")


@dataclass
class EngineConfig:
    """引擎选择与运行方式。

    ``default`` 与 ``mode`` 是两个**正交**的维度，不要混淆：

    * ``default``：用哪个引擎*对象*。``"cascade"`` 会解析成 ``deep``
      （deep 引擎内部本就"CV 定位 → 纯识别网络"，这就是级联的含义）。
    * ``mode``：**跨引擎兜底**。``"cascade"`` 时，主引擎返回空结果会自动再
      尝试另一个可用引擎（例如 deep 完全没命中时让 cv 兜一次），
      ``"single"`` 则只跑主引擎。

    因此 ``default="cascade"`` + ``mode="single"`` 并不矛盾：前者选引擎对象，
    后者决定要不要跨引擎兜底。
    """

    #: 默认引擎："cv"（传统，快但精度有限）| "deep"（深度学习，推荐）| "cascade"（= deep，含 CV 定位引导）
    default: str = "cascade"
    #: 跨引擎兜底策略："single"（只跑主引擎）| "cascade"（主引擎未命中时再试另一个可用引擎）
    mode: str = "single"
    #: 传统 CV 引擎参数
    locate: LocateConfig = field(default_factory=LocateConfig)
    #: 字符识别时是否启用 SVM 置信度加权投票
    use_confidence: bool = True

    def validate(self) -> None:
        if self.default not in ("cv", "deep", "cascade"):
            raise ValueError(f"未知引擎: {self.default}")
        if self.mode not in ("single", "cascade"):
            raise ValueError(f"未知模式: {self.mode}")
        self.locate.validate()


@dataclass
class UIConfig:
    """界面相关。"""

    window_title: str = "车牌识别系统 v2.3.9"
    window_size: tuple[int, int] = (1280, 800)
    #: 摄像头识别间隔（秒），原版固定 2 秒，这里可调
    camera_interval: float = 0.35
    #: 摄像头采集分辨率
    camera_resolution: tuple[int, int] = (1280, 720)
    theme: str = "light"
    max_history: int = 200


@dataclass
class SecurityConfig:
    """安全与资源限制（对应 ``utils.security.Limits`` / ``PrivacyPolicy``）。"""

    # ---- 输入限制
    max_file_mb: float = 32.0
    max_pixels: int = 50_000_000
    max_side: int = 20_000
    max_batch_files: int = 5000

    # ---- 隐私
    #: 输出中是否对车牌号打码
    mask_output: bool = False
    #: 打码保留的尾部字符数
    mask_keep_tail: int = 3
    #: 日志中是否自动脱敏车牌号（强烈建议 True）
    mask_logs: bool = True
    #: 结果中是否保留车牌 ROI 图像
    keep_roi: bool = True

    # ---- 性能
    #: 单张识别超时（秒），0 为不限制
    predict_timeout: float = 30.0
    #: 每秒最多识别多少次（限流，防止并发打满 CPU）
    max_qps: float = 20.0

    # ---- 模型完整性
    #: 模型哈希白名单 ``{文件名: sha256}``。
    #:
    #: 默认内置随仓库分发的两个 SVM 模型的哈希，因此 ``lpr models --verify``
    #: 开箱即可真正做完整性校验（此前该白名单无处配置，校验恒为"跳过"）。
    #: 换成自训练模型后需同步更新这里，否则会被判为不匹配。
    model_hashes: dict[str, str] = field(
        default_factory=lambda: {
            "svm.dat": "ce3fdffb3e9529fceaacdd3aeb091ce23161f24510c4cc32ffa47708dcda47c5",
            "svmchinese.dat": ("dd3982d022008bf163dc0b02719a4a4b454a5fb8093653c13fc3d28fc67f41d8"),
        }
    )

    def to_limits(self):
        """转成 ``utils.security.Limits``。"""
        from .utils.security import Limits

        return Limits(
            max_file_bytes=int(self.max_file_mb * 1024 * 1024),
            max_pixels=self.max_pixels,
            max_side=self.max_side,
            max_batch_files=self.max_batch_files,
            predict_timeout=self.predict_timeout,
        )

    def to_privacy(self):
        """转成 ``utils.security.PrivacyPolicy``。"""
        from .utils.security import PrivacyPolicy

        return PrivacyPolicy(
            mask_output=self.mask_output,
            mask_keep_tail=self.mask_keep_tail,
            mask_logs=self.mask_logs,
            keep_roi=self.keep_roi,
        )

    def to_context(self):
        """一次性转成 ``utils.security.SecurityContext``（含模型哈希白名单）。"""
        from .utils.security import SecurityContext

        return SecurityContext(
            limits=self.to_limits(),
            privacy=self.to_privacy(),
            model_hashes=dict(self.model_hashes),
            install_log_masking=self.mask_logs,
        )


@dataclass
class Config:
    """顶层配置。"""

    engine: EngineConfig = field(default_factory=EngineConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    #: 日志级别
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """从 JSON 文件加载配置，文件不存在则使用默认值。

        支持点号路径的浅层覆盖，例如 ``{"engine": {"locate": {"max_side": 1600}}}``。
        """
        cfg = cls()
        path = Path(path) if path else DEFAULT_CONFIG_PATH

        if path.exists():
            with path.open(encoding="utf-8") as fp:
                raw = json.load(fp)
            _apply_overrides(cfg.engine, raw.get("engine", {}))
            _apply_overrides(cfg.ui, raw.get("ui", {}))
            _apply_overrides(cfg.security, raw.get("security", {}))
            if "log_level" in raw:
                cfg.log_level = str(raw["log_level"])

        # 环境变量优先级最高，便于临时调试
        if env_engine := os.getenv("LPR_ENGINE"):
            cfg.engine.default = env_engine
        if env_level := os.getenv("LPR_LOG_LEVEL"):
            cfg.log_level = env_level
        if os.getenv("LPR_MASK_OUTPUT", "").lower() in ("1", "true", "yes"):
            cfg.security.mask_output = True

        cfg.engine.validate()
        cfg.security.to_limits().validate()
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        """把当前配置写回 JSON 文件。"""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        payload = self.to_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        return path

    def to_payload(self) -> dict:
        """转成可 JSON / pickle 传递的纯字典。

        多进程批量识别要把配置送进子进程，这里避免直接把 dataclass 传过去
        （子进程里 dataclass 的构造参数可能与父进程不一致）。
        """
        return {
            "engine": _to_plain(self.engine),
            "ui": _to_plain(self.ui),
            "security": _to_plain(self.security),
            "log_level": self.log_level,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> Config:
        """从 :meth:`to_payload` 的结果还原配置。"""
        cfg = cls()
        if payload:
            _apply_overrides(cfg.engine, payload.get("engine", {}))
            _apply_overrides(cfg.ui, payload.get("ui", {}))
            _apply_overrides(cfg.security, payload.get("security", {}))
            if "log_level" in payload:
                cfg.log_level = str(payload["log_level"])
        cfg.engine.validate()
        cfg.security.to_limits().validate()
        return cfg


def _to_plain(obj: Any) -> Any:
    """把 dataclass 转成纯 dict/list/tuple。"""
    if hasattr(obj, "__dataclass_fields__"):
        out = {}
        for f in fields(obj):
            out[f.name] = _to_plain(getattr(obj, f.name))
        return out
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _apply_overrides(target: Any, overrides: dict) -> None:
    """把 overrides 中认识的字段写进 dataclass 实例，忽略未知键。"""
    known = {f.name: f for f in fields(target)}
    for key, value in overrides.items():
        if key not in known:
            continue
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            ftype = known[key].type
            setattr(target, key, _coerce(value, current, ftype))


def _coerce(value: Any, current: Any, ftype: Any) -> Any:
    """尽量把 JSON 里的 list 还原成 tuple，把数字还原成 int/float。"""
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "Config",
    "EngineConfig",
    "LocateConfig",
    "SecurityConfig",
    "UIConfig",
    "asdict",
]
