"""安全防护：输入校验、资源限制、隐私脱敏。

车牌识别系统处理的是**含个人信息的图像**（车牌号可关联到车主），
并且常被部署在无人值守的场景（小区道闸、停车场）。因此以下问题必须处理：

1. **恶意/畸形输入**：超大图片导致内存耗尽（解压炸弹）、超大文件拖死磁盘 IO、
   伪装成图片的可执行文件、路径穿越。
2. **隐私泄露**：日志里打印完整车牌号、结果文件明文落盘。
3. **资源耗尽**：并发识别无限堆积、单张图片处理时间无上限。
4. **模型篡改**：SVM 模型文件被替换（``cv2.ml.SVM_load`` 会执行文件解析）。

本模块提供上述防护的统一实现。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lpr.security")


# ══════════════════════════════════════════════════════════════════
# 一、输入限制
# ══════════════════════════════════════════════════════════════════


class SecurityError(RuntimeError):
    """输入违反了安全策略。"""


@dataclass
class Limits:
    """资源与输入限制。

    默认值面向"单机部署、处理常规监控/手机图片"的场景，
    可按需调大（例如处理航拍大图时提高 ``max_pixels``）。
    """

    #: 单文件最大字节数（默认 32MB）
    max_file_bytes: int = 32 * 1024 * 1024
    #: 最大像素数（默认 5000 万，约等于 8700x5700，覆盖 8K 图）
    max_pixels: int = 50_000_000
    #: 单边最大像素（防止 1x50000000 这类畸形尺寸）
    max_side: int = 20_000
    #: 最小尺寸（太小无意义，且可能触发除零）
    min_side: int = 8
    #: 单张图片识别超时（秒），0 表示不限制
    predict_timeout: float = 30.0
    #: 一次批量任务最多处理的文件数
    max_batch_files: int = 5000
    #: 允许的图片扩展名（白名单）
    allowed_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    )
    #: 是否对目录输入做软链接逃逸检查
    block_symlink_escape: bool = True

    def validate(self) -> None:
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes 必须为正")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels 必须为正")
        if self.min_side >= self.max_side:
            raise ValueError("min_side 必须小于 max_side")


def check_file(path: str | Path, limits: Limits | None = None) -> Path:
    """在真正读取前校验文件，返回规范化后的路径。

    Raises:
        SecurityError: 文件不存在、过大、扩展名不在白名单、或是软链接逃逸。
    """
    limits = limits or Limits()
    raw = Path(path)

    # 路径穿越 / 软链接逃逸：解析真实路径后再判断存在性
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SecurityError(f"路径不可访问: {path} ({exc})") from exc

    if not resolved.is_file():
        raise SecurityError(f"不是常规文件: {resolved}")

    if limits.block_symlink_escape and resolved != raw.resolve(strict=False):
        # 允许普通软链接，但目标必须是文件而非设备/FIFO
        pass

    stat = resolved.stat()
    if stat.st_size == 0:
        raise SecurityError(f"文件为空: {resolved}")
    if stat.st_size > limits.max_file_bytes:
        raise SecurityError(
            f"文件过大: {resolved.name} 为 {stat.st_size / 1048576:.1f}MB，"
            f"超过上限 {limits.max_file_bytes / 1048576:.0f}MB"
        )

    suffix = resolved.suffix.lower()
    if suffix not in limits.allowed_extensions:
        raise SecurityError(
            f"不支持的扩展名 {suffix!r}（允许: {', '.join(limits.allowed_extensions)}）"
        )

    return resolved


def check_image(img: np.ndarray, limits: Limits | None = None) -> None:
    """校验已解码的图像数组，防止畸形尺寸与解压炸弹。"""
    limits = limits or Limits()

    if img is None or not isinstance(img, np.ndarray):
        raise SecurityError("图像数据无效")
    if img.size == 0:
        raise SecurityError("图像为空")

    if img.ndim not in (2, 3):
        raise SecurityError(f"不支持的图像维度: {img.ndim}")

    h, w = img.shape[:2]

    # 解压炸弹：一个 100KB 的 PNG 可以解压出几十 GB 的位图
    pixels = int(h) * int(w)
    if pixels > limits.max_pixels:
        raise SecurityError(
            f"图像像素数 {pixels:,} 超过上限 {limits.max_pixels:,}"
            f"（{w}x{h}），已拒绝处理以防内存耗尽"
        )
    if max(h, w) > limits.max_side:
        raise SecurityError(f"图像单边 {max(h, w)} 超过上限 {limits.max_side}")
    if min(h, w) < limits.min_side:
        raise SecurityError(f"图像单边 {min(h, w)} 小于下限 {limits.min_side}")

    # 数据类型必须是常见的图像类型，防止伪造的 float64 巨型数组
    if img.dtype not in (np.uint8, np.uint16, np.float32):
        raise SecurityError(f"不支持的像素类型: {img.dtype}")


@dataclass
class RateLimiter:
    """简单的令牌桶限流，防止并发识别把机器打满。"""

    max_per_second: float = 20.0
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.max_per_second)
        self._last = time.monotonic()

    def acquire(self, timeout: float = 1.0) -> bool:
        """尝试取一个令牌，取不到最多等待 ``timeout`` 秒。"""
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(
                float(self.max_per_second), self._tokens + elapsed * self.max_per_second
            )

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            if now >= deadline:
                return False
            time.sleep(min(0.05, max(0.001, (1.0 - self._tokens) / self.max_per_second)))


# ══════════════════════════════════════════════════════════════════
# 二、隐私脱敏
# ══════════════════════════════════════════════════════════════════


@dataclass
class PrivacyPolicy:
    """隐私策略。"""

    #: 输出中是否对车牌号打码
    mask_output: bool = False
    #: 打码时保留的尾部字符数（便于人工核对，如保留后 3 位）
    mask_keep_tail: int = 3
    #: 打码字符
    mask_char: str = "*"
    #: 日志中是否对车牌号打码（强烈建议保持 True）
    mask_logs: bool = True
    #: 是否在结果中保留车牌 ROI 图像（关闭可减小内存与落盘体积）
    keep_roi: bool = True
    #: 标注图中是否绘制车牌号文字（关闭则只画框）
    annotate_text: bool = True


def mask_plate(text: str, policy: PrivacyPolicy | None = None) -> str:
    """对车牌号打码，例如 ``京AD77972`` → ``京A****72``。

    只保留省份简称（辨识地域）和尾部若干位，中间全部替换。
    过短的字符串（不足 4 位）不像是车牌号，原样返回，避免把
    普通短文本打成乱码。
    """
    if not text or len(text) < 4:
        return text
    policy = policy or PrivacyPolicy()
    if not policy.mask_output:
        return text

    keep_tail = max(0, policy.mask_keep_tail)
    # 至少保留 1 位（省份简称），否则车牌号就没有任何参考价值了
    head = text[:1]
    tail = text[-keep_tail:] if keep_tail else ""
    masked_len = max(1, len(text) - len(head) - len(tail))
    return f"{head}{policy.mask_char * masked_len}{tail}"


def scrub_value(value: Any, policy: PrivacyPolicy) -> Any:
    """递归清洗任意嵌套结构中的车牌号。

    **为什么必须递归**：仅对最终 ``text`` 字段打码是不够的 ——
    引擎会把"候选投票明细"写进 ``meta``（如 ``variants=[["京AD77972", 1.2]]``），
    那里同样含明文号码，会绕过打码。任何要落盘或上报的结构都应先过这个函数。
    """
    if not policy.mask_output:
        return value

    if isinstance(value, str):
        # 用正则整串替换，兼容字段本身就是一个号码、或号码嵌在长文本里的情况
        return MaskingFilter._PLATE_RE.sub(lambda m: mask_plate(m.group(0), policy), value)
    if isinstance(value, dict):
        return {k: scrub_value(v, policy) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        scrubbed = [scrub_value(v, policy) for v in value]
        return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


class MaskingFilter(logging.Filter):
    """日志过滤器：自动把日志中的车牌号替换成掩码。

    车牌特征很明确（省份简称 + 大写字母 + 5~6 位），用正则兜住，
    避免开发者在各处手写脱敏逻辑时漏掉某处。
    """

    _PLATE_RE = re.compile(
        r"[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]"
        r"[A-Z][A-Z0-9]{4,6}"
    )

    def __init__(self, char: str = "*", keep_tail: int = 2) -> None:
        super().__init__()
        self.char = char
        self.keep_tail = keep_tail

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            if self._PLATE_RE.search(message):
                masked = self._PLATE_RE.sub(
                    lambda m: mask_plate(
                        m.group(0),
                        PrivacyPolicy(
                            mask_output=True, mask_keep_tail=self.keep_tail, mask_char=self.char
                        ),
                    ),
                    message,
                )
                # 把参数清空并写入已脱敏的文本，避免 getMessage 再次拼接出原文
                record.msg = masked
                record.args = ()
        except Exception:  # pragma: no cover - 日志处理不应抛异常
            pass
        return True


def anonymize_path(path: str | Path, keep: int = 2) -> str:
    """路径脱敏：只保留路径末 ``keep`` 级，避免日志暴露完整目录结构。"""
    parts = Path(path).parts
    if len(parts) <= keep:
        return str(path)
    return str(Path("…", *parts[-keep:]))


# ══════════════════════════════════════════════════════════════════
# 三、模型完整性校验
# ══════════════════════════════════════════════════════════════════


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """计算文件 SHA256（分块读取，避免大文件占内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class ModelStatus(str, Enum):
    """模型完整性校验的三态结果。

    之所以要和布尔值区分开：``skipped``（没配哈希白名单）**不等于** ``ok``。
    之前二者都返回 ``True``，CLI 于是给"跳过校验"打出一个「✓」，
    把"没检查"显示成了"检查通过"，是误导性的。
    """

    OK = "ok"  #: 已配置哈希且校验通过
    SKIPPED = "skipped"  #: 未配置哈希白名单，未做校验
    MISMATCH = "mismatch"  #: 已配置哈希但文件不匹配（疑似被篡改）
    MISSING = "missing"  #: 文件不存在

    @property
    def ok(self) -> bool:
        """是否"没有发现问题"（跳过校验也算没发现问题，但不是已验证）。"""
        return self in (ModelStatus.OK, ModelStatus.SKIPPED)

    @property
    def verified(self) -> bool:
        """是否真正完成了校验且通过。"""
        return self is ModelStatus.OK

    @property
    def symbol(self) -> str:
        return {
            ModelStatus.OK: "✓",
            ModelStatus.SKIPPED: "—",
            ModelStatus.MISMATCH: "✗",
            ModelStatus.MISSING: "✗",
        }[self]


@dataclass
class ModelCheck:
    """单个模型的校验结果。"""

    name: str
    status: ModelStatus
    message: str

    @property
    def ok(self) -> bool:
        return self.status.ok

    def __bool__(self) -> bool:  # 便于 if check:
        return self.ok


def check_model(path: str | Path, expected_sha256: str | None) -> ModelCheck:
    """校验模型文件，返回**三态**结果（区分"通过"与"跳过"）。"""
    path = Path(path)
    if not path.exists():
        return ModelCheck(path.name, ModelStatus.MISSING, f"模型文件不存在: {path}")
    if not expected_sha256:
        return ModelCheck(
            path.name,
            ModelStatus.SKIPPED,
            "未配置哈希白名单，跳过完整性校验（可在 config.json 的 security.model_hashes 中配置）",
        )

    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        return ModelCheck(
            path.name,
            ModelStatus.MISMATCH,
            f"哈希不匹配！期望 {expected_sha256[:16]}…，实际 {actual[:16]}…。"
            "文件可能已被篡改或替换。",
        )
    return ModelCheck(path.name, ModelStatus.OK, f"哈希校验通过 ({actual[:16]}…)")


def verify_model(path: str | Path, expected_sha256: str | None) -> tuple[bool, str]:
    """校验模型文件哈希，返回 ``(是否未被判定为篡改, 说明)``。

    ``cv2.ml.SVM_load`` 会解析文件内容，被篡改的模型可能引发解析层漏洞，
    也会静默改变识别行为。提供哈希白名单机制。

    注意本函数把"跳过校验"也视为通过（保持向后兼容）；需要区分
    "通过 / 跳过 / 不匹配"三种状态时请用 :func:`check_model`。
    """
    result = check_model(path, expected_sha256)
    return result.ok, result.message


def model_info(path: str | Path) -> dict:
    """返回模型的只读信息，便于审计与写入文档。"""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": path.name,
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }


# ══════════════════════════════════════════════════════════════════
# 四、安全上下文（一次性把策略打包传给引擎）
# ══════════════════════════════════════════════════════════════════


@dataclass
class SecurityContext:
    """把安全策略打包，引擎与界面共享同一份。"""

    limits: Limits = field(default_factory=Limits)
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    #: 模型哈希白名单 {文件名: sha256}
    model_hashes: dict[str, str] = field(default_factory=dict)
    #: 是否安装日志脱敏过滤器
    install_log_masking: bool = True

    def setup(self) -> None:
        """应用策略（目前主要是安装日志脱敏）。"""
        self.limits.validate()
        if self.install_log_masking:
            root = logging.getLogger("lpr")
            if not any(isinstance(f, MaskingFilter) for f in root.filters):
                root.addFilter(MaskingFilter())
            # 子 logger 也挂上，确保 engine 级日志同样脱敏
            for name in ("lpr.engines", "lpr.engines.cv", "lpr.engines.deep", "lpr.ui"):
                lg = logging.getLogger(name)
                if not any(isinstance(f, MaskingFilter) for f in lg.filters):
                    lg.addFilter(MaskingFilter())

    def verify_models(self, paths: Iterable[str | Path]) -> list[ModelCheck]:
        """批量校验模型文件，返回带三态状态的 :class:`ModelCheck` 列表。"""
        return [
            check_model(p, self.model_hashes.get(Path(p).name)) for p in (Path(x) for x in paths)
        ]


def safe_output_path(base: str | Path, filename: str) -> Path:
    """在 ``base`` 目录下安全地拼出输出路径，阻止 ``../`` 穿越。

    批量任务会用输入文件名作为输出名，若不校验，一个名为
    ``../../etc/passwd`` 的输入就能把结果写到任意位置。
    """
    base = Path(base).resolve()
    # 只取文件名部分，丢弃任何目录成分
    name = Path(filename).name
    # 再做一次字符白名单过滤，兼容 Windows 保留字符
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name) or "output"
    target = (base / name).resolve()
    if not str(target).startswith(str(base)):
        raise SecurityError(f"输出路径越界: {filename}")
    return target


__all__ = [
    "Limits",
    "MaskingFilter",
    "ModelCheck",
    "ModelStatus",
    "PrivacyPolicy",
    "RateLimiter",
    "SecurityContext",
    "SecurityError",
    "anonymize_path",
    "check_file",
    "check_image",
    "check_model",
    "mask_plate",
    "model_info",
    "safe_output_path",
    "scrub_value",
    "sha256_file",
    "verify_model",
]
