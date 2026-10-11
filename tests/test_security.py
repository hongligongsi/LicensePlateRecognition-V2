"""安全与隐私模块测试。

覆盖三个方面：
1. 输入限制（大小 / 像素 / 尺寸 / 解压炸弹）
2. 隐私脱敏（打码、递归清洗、日志过滤、路径匿名化）
3. 完整性校验与路径安全
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from lpr.utils.security import (
    Limits,
    MaskingFilter,
    PrivacyPolicy,
    RateLimiter,
    SecurityError,
    anonymize_path,
    check_file,
    check_image,
    mask_plate,
    safe_output_path,
    scrub_value,
    sha256_file,
    verify_model,
)


# ══════════════════════════════════════════════════════ 输入限制


def test_limits_validate_ok() -> None:
    Limits().validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_file_bytes", 0),
        ("max_pixels", 0),
        ("max_side", 1),
    ],
)
def test_limits_validate_rejects_bad_values(field: str, value: int) -> None:
    limits = Limits(**{field: value})
    with pytest.raises(ValueError):
        limits.validate()


def test_check_file_rejects_missing(tmp_path) -> None:
    with pytest.raises(SecurityError, match="不可访问"):
        check_file(tmp_path / "nope.jpg")


def test_check_file_rejects_empty(tmp_path) -> None:
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    with pytest.raises(SecurityError, match="为空"):
        check_file(p)


def test_check_file_rejects_oversize(tmp_path) -> None:
    p = tmp_path / "big.jpg"
    p.write_bytes(b"x" * 2048)
    with pytest.raises(SecurityError, match="过大"):
        check_file(p, Limits(max_file_bytes=1024))


def test_check_file_rejects_bad_extension(tmp_path) -> None:
    p = tmp_path / "evil.exe"
    p.write_bytes(b"payload")
    with pytest.raises(SecurityError, match="扩展名"):
        check_file(p)


def test_check_file_accepts_normal_image(tmp_path) -> None:
    p = tmp_path / "ok.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 100)
    resolved = check_file(p)
    assert resolved.is_file()


def test_check_image_rejects_non_array() -> None:
    with pytest.raises(SecurityError):
        check_image(None)  # type: ignore[arg-type]


def test_check_image_rejects_tiny_image() -> None:
    with pytest.raises(SecurityError, match="小于下限"):
        check_image(np.zeros((4, 4, 3), dtype=np.uint8))


def test_check_image_rejects_huge_pixel_count() -> None:
    # 不真的分配内存：用 reshape 构造元数据，check_image 只读 shape
    big = np.broadcast_to(np.uint8(0), (200, 200, 3))
    with pytest.raises(SecurityError, match="像素"):
        check_image(big, Limits(max_pixels=1000))


def test_check_image_rejects_lopsided_side() -> None:
    """1 x 50000 这类畸形尺寸应被单边上限拦住。"""
    lopsided = np.broadcast_to(np.uint8(0), (4, 50_000, 3))
    with pytest.raises(SecurityError, match="单边"):
        check_image(lopsided, Limits(max_side=1000, max_pixels=10**9))


def test_check_image_accepts_normal() -> None:
    check_image(np.zeros((480, 640, 3), dtype=np.uint8))


# ══════════════════════════════════════════════════════ 限流


def test_rate_limiter_allows_burst_then_throttles() -> None:
    limiter = RateLimiter(max_per_second=5.0)
    granted = sum(limiter.acquire(timeout=0.0) for _ in range(20))
    assert 1 <= granted <= 6, f"突发放行数异常: {granted}"


def test_rate_limiter_does_not_block_forever() -> None:
    limiter = RateLimiter(max_per_second=1.0)
    for _ in range(5):
        limiter.acquire(timeout=0.0)
    # 桶已空，超时 0 应立即返回 False 而不是挂住
    assert limiter.acquire(timeout=0.0) is False


# ══════════════════════════════════════════════════════ 脱敏


def test_mask_plate_disabled_by_default() -> None:
    assert mask_plate("京AD77972") == "京AD77972"


def test_mask_plate_respects_policy() -> None:
    policy = PrivacyPolicy(mask_output=True)
    masked = mask_plate("京AD77972", policy)
    assert masked != "京AD77972"
    assert masked.startswith("京")
    assert masked.endswith("972")


def test_mask_plate_keep_tail_configurable() -> None:
    policy = PrivacyPolicy(mask_output=True, mask_keep_tail=1)
    masked = mask_plate("京AD77972", policy)
    assert masked.endswith("2")
    assert masked[:-1].count("*") >= 5


def test_mask_plate_short_strings_pass_through() -> None:
    policy = PrivacyPolicy(mask_output=True)
    for short in ("", "京", "京A", "京A1"):
        assert mask_plate(short, policy) == short


def test_mask_plate_no_tail_when_disabled() -> None:
    policy = PrivacyPolicy(mask_output=True, mask_keep_tail=0)
    masked = mask_plate("京AD77972", policy)
    assert masked == "京" + "*" * 7


# ------------------------------------------------ scrub_value 递归清洗


def test_scrub_value_passthrough_when_disabled() -> None:
    policy = PrivacyPolicy(mask_output=False)
    payload = {"text": "京AD77972"}
    assert scrub_value(payload, policy) == payload


def test_scrub_value_scrubs_plain_string() -> None:
    policy = PrivacyPolicy(mask_output=True)
    assert "D779" not in scrub_value("京AD77972", policy)


def test_scrub_value_reaches_nested_structures() -> None:
    """核心回归：meta.variants 里的明文号码必须被打掉。"""
    policy = PrivacyPolicy(mask_output=True)
    payload = {
        "text": "京AD77972",
        "meta": {
            "variants": [["京AD77972", 1.2], ["京AD77973", 0.4]],
            "nested": {"deep": ["prefix 京AD77972 suffix"]},
        },
    }
    scrubbed = scrub_value(payload, policy)

    flat = repr(scrubbed)
    assert "京AD77972" not in flat, f"嵌套结构泄露明文: {flat}"
    assert "京AD77973" not in flat, f"嵌套结构泄露明文: {flat}"
    assert scrubbed["meta"]["variants"][0][1] == 1.2, "非字符串值不应被改动"


def test_scrub_value_preserves_tuple_type() -> None:
    policy = PrivacyPolicy(mask_output=True)
    out = scrub_value(("京AD77972", 1.0), policy)
    assert isinstance(out, tuple)
    assert out[1] == 1.0


def test_scrub_value_keeps_non_string_scalars() -> None:
    policy = PrivacyPolicy(mask_output=True)
    for value in (1, 1.5, True, None):
        assert scrub_value(value, policy) == value


def test_scrub_value_scans_long_text_embeddings() -> None:
    policy = PrivacyPolicy(mask_output=True)
    text = "检测到车辆 京AD77972，另有 沪B12345 驶入。"
    out = scrub_value(text, policy)
    assert "京AD77972" not in out
    assert "沪B12345" not in out
    assert "检测到车辆" in out, "非车牌内容应保留"


# ------------------------------------------------ 日志过滤


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, "f", 1, msg, None, None)


def test_masking_filter_masks_plate_in_log() -> None:
    filt = MaskingFilter()
    record = _record("识别到京AD77972")
    assert filt.filter(record) is True
    assert "京AD77972" not in record.getMessage()
    assert "*" in record.getMessage()


def test_masking_filter_keeps_other_text() -> None:
    filt = MaskingFilter()
    record = _record("引擎就绪，耗时 123ms")
    filt.filter(record)
    assert record.getMessage() == "引擎就绪，耗时 123ms"


# ------------------------------------------------ 路径匿名化


def test_anonymize_path_keeps_tail() -> None:
    out = anonymize_path("/home/user/secret/data/plate.jpg", keep=2)
    assert out.endswith("data/plate.jpg") or out.endswith("data\\plate.jpg")
    assert "secret" not in out


def test_anonymize_path_short_path_unchanged() -> None:
    assert anonymize_path("plate.jpg", keep=2) == "plate.jpg"


# ══════════════════════════════════════════════════════ 完整性校验


def test_sha256_file_deterministic(tmp_path) -> None:
    p = tmp_path / "m.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == sha256_file(p)
    assert len(sha256_file(p)) == 64


def test_verify_model_missing_file(tmp_path) -> None:
    ok, msg = verify_model(tmp_path / "nope.dat", None)
    assert not ok
    assert "不存在" in msg


def test_verify_model_without_whitelist_skips(tmp_path) -> None:
    p = tmp_path / "m.dat"
    p.write_bytes(b"data")
    ok, msg = verify_model(p, None)
    assert ok
    assert "跳过" in msg


def test_verify_model_detects_tampering(tmp_path) -> None:
    p = tmp_path / "m.dat"
    p.write_bytes(b"original")
    good = sha256_file(p)
    p.write_bytes(b"tampered")
    ok, msg = verify_model(p, good)
    assert not ok
    assert "不匹配" in msg


def test_verify_model_accepts_matching_hash(tmp_path) -> None:
    p = tmp_path / "m.dat"
    p.write_bytes(b"model-weights")
    ok, msg = verify_model(p, sha256_file(p).upper())
    assert ok, msg
    assert "通过" in msg


# ══════════════════════════════════════════════════════ 路径安全


def test_safe_output_path_blocks_traversal(tmp_path) -> None:
    base = tmp_path / "out"
    base.mkdir()
    out = safe_output_path(base, "../../../etc/passwd")
    assert out.parent == base.resolve() or base.resolve() in out.parents
    assert "etc" not in out.parts


def test_safe_output_path_blocks_absolute(tmp_path) -> None:
    base = tmp_path / "out"
    base.mkdir()
    out = safe_output_path(base, "/etc/shadow")
    assert base.resolve() in out.parents


def test_safe_output_path_allows_normal_name(tmp_path) -> None:
    base = tmp_path / "out"
    base.mkdir()
    out = safe_output_path(base, "result.json")
    assert out.name == "result.json"
    assert out.parent == base.resolve()


def test_safe_output_path_strips_null_byte(tmp_path) -> None:
    base = tmp_path / "out"
    base.mkdir()
    out = safe_output_path(base, "ev\x00il.jpg")
    assert "\x00" not in str(out)
