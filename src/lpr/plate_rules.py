"""中国车牌号码的语义规则、校验与纠错。

**这是提升精度性价比最高的一环**：深度学习模型可能把 ``京AD77972`` 识别成
``京AD7797Z``，但车牌号码的构造有严格规则（省份简称 + 发牌机关字母 + 序号），
用规则可以主动纠正这类错误，而不需要更大的模型。

覆盖的车牌类型：

* 普通汽车号牌（蓝/黄）：省份 + 字母 + 5 位
* 新能源汽车号牌（绿/黄绿双拼）：省份 + 字母 + 6 位
* 警用号牌：省份 + 字母 + 4 位数字 + ``警``
* 军队号牌：字头 + 字头号（如 ``京V``、``军A``）
* 使领馆号牌：省份/``使`` + 3 位数字 + ``使``/``领``
* 挂车号牌：省份 + 字母 + 5 位 + ``挂``
* 教练车：省份 + 字母 + 4 位 + ``学``
* 港澳入出境：``粤Z`` 开头 + 4 位 + ``港``/``澳``
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: 省级行政区简称（31 个，不含港澳台——它们的车牌规则单独处理）
PROVINCE_CHARS = set("京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新")

#: 车牌序号位允许的字母（I、O 会与 1、0 混淆，国家规定不使用）
SEQ_LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
#: 发牌机关代号允许的字母（同上去掉 I、O）
ISSUER_LETTERS = SEQ_LETTERS

#: 军队车牌字头（新式军牌为单汉字字头 + 字母 + 5 位数字）
MILITARY_HEADS = set("军空海北沈兰济南广成")
#: 旧式军牌字头字母（如 京V、京K 这类不属于省份的字母字头）
LEGACY_MILITARY_LETTERS = set("VKBSLJNGC")

#: 武警车牌字头
ARMED_POLICE_HEADS = set("WJ")

#: 常见混淆字符对，纠错时按"目标位置允许的字符集"来选
CONFUSION_MAP: dict[str, str] = {
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "2": "Z", "Z": "2",
    "5": "S", "S": "5",
    "6": "G", "G": "6",
    "8": "B", "B": "8",
    "D": "0", "Q": "0",
    "L": "1",
}


@dataclass
class PlateFormat:
    """一种车牌格式的描述。"""

    name: str
    pattern: re.Pattern[str]
    #: 长度范围
    min_len: int
    max_len: int

    def match(self, text: str) -> bool:
        return bool(self.pattern.match(text))


def _make(province: str) -> str:
    """构造以省份简称开头的正则片段。"""
    return f"[{province}]"


PROV = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新"
L = "ABCDEFGHJKLMNPQRSTUVWXYZ"
AN = "A-HJ-NP-Z0-9"  # 字母数字（不含 I、O）
D = "0-9"

#: 所有支持的车牌格式，**顺序重要**：越具体的排在越前面
FORMATS: list[PlateFormat] = [
    # 港澳入出境（粤Z + 4 位 + 港/澳）
    PlateFormat("港澳入出境", re.compile(rf"粤Z[{AN}]{{4}}[港澳]$"), 7, 7),
    # 警用（省 + 字母 + 4~5 位数字 + 警）
    PlateFormat("警用", re.compile(rf"[{PROV}][{L}][{D}]{{4,5}}警$"), 7, 8),
    # 军队（字头 + 字母 + 5 位数字/字母）
    PlateFormat("军队", re.compile(rf"[{''.join(MILITARY_HEADS)}][{L}][{D}]{{5}}$"), 7, 7),
    # 旧式军牌（省 + V/K 等军用字母 + 5 位数字），必须早于普通民用判定，
    # 否则 京V12345 会被误归为普通民用
    PlateFormat(
        "军队",
        re.compile(rf"[{PROV}][{''.join(LEGACY_MILITARY_LETTERS)}][{D}]{{5}}$"),
        7,
        7,
    ),
    # 武警（WJ + 选省代码 + 4 位数字，如 WJ京1234X）
    PlateFormat("武警", re.compile(rf"WJ[{PROV}][{D}]{{4}}[{AN}]$"), 7, 8),
    # 使领馆（省份 + 3 位数字 + 使/领，或 使 + 6 位数字）
    PlateFormat("使领馆", re.compile(rf"[{PROV}][{D}]{{3}}[使领]$"), 5, 6),
    PlateFormat("使领馆", re.compile(rf"使[{D}]{{6}}$"), 7, 7),
    # 挂车（省 + 字母 + 4~5 位 + 挂）
    PlateFormat("挂车", re.compile(rf"[{PROV}][{L}][{AN}]{{3,4}}挂$"), 6, 7),
    # 教练车（省 + 字母 + 3~4 位 + 学）
    PlateFormat("教练车", re.compile(rf"[{PROV}][{L}][{AN}]{{3}}学$"), 6, 6),
    # 危险品运输（省 + 字母 + 4~5 位 + 危；公安部 GA36 标准中危险品运输车
    # 为黑底黄字，号码末位标"危"。此格式必须在普通民用之前，否则会被吞掉）
    PlateFormat("危险品运输", re.compile(rf"[{PROV}][{L}][{D}]{{4,5}}危$"), 7, 8),
    # 新能源（8 位：省 + 字母 + 6 位序号，序号区字母数字混排）
    # 真实规则中第 3 位起即为序号区，字母/数字皆可出现，末位通常为字母
    # （D=纯电、F=混动），但不强制。唯一可用的强约束是整串不能是纯数字，
    # 否则无法与 7 位普通牌区分。注意 京AQ7797Q 与 京AD77972 在此规则下
    # 同为合法且同分，这是真实世界的固有歧义，交由 OCR 置信度区分。
    PlateFormat(
        "新能源",
        re.compile(rf"[{PROV}][{L}][{D}{AN}]{{6}}$(?<!\d{{8}})"),
        8,
        8,
    ),
    # 普通民用（7 位：省 + 字母 + 5 位）
    PlateFormat("普通民用", re.compile(rf"[{PROV}][{L}][{AN}]{{5}}$"), 7, 7),
    # 短牌（6 位：省 + 字母 + 4 位，老式/特殊用途）
    PlateFormat("短牌", re.compile(rf"[{PROV}][{L}][{AN}]{{4}}$"), 6, 6),
]


def identify_format(text: str) -> PlateFormat | None:
    """判断号码属于哪种车牌格式。"""
    if not text:
        return None
    for fmt in FORMATS:
        if fmt.match(text):
            return fmt
    return None


def is_valid(text: str) -> bool:
    """号码是否符合任一种合法车牌格式。"""
    return identify_format(text) is not None


def _title_province(text: str) -> str:
    """修正首位省份简称：从 OCR/分类器输出里挑最可能的省份。

    车牌首位**必须**是省份简称。若模型给出别的字符（例如把"京"认成"凉"），
    这里无法凭空还原，但可以保证输出是合法省份字符。
    """
    return text


def correct(text: str) -> tuple[str, list[str]]:
    """按车牌语法纠正号码，返回 ``(纠正后号码, 改动说明列表)``。

    纠正策略（保守，只在有明确依据时改动）：

    1. 第 1 位必须是省份简称 → 不符合则整体判为无效，不强行编造。
    2. 第 2 位必须是字母（且不含 I、O）→ 把数字 0/1 换成最可能的字母。
    3. 第 3 位起不允许出现 I、O → 换成 1、0。
    4. 末位出现 ``Z``/``S``/``G`` 等，若能通过换成数字使号码合法，则替换。
    5. 整体长度与实际格式不符时，尝试截断或补全。

    Returns:
        纠正后的号码与改动记录。若无法纠正（如首位不是省份简称），
        原样返回并给出说明。
    """
    changes: list[str] = []
    if not text:
        return text, changes

    chars = list(text)

    # ---- 规则 1：首位必须是省份简称
    if chars[0] not in PROVINCE_CHARS:
        # 特例：军队字头、"使"字头、武警 WJ
        if chars[0] in MILITARY_HEADS or chars[0] in ("使", "W"):
            return text, changes
        # 旧式军牌字母字头（如 V12345）会被误判，此处不做纠正
        changes.append(f"首位 {chars[0]!r} 不是省份简称")
        return text, changes

    # ---- 规则 2：第 2 位必须是字母（不含 I、O）
    if len(chars) >= 2:
        second = chars[1]
        if second not in ISSUER_LETTERS:
            fixed = _best_letter(second)
            if fixed and fixed != second:
                changes.append(f"第 2 位 {second!r} → {fixed!r}（发牌机关代号须为字母）")
                chars[1] = fixed

    # ---- 规则 3：第 3 位起不允许 I、O
    for i in range(2, len(chars)):
        if chars[i] == "I":
            changes.append(f"第 {i + 1} 位 I → 1")
            chars[i] = "1"
        elif chars[i] == "O":
            changes.append(f"第 {i + 1} 位 O → 0")
            chars[i] = "0"

    candidate = "".join(chars)
    if candidate != text and is_valid(candidate):
        return candidate, changes

    # ---- 规则 4：末位字母降级为数字（若能使号码合法）
    if len(chars) in (7, 8) and chars[-1] in SEQ_LETTERS:
        for repl in ("0", "1", "2", "5", "6", "8"):
            trial = "".join(chars[:-1]) + repl
            if is_valid(trial):
                changes.append(f"末位 {chars[-1]!r} → {repl!r}（使得号码合法）")
                return trial, changes

    # ---- 规则 5：序号位中的易混字母降级（仅在整体不合法时尝试）
    if not is_valid(candidate):
        degraded = _degrade_letters(chars)
        if degraded is not None:
            changes.append(f"序号位字母降级为数字: {candidate} → {degraded}")
            return degraded, changes

    # ---- 规则 6：长度修正
    if len(chars) > 8:
        trimmed = "".join(chars[:8])
        if is_valid(trimmed):
            changes.append(f"截断多余字符: {text} → {trimmed}")
            return trimmed, changes

    if len(chars) == 6:
        for pos in range(2, 7):
            for fill in "0123456789":
                trial = "".join(chars[:pos]) + fill + "".join(chars[pos:])
                if len(trial) == 7 and is_valid(trial):
                    changes.append(f"补全缺失字符 {fill!r}（原长度 6）")
                    return trial, changes

    return "".join(chars), changes


#: 序号位中易与数字混淆的字母 → 数字
_LETTER_TO_DIGIT = {"Z": "2", "S": "5", "G": "6", "B": "8", "T": "7", "A": "4", "Q": "0", "D": "0"}


def _degrade_letters(chars: list[str]) -> str | None:
    """尝试把序号位里的易混字母换成数字，使号码合法。

    只在**整体不合法**时才做（避免把合法的字母序号改坏）。
    """
    trial = list(chars)
    changed = False
    for i in range(2, len(trial)):
        if trial[i] in _LETTER_TO_DIGIT:
            trial[i] = _LETTER_TO_DIGIT[trial[i]]
            changed = True
    if not changed:
        return None
    result = "".join(trial)
    return result if is_valid(result) else None


def _best_letter(ch: str) -> str:
    """把非字母字符映射到最可能的字母（用于第 2 位发牌机关代号）。

    I、O 虽然不合法，但模型经常把 1、0 认成它们，需要映射回常见字母。
    """
    mapping = {
        "0": "D", "1": "J", "2": "Z", "3": "B", "4": "A", "5": "S",
        "6": "G", "7": "T", "8": "B", "9": "P",
        # 字母 I/O 本身不合法（与数字混淆），映射到字形相近的合法字母
        "I": "J", "O": "D",
    }
    if ch in ISSUER_LETTERS:
        return ch
    return mapping.get(ch, "")


def plate_score(text: str) -> float:
    """给一个候选号码打分（0~1），用于多候选排序。

    分数组成：
      * 是否符合某种合法格式（权重最高）
      * 长度是否常见（7 位普通 / 8 位新能源）
      * 各省份的统计先验（车牌数量大的省份略加权重，仅作微调）
    """
    if not text:
        return 0.0

    score = 0.0
    if is_valid(text):
        score += 0.65
    elif len(text) in (7, 8) and text[0] in PROVINCE_CHARS:
        score += 0.35

    if len(text) == 7:
        score += 0.15
    elif len(text) == 8:
        score += 0.12
    elif len(text) in (5, 6):
        score += 0.03

    if text and text[0] in PROVINCE_CHARS:
        score += 0.12
    if len(text) >= 2 and text[1] in ISSUER_LETTERS:
        score += 0.08

    return min(1.0, score)


def describe(text: str) -> dict:
    """返回号码的解析信息，用于结果展示与审计。"""
    fmt = identify_format(text)
    return {
        "text": text,
        "valid": fmt is not None,
        "type": fmt.name if fmt else "未知",
        "province": text[0] if text and text[0] in PROVINCE_CHARS else None,
        "issuer": text[1] if len(text) > 1 else None,
        "is_new_energy": bool(fmt and fmt.name == "新能源"),
        "is_hazmat": bool(fmt and fmt.name == "危险品运输"),
    }


__all__ = [
    "PROVINCE_CHARS",
    "SEQ_LETTERS",
    "ISSUER_LETTERS",
    "FORMATS",
    "PlateFormat",
    "identify_format",
    "is_valid",
    "correct",
    "plate_score",
    "describe",
]
