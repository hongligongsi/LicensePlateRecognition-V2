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
* 民航号牌：``民航`` + 管理局字母 + 4 位数字（如 ``民航A0001``）
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

#: 民航号牌的管理局代号字母（依据 CCAR-331SB-R1）
#: A=华北局 B=华东局 C=中南局 D=西南局 E=西北局 F=东北局 G=新疆局
CIVIL_AVIATION_HEADS = "ABCDEFG"

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

#: 省份简称的**形近字** → 正确省份。
#:
#: 这些字都不在省份集内，一旦 OCR 认错，整条号码就会被判为无效而丢弃。
#: 把字形/字音相近的常见错误收敛回来，是零成本提升召回的手段。
#:
#: 收录原则：只收**不会与其它省份简称混淆**的形近字。例如
#: ``"淅"→"浙"`` 是安全的（"淅"本身不是简称），但
#: ``"南"→"赣"`` 就不该收（"南"可能出现在非车牌文本里，容易误伤）。
PROVINCE_LOOKALIKES: dict[str, str] = {
    # 京
    "凉": "京", "享": "京", "亭": "京", "高": "京",
    # 沪
    "炉": "沪", "户": "沪", "护": "沪", "泸": "沪",
    # 津
    "律": "津", "建": "津", "津": "津",
    # 渝
    "愉": "渝", "瑜": "渝", "榆": "渝",
    # 冀 / 粤 / 鲁 / 豫
    "翼": "冀", "異": "冀",
    "奥": "粤", "粵": "粤", "専": "粤",
    "噜": "鲁", "兽": "鲁", "曾": "鲁",
    "预": "豫", "予": "豫", "務": "豫",
    # 浙 / 苏 / 皖 / 闽
    "浙": "浙", "淅": "浙", "折": "浙",
    "辦": "苏", "酥": "苏", "苏": "苏",
    "皖": "皖", "腕": "皖", "碗": "皖",
    "闻": "闽", "闵": "闽", "闩": "闽",
    # 赣 / 湘 / 鄂 / 桂
    "干": "赣", "赣": "赣", "贛": "赣",
    "相": "湘", "箱": "湘", "想": "湘",
    "卾": "鄂", "愕": "鄂", "鳄": "鄂",
    "挂": "桂", "贵": "桂", "柱": "桂",
    # 川 / 贵 / 云 / 藏
    "州": "川", "洲": "川", "川": "川",
    "潰": "贵", "桂": "贵", "遗": "贵",
    "雲": "云", "运": "云", "雲": "云",
    "藏": "藏", "臟": "藏", "藏": "藏",
    # 陕 / 甘 / 青 / 宁
    "陕": "陕", "狭": "陕", "峡": "陕",
    "甘": "甘", "廿": "甘", "泔": "甘",
    "清": "青", "精": "青", "情": "青",
    "宁": "宁", "宇": "宁", "甯": "宁",
    # 新 / 蒙 / 辽 / 吉
    "薪": "新", "亲": "新", "新": "新",
    "蒙": "蒙", "檬": "蒙", "艨": "蒙",
    "辽": "辽", "疗": "辽", "辽": "辽",
    "告": "吉", "洁": "吉", "结": "吉",
    # 黑 / 沈 / 济
    "墨": "黑", "嘿": "黑", "熏": "黑",
    "沉": "沈", "枕": "沈", "忱": "沈",
    "剂": "济", "挤": "济", "齐": "济",
}

#: 特殊号牌**末位后缀**的形近字 → 正确后缀（军/警/使领馆/挂车/教练等）。
#:
#: 省份简称有一整张形近字表，而 ``警/挂/学/危/使/领/港/澳`` 这些特种牌
#: 专属字符此前**没有任何纠错映射** —— OCR 把「警」认成「敬」、「领」认成
#: 「顿」时整条号码直接判无效。这是特种牌召回低的主要原因之一。
#:
#: 与省份表同样的收录原则：只收字形相近、且**不会与其他常用字冲突**的映射。
#: 该表只在**末位**使用（与 :data:`HEAD_LOOKALIKES` 按位置区分，同一个字
#: 在首位和末位的正确归宿不同，如「字」末位→「学」、首位→「军」），
#: 且替换后必须整体通过 :func:`is_valid` 才采纳，误伤面被严格闸门兜住。
SUFFIX_LOOKALIKES: dict[str, str] = {
    # 警（警用号牌末位，红字）
    "敬": "警", "擎": "警", "惊": "警", "謦": "警",
    # 挂（挂车号牌末位）——「桂」是省份简称，但出现在末位只可能是「挂」
    "桂": "挂", "圭": "挂", "佳": "挂",
    # 学（教练车号牌末位）
    "宇": "学", "字": "学", "觉": "学",
    # 危（危险品运输号牌末位）
    "免": "危", "色": "危", "负": "危",
    # 领（使领馆号牌末位）
    "顿": "领", "颁": "领", "频": "领", "颀": "领",
    # 使（使领馆号牌末位）
    "便": "使", "吏": "使", "更": "使",
    # 港 / 澳（港澳入出境号牌末位）——「奥」首位→「粤」已在省份表，末位→「澳」
    "巷": "港", "奥": "澳", "懊": "澳",
}

#: 军队/使领馆**字头**的形近字 → 正确字头（仅用于首位）。
#:
#: 军牌字头（军/空/海/北/沈/兰/济/南/广/成）与使领馆「使」都是低频字，
#: OCR 认错后整条号码失效。同样只在首位、且替换后整体合法才采纳。
HEAD_LOOKALIKES: dict[str, str] = {
    # 军
    "罕": "军", "牢": "军", "字": "军", "宇": "军",
    # 空
    "穹": "空", "究": "空",
    # 海
    "每": "海", "悔": "海", "梅": "海", "诲": "海",
    # 使
    "便": "使", "吏": "使", "更": "使",
    # 北 / 沈 / 兰 / 济（与省份形近字表部分重合，此处保证字头归宿一致）
    "兆": "北",
    "沉": "沈", "忱": "沈",
    "岚": "兰", "烂": "兰",
    "剂": "济", "挤": "济",
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
    # 民航号牌（"民航" + 管理局字母 + 4 位数字，如 民航A0001）。
    # 必须放在最前："民"不是省份简称、也不是军队/使领馆字头，现有格式都不会
    # 误吞它，但它开头的两个字是中文，任何以省份简称开头的规则都匹配不到，
    # 所以放在最前只是为了让匹配语义更清晰（越具体越靠前）。
    # 依据 CCAR-331SB-R1：绿底白字白框线，440x140mm，用于机场摆渡车、
    # 专用作业车、登机梯车等场内车辆。
    PlateFormat(
        "民航",
        re.compile(rf"民航[{CIVIL_AVIATION_HEADS}][{D}]{{4}}$"),
        6,
        6,
    ),
    # 港澳入出境（粤Z + 4 位 + 港/澳）
    PlateFormat("港澳入出境", re.compile(rf"粤Z[{AN}]{{4}}[港澳]$"), 7, 7),
    # 警用（省 + 字母 + 4~5 位数字 + 警）
    PlateFormat("警用", re.compile(rf"[{PROV}][{L}][{D}]{{4,5}}警$"), 7, 8),
    # 军队（字头 + 字母 + 5 位数字/字母）
    PlateFormat("军队", re.compile(rf"[{''.join(MILITARY_HEADS)}][{L}][{D}]{{5}}$"), 7, 7),
    # 旧式军牌（京V + 5 位数字），必须早于普通民用判定，
    # 否则 京V12345 会被误归为普通民用。
    # 只保留无歧义的"京V"：其它省份 + V/K 等字母的组合几乎都是民用车牌
    # （如 皖K = 六安、粤V = 揭阳），放宽会大面积误判。
    PlateFormat(
        "军队",
        re.compile(rf"京V[{D}]{{5}}$"),
        7,
        7,
    ),
    # 武警（WJ + 省份 + 4 位数字 + 可选尾位字母，如 WJ京1234X）。
    # 尾位字母改为可选：部分批次/旧牌末位无字母（WJ京1234），
    # 强制要求会让这类号码在严格层直接失效。
    PlateFormat("武警", re.compile(rf"WJ[{PROV}][{D}]{{4}}[{AN}]?$"), 7, 8),
    # 武警旧式（WJ + 2 位机构码 + 4~5 位序号，写作 WJ01-12345，
    # 清洗分隔符后为 WJ0112345 共 9 位）。新式武警第 3 位是省份汉字，
    # 与纯数字形态无重叠，不会互相误吞。
    PlateFormat("武警", re.compile(rf"WJ[{D}]{{6,7}}$"), 8, 9),
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


#: 宽松召回用的结构正则（**只校验整体骨架，不约束具体位**）。
#:
#: 借鉴参考实现 YOLO26-plate 的分层思路：它的 ``_NEW_ENERGY_LOOSE``
#: 明确注明"仅校验整体结构，不再强约束特定位字母"。原因是——
#: OCR 输出常有字母/数字混淆（``D``↔``0``、``B``↔``8``），若在**召回阶段**
#: 就套用严格规则，一个字符错位就会让整条号码被丢弃，等于白识别。
#:
#: 因此分成两层：
#:
#: * **召回层**（本正则）：只看「1 位汉字 + 1 位字母 + 3~7 位字母数字 + 可选中文后缀」，
#:   尽可能把候选捞出来；
#: * **校验层**（:data:`FORMATS`）：严格判定类型与合法性，用于打分与展示。
#:
#: 两层之间的差额由 :func:`correct` 尝试补齐（纠错后再走严格校验）。
#:
#: 注意首位用 ``[\u4e00-\u9fff]`` 而非省份集：因为省份形近字（凉/泸/律…）
#: 也要能在召回阶段被捞出来，否则形近字纠错根本没有机会执行。
LOOSE_HEAD_CLASS = rf"[{PROV}{''.join(PROVINCE_LOOKALIKES)}]"

LOOSE_STRUCTURE = re.compile(
    # 普通民用 / 新能源 / 各类后缀牌：汉字 + 字母 + 3~7 位字母数字 + 可选后缀
    rf"{LOOSE_HEAD_CLASS}[{L}][{AN}]{{3,7}}[警挂学危港澳使领]?"
    # 军队：汉字字头 + 字母 + 3~7 位
    rf"|[军空海北沈兰济南广成使][{L}][{AN}]{{3,7}}[警挂学危]?"
    # 旧式军牌：省份 + 军用字母 + 5 位数字
    rf"|[{PROV}][{''.join(sorted(LEGACY_MILITARY_LETTERS))}][{D}]{{5}}"
    # 武警：WJ + 省份 + 4 位 + 可选 1 位；或旧式 WJ + 纯数字
    rf"|WJ[{PROV}][{D}]{{4}}[{AN}]?|WJ[{D}]{{5,7}}"
    # 使领馆：纯数字 + 使，或 省份/使 开头
    rf"|使[{D}]{{4,6}}|[{PROV}][{D}]{{3}}[使领]"
    # 民航：民航 + 管理局字母 + 3~5 位数字
    rf"|民航[{CIVIL_AVIATION_HEADS}][{D}]{{3,5}}"
)


def loose_match(text: str, start: int = 0) -> str | None:
    """从文本中**宽松**抽取一段可能的车牌号（召回用）。

    只要求骨架正确，不要求逐位合法。这样即便 OCR 把 ``D`` 认成 ``0``，
    也不会在抽取阶段就丢掉整条号码 —— 留给 :func:`correct` 去修。

    Args:
        text: 已清洗（去分隔符、转大写）的文本。
        start: 从此位置开始搜索。默认 0；传 ``1`` 可跳过首位再找
            （用于"整段不是车牌，但后半段是"的场景）。

    Returns:
        匹配到的子串，或 ``None``。
    """
    if not text:
        return None
    m = LOOSE_STRUCTURE.search(text, start)
    return m.group(0) if m else None


def is_valid(text: str) -> bool:
    """号码是否符合任一种合法车牌格式（**严格校验层**）。"""
    return identify_format(text) is not None


def _title_province(text: str) -> str:
    """修正首位省份简称：从 OCR/分类器输出里挑最可能的省份。

    车牌首位**必须**是省份简称。若模型给出别的字符（例如把"京"认成"凉"），
    这里无法凭空还原，但可以保证输出是合法省份字符。
    """
    return text


def _fix_io_inplace(chars: list[str], changes: list[str], start: int = 2) -> None:
    """就地纠正 ``start`` 位之后的 I/O 混淆（I→1、O→0）。

    军队/使领馆/武警牌同样受此约束（序号位不出现 I、O），
    此前 :func:`correct` 对这类字头**早退**，导致这一规则从不执行。
    """
    for i in range(start, len(chars)):
        if chars[i] == "I":
            changes.append(f"第 {i + 1} 位 I → 1")
            chars[i] = "1"
        elif chars[i] == "O":
            changes.append(f"第 {i + 1} 位 O → 0")
            chars[i] = "0"


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

    # ---- 规则 -1：已经合法的号码一个字符都不动。
    # 此前没有这条短路，使领馆牌「京123使」（第 2 位合法地是数字）会被
    # "第 2 位必须字母"的省份规则强行改成 京J23使，把合法号码纠成非法。
    if is_valid(text):
        return text, changes

    chars = list(text)

    # ---- 规则 0：省份形近字纠正（OCR 最常见的一类错误）
    # "京"被认成"凉"/"享"、"沪"被认成"炉"、"津"被认成"律"…
    # 这些字在字符集之外，若不纠正则整条号码直接判无效、白白丢掉。
    # 注意：只在**首位且不在省份集**时才尝试，避免改坏本来正确的号码。
    if chars and chars[0] not in PROVINCE_CHARS:
        fixed_prov = PROVINCE_LOOKALIKES.get(chars[0])
        if fixed_prov is not None:
            changes.append(f"首位 {chars[0]!r} → {fixed_prov!r}（形近省份简称）")
            chars[0] = fixed_prov

    # ---- 规则 0b：军队/使领馆字头形近字纠正
    # 与省份表同理：军牌字头（军/空/海…）被认成"罕/穹/每"时整条号码失效。
    # 替换后必须整体合法才采纳（is_valid 是硬闸门），采纳即返回——
    # 否则后续"第 2 位必须字母"等省份专属规则会把数字位改坏
    # （使领馆牌第 2 位本来就是数字）。
    if chars and chars[0] not in PROVINCE_CHARS and len(chars) >= 2:
        fixed_head = HEAD_LOOKALIKES.get(chars[0])
        if fixed_head is not None:
            trial = fixed_head + "".join(chars[1:])
            if is_valid(trial):
                changes.append(f"字头 {chars[0]!r} → {fixed_head!r}（形近军队/使领馆字头）")
                return trial, changes

    # ---- 规则 0c：特殊**末位后缀**形近字纠正
    # "警"被认成"敬"、"挂"被认成"桂"、"领"被认成"顿"…
    # 同样只在末位使用、替换后整体合法才采纳、采纳即返回（理由同上）。
    if len(chars) >= 2:
        fixed_suf = SUFFIX_LOOKALIKES.get(chars[-1])
        if fixed_suf is not None:
            trial = "".join(chars[:-1]) + fixed_suf
            if is_valid(trial):
                changes.append(f"末位 {chars[-1]!r} → {fixed_suf!r}（形近特种牌后缀）")
                return trial, changes

    # ---- 规则 1：首位必须是省份简称
    if chars[0] not in PROVINCE_CHARS:
        # 特例：军队字头、"使"字头、武警 WJ、民航——跳过"省份专属"规则，
        # 但**仍执行序号位的通用纠正**（I/O 混淆对特种牌同样存在，
        # 此前在这里整体早退，军A1O345 里的 O 从来没被修过）。
        if chars[0] in MILITARY_HEADS or chars[0] in ("使", "W") or chars[0] == "民":
            _fix_io_inplace(chars, changes)
            return "".join(chars), changes
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
    _fix_io_inplace(chars, changes)

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
        # 中文 OCR 把字母 E 误读成汉字"三"非常常见（字形几乎相同），
        # 双层黄牌上牌（省份行）首当其冲，如 京E → 京三
        "三": "E",
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
    """返回号码的解析信息，用于结果展示与审计。

    除了格式名，还给出**该格式通常对应的版式**（``layout_hint``）。
    号码本身无法表达"单行/双层"（``京A12345`` 既是单行大车牌也可能是双层
    大型车后号牌），若调用方从图像上量出了版式，应以图像判定为准；
    这里的 hint 只在图像无法判定时作为兜底先验。
    """
    fmt = identify_format(text)
    name = fmt.name if fmt else "未知"
    return {
        "text": text,
        "valid": fmt is not None,
        "type": name,
        "province": text[0] if text and text[0] in PROVINCE_CHARS else None,
        "issuer": text[1] if len(text) > 1 else None,
        "is_new_energy": name == "新能源",
        "is_hazmat": name == "危险品运输",
        "is_civil_aviation": name == "民航",
        "layout_hint": DEFAULT_LAYOUT.get(name),
    }


#: 各格式**默认**的版式先验。真实世界存在例外（如大型新能源车牌为双行绿牌），
#: 因此这里只作为图像判定失败时的兜底值，不作为硬性结论。
DEFAULT_LAYOUT: dict[str, str] = {
    # 大型汽车的后号牌是双行（前号牌为单行，但识别时通常看到的是后牌）
    "挂车": "double",
    "危险品运输": "double",
    # 民航、使领馆、警用、教练车、军队、武警均为单行
    "民航": "single",
    "使领馆": "single",
    "警用": "single",
    "教练车": "single",
    "港澳入出境": "single",
    "新能源": "single",
    "普通民用": "single",
    "短牌": "single",
    "军队": "single",
    "武警": "single",
}


__all__ = [
    "PROVINCE_CHARS",
    "PROVINCE_LOOKALIKES",
    "SUFFIX_LOOKALIKES",
    "HEAD_LOOKALIKES",
    "SEQ_LETTERS",
    "ISSUER_LETTERS",
    "CIVIL_AVIATION_HEADS",
    "CONFUSION_MAP",
    "FORMATS",
    "DEFAULT_LAYOUT",
    "LOOSE_STRUCTURE",
    "PlateFormat",
    "identify_format",
    "loose_match",
    "is_valid",
    "correct",
    "plate_score",
    "describe",
]
