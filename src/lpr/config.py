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
    #: 双行牌召回下限：双行牌理论宽高比 2.0（440×220mm），但透视/倾斜
    #: /裁剪后实测可低至 ~1.3（double_yellow.jpg 实测 1.86）。
    #: 落在 [double_min_aspect, min_aspect] 的轮廓不直接丢弃，
    #: 经"强底色"闸门后进入独立候选池。
    double_min_aspect: float = 1.3
    #: 双行牌区间候选的底色强度下限：核心区内蓝/黄/绿单色占比。
    #: 真双行牌底色占比极高（实测双层黄牌 0.93），0.20 余量充足；
    #: 方块状车身件/车灯的杂色很难越过这道闸门。
    double_color_ratio: float = 0.20
    #: 双行牌候选的独立配额（与常规/补救池分开，不挤占名额）
    double_max_candidates: int = 2
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
    #: 绿牌渐变严重，横向阈值单独放宽
    color_row_ratio_green: float = 0.5
    #: 精定位裁剪边界时行/列的**边缘占比下限**：某行/列只要还有这么多
    #: 底色像素就认为仍落在牌内，予以保留。
    #:
    #: 旧实现用绝对底色阈值 0.8 定边界，但它只能罩住"纯底色"的行列，而
    #: 车牌**末位字符紧贴牌缘**时，穿过该字符的列底色占比会被字符笔画咬到
    #: 0.3~0.7，达不到 0.8 就被当边界裁掉——整字丢失。实测 CCPD
    #: 皖AN9491→皖AN949、皖AV8008→皖A·V800（原始裁剪本已读对，精定位反而
    #: 把尾字裁掉）。降到 0.15 后，牌外的行列占比≈0 仍被裁掉，牌内带字符
    #: 的行列则全部保留，既保住尾字又不放过越界背景。
    #: 绿牌因渐变严重（真实底色占比本就低）沿用 ``color_row_ratio_green``。
    color_edge_ratio: float = 0.15
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
    #: 危险品牌判定：白色像素达到黄色像素的该倍数时，判定文字为白色
    #: （黑底白字的普通黑牌，如港澳/使领馆牌），排除出危险品。
    #: 真危险品牌上黄字是绝对前景、白色像素≈0；实测误判的黑底白字牌
    #: white/yellow ≈ 1.0~3.2（黄色只是反光/红字渗色等噪声），取 0.7 隔开。
    hazmat_white_ratio: float = 0.7
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
        if not 0 < self.color_edge_ratio < 1:
            raise ValueError("color_edge_ratio 必须在 (0, 1)")
        if self.min_aspect >= self.max_aspect:
            raise ValueError("min_aspect 必须小于 max_aspect")
        if not 1.0 <= self.double_min_aspect < self.min_aspect:
            raise ValueError("double_min_aspect 必须在 [1.0, min_aspect)")
        if not 0 < self.double_color_ratio < 1:
            raise ValueError("double_color_ratio 必须在 (0, 1)")
        if self.min_chars < 1:
            raise ValueError("min_chars 必须为正")


#: 可作为 ``engine.default`` 的取值（**单一事实来源**）。
#:
#: 这里既有真引擎（``cv`` / ``deep`` / ``yolo`` / ``fusion``），也有两个
#: **策略别名**：``auto``（按实测强度挑最强可用引擎，见
#: :data:`BEST_ENGINE_ORDER`）与 ``cascade``（解析成 deep；deep 内部本就是
#: "CV 定位 → 纯识别"的级联）。
#: 之所以收成常量：此前 ``("cv", "deep", "cascade")`` 在白名单里抄了三份
#: （config.validate / cli 的 argparse choices ×2），加第三个引擎时漏改
#: 一处就会出现"config 认、CLI 不认"的割裂。
ENGINE_CHOICES = ("auto", "cv", "deep", "cascade", "yolo", "fusion")

#: ``auto`` 挑"最强可用引擎"的顺序：**按实测严格命中率排，不按名义上的新/旧**。
#:
#: 100 张 CCPD 实测（``docs/ACCURACY.md`` §2.1）：``yolo`` 89%、``deep`` 79%、
#: ``cv`` 0%。所以只要有权重就用 yolo —— 这与"yolo 权重不入库"并不矛盾：
#: 没有权重的机器上 ``yolo`` 的 ``is_available()`` 为 False，``auto`` 自动落到
#: ``deep``，行为与旧版默认（``cascade``）完全一致。
#:
#: ⚠️ **这不是"yolo 处处更强"**：特殊牌（港/警/领/学/双层）自有样张上
#: ``deep`` 10/10、``yolo`` 8/10，排名反转（§4）。默认值按"常规牌占绝大多数"
#: 的现实取 yolo 优先；如果你的车牌构成以特殊牌为主，把 ``engine.default``
#: 显式写成 ``deep`` 或 ``cascade`` —— 那是唯一需要改的一处配置。
BEST_ENGINE_ORDER = ("yolo", "deep", "cv")

#: ``cascade`` 解析成**具体引擎**的尝试顺序：先 ``deep``（"CV 定位 → 纯识别
#: 网络"的级联就是它的内在行为），deep 不可用再退 ``cv``。
#:
#: 与 ``ENGINE_CHOICES`` 一样是**单一事实来源**：``api.Recognizer._resolve_engine``
#: 用它挑引擎，界面下拉框用它算 ``cascade`` 的可用性与"退化后会跑谁"。
#: （``auto`` **不用**它 —— 那是个独立的偏好序，见 :data:`BEST_ENGINE_ORDER`；
#: 这里刻意不合并：``cascade`` 的语义是"deep 的别名"，永远不该悄悄变成 yolo。）
#: 两边各写一份的话，界面提示就会和运行时实际干的活分叉。
#:
#: 注意别和 ``api.CASCADE_ORDER = ("cv", "deep")`` 混淆：那是
#: ``engine.mode = "cascade"`` 时**挑兜底引擎**的顺序（cv 优先，因为它快），
#: 和"``cascade`` 这个引擎名等于谁"是两件事。
CASCADE_RESOLVE_ORDER = ("deep", "cv")

#: 可以作为**融合成员**的引擎：都是具体引擎，不含别名与 ``fusion`` 自己
FUSION_MEMBER_CHOICES = ("cv", "deep", "yolo")
#: 融合引擎的策略
FUSION_STRATEGIES = ("ensemble", "chain", "auto")


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

    #: 默认引擎：
    #:   ``"auto"``（**默认**，按实测强度挑最强可用引擎：yolo → deep → cv，
    #:   见 :data:`BEST_ENGINE_ORDER`）
    #:   | ``"cv"``（传统，快但精度有限）| ``"deep"``（深度学习）
    #:   | ``"cascade"``（= deep，含 CV 定位引导）
    #:   | ``"yolo"``（检测识别引擎，需自备 ONNX 权重，见 models/yolo/README.md）
    default: str = "auto"
    #: 跨引擎兜底策略："single"（只跑主引擎）| "cascade"（主引擎未命中时再试另一个可用引擎）
    mode: str = "single"
    #: 传统 CV 引擎参数
    locate: LocateConfig = field(default_factory=LocateConfig)
    #: 字符识别时是否启用 SVM 置信度加权投票
    use_confidence: bool = True

    #: 第二识别模型（ONNX）路径，用于**低置信交叉验证**。
    #:
    #: 主识别模型对某个候选完全读不出结果时，才用该模型补读并把读数
    #: 并入投票——只在"本来会失败"的路径上激活，因此对已命中的样本
    #: 零回归风险，只可能把失败样本救回来。实测 PP-OCRv4 mobile 与
    #: 内置 v3 模型的错误模式互补（如低分辨率省份字），是合适的第二票。
    #:
    #: 空字符串 = 关闭（默认）。相对路径按项目根目录解析。
    #: 模型下载（约 10MB）：
    #:   https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/
    #:   onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx
    secondary_rec_model: str = ""
    #: 第二模型读数的票权折扣（0~1]。主模型读数权为 1，第二模型略低，
    #: 避免在两模型意见相左时反客为主。
    secondary_vote_weight: float = 0.8

    #: 补救预处理：主识别计划**全部落空**时，换一组"增强"图像再试一次
    #: （见 ``deep_engine._RESCUE_VARIANTS``）。
    #:
    #: 实测收益**因引擎而异**（``docs/ACCURACY.md`` §5.3 / §5.5）：
    #:
    #: * ``yolo``：+1 张严格命中，字符级 +2.9pp，「有框无字」8 → 4 张；
    #: * ``deep``：识别收益 **0 张**（它的失败多是"无框"，不是"读不出"），
    #:   却要为它付约 30% 的耗时（716 → 991 ms/张，中位数）。
    #:
    #: 所以留成开关：吞吐敏感、且以 ``deep`` 为主的服务可以关掉。
    #: 默认 ``True``（= v2.7.2 起的行为，不动）。
    #:
    #: 注意这是**引擎级**开关而不是 ``deep`` 专属：``yolo`` 复用同一条识别
    #: 管线（只是识别计划不同），关掉它对两者同时生效。
    rescue_preprocess: bool = True

    #: yolo 引擎：检测模型输出的**属性列顺序**：``"cls_first"``
    #: （Ultralytics 导出：box+cls+kpts）| ``"kpt_first"``（we0091234 系：
    #: box+obj+kpts+cls）。
    #:
    #: 两种布局**属性个数相同，形状区分不出来**，用错了框会整体错位，
    #: 因此必须显式声明。默认 ``"kpt_first"``：``models/yolo/README.md``
    #: 路径 A（GPL-3.0 的 we0091234 权重，经官方 export.py 导出 ONNX）就是
    #: 这个列序，也是绝大多数用户的取权重路径。用 Ultralytics 自训练导出
    #: 的请改成 ``"cls_first"``。
    yolo_attr_order: str = "kpt_first"
    #: yolo 引擎：字符识别后端。``"auto"``（优先检测模型自带的识别模型
    #: ``plate_rec.onnx``，没有则复用 deep 引擎的 OCR）| ``"crnn"``（只用
    #: 自带识别模型）| ``"deep"``（强制走 deep 的 OCR）| ``"none"``（纯检测，
    #: 返回带框但 text 为空的结果）。
    yolo_recognize_backend: str = "auto"
    #: yolo 引擎：权重目录（放 ``plate_det.onnx`` 的地方）。空 = 默认
    #: ``<项目根>/models/yolo``。
    yolo_models_dir: str = ""

    #: 三合一引擎（``fusion``）的策略：
    #:   ``"ensemble"``（默认）并联投票融合 —— 所有可用成员各跑一次，按空间
    #:   分组后组内投票，准确率与召回都高于任一单引擎，耗时≈三者相加；
    #:   ``"chain"`` 顺序兜底链 —— 依次尝试，拿到够可信的结果就停；
    #:   ``"auto"`` 特征选优 —— 按图像特征（分辨率/模糊度）排序后按序试。
    fusion_strategy: str = "ensemble"
    #: 融合成员及顺序（顺序即 ``chain`` 的尝试顺序）。
    #: 只能填具体引擎；成员缺依赖或缺权重时**自动退席**，不影响其余成员。
    fusion_members: tuple[str, ...] = ("yolo", "deep", "cv")
    #: 各成员的票权（仅 ``ensemble`` 用）。缺省成员按 1.0 计。
    #: 传统 CV 给 0.6：清晰正视角下很可靠，但整体错误率高于另两个。
    fusion_engine_weights: dict[str, float] = field(
        default_factory=lambda: {"yolo": 1.0, "deep": 1.0, "cv": 0.6}
    )
    #: 同一车牌的分组 IoU 阈值（``ensemble`` 用）：两框重叠超过它即视为
    #: 同一个车牌，进入同一组投票。
    fusion_group_iou: float = 0.4
    #: ``chain`` 的"够可信"门槛：某成员给出 ``confidence >= 该值`` 的读数就停，
    #: 不再试后面的成员。
    fusion_chain_accept: float = 0.75
    #: ``auto`` 的"够可信"判定之外唯一的可调项：**图像清晰且够大时**先试哪个成员。
    #: 小图/糊图固定检测网络（``yolo``）优先，不归它管。
    #:
    #: 默认 ``"auto"`` = 在**当前在岗的成员**里按 :data:`BEST_ENGINE_ORDER`
    #: 取最强的一个（有权重就是 yolo）。为什么默认不是写死的名字：两套实测
    #: 数据指向相反 —— CCPD 常规牌上 yolo 更强（89% vs deep 79%），而特殊牌
    #: （港/警/领/学/双层）自有样张上 deep 全对（10/10 vs yolo 8/10）。
    #: 写死任何一边都会在另一边付出代价，见 ``docs/ACCURACY.md`` §5.2。
    #:
    #: 显式写成员名（``"deep"`` / ``"yolo"`` / ``"cv"``）就按你写的来，
    #: 不做任何"更懂你"的替换。
    fusion_auto_clear_first: str = "auto"
    #: 融合结果要留下所需的**最少一致成员数**（``ensemble`` 用）。
    #: 默认 2 = 至少两个成员一致才留（或票权达到 ``fusion_trusted_weight``
    #: 的成员独存）。
    #:
    #: 为什么不是 1（并集）？实测：``cv`` 引擎在 4096×2304 的真实样张上给
    #: 9 条候选（7 条误检），而 ``yolo`` 给 2 条全中、``deep`` 给 1 条全中。
    #: 只做并集会让最弱成员的误检全部污染输出。取 1 则退化为并集（最大召回）。
    fusion_min_support: int = 2
    #: 单成员独存所需的票权阈值：胜出号码的票权达到它就保留，不看票数。
    #: 默认 1.0 = ``yolo`` / ``deep`` 这类权重 1.0 的成员可以独存；
    #: ``cv``（默认票权 0.6）的孤立结果则需要第二个成员背书。
    fusion_trusted_weight: float = 1.0

    def validate(self) -> None:
        #: ``yolo`` / ``fusion`` 都在列 —— 它们是**真引擎**，不是 cascade 那样的别名。
        #: 此前白名单里没有 yolo，导致权重装好了也选不动（只能在 CLI 里看到它"不可用"）。
        if self.default not in ENGINE_CHOICES:
            raise ValueError(f"未知引擎: {self.default}（可选: {', '.join(ENGINE_CHOICES)}）")
        if self.mode not in ("single", "cascade"):
            raise ValueError(f"未知模式: {self.mode}")
        if self.yolo_attr_order not in ("cls_first", "kpt_first"):
            raise ValueError(
                f"未知 yolo_attr_order: {self.yolo_attr_order}（可选: cls_first, kpt_first）"
            )
        if self.yolo_recognize_backend not in ("auto", "crnn", "deep", "none"):
            raise ValueError(
                f"未知 yolo_recognize_backend: {self.yolo_recognize_backend}"
                "（可选: auto, crnn, deep, none）"
            )
        self._validate_fusion()
        self.locate.validate()

    def _validate_fusion(self) -> None:
        """融合引擎的字段校验。

        这里**故意不 import** ``engines`` 包（config 是被引擎导入的底层模块，
        反向依赖会成环），成员白名单用 :data:`FUSION_MEMBER_CHOICES` 这份常量；
        两者的同步由 ``tests/test_fusion_engine.py`` 断言。
        """
        if self.fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"未知 fusion_strategy: {self.fusion_strategy}"
                f"（可选: {', '.join(FUSION_STRATEGIES)}）"
            )
        if not self.fusion_members:
            raise ValueError("fusion_members 不能为空")
        seen: set[str] = set()
        for name in self.fusion_members:
            if name not in FUSION_MEMBER_CHOICES:
                raise ValueError(
                    f"未知 fusion_members 成员: {name!r}"
                    f"（可选: {', '.join(FUSION_MEMBER_CHOICES)}；策略名与 fusion 本身不能当成员）"
                )
            if name in seen:
                raise ValueError(f"fusion_members 成员重复: {name!r}")
            seen.add(name)
        for name, weight in self.fusion_engine_weights.items():
            if weight <= 0:
                raise ValueError(f"fusion_engine_weights[{name!r}] 必须为正数，当前 {weight}")
        if not 0.0 < self.fusion_group_iou <= 1.0:
            raise ValueError(f"fusion_group_iou 必须在 (0, 1]，当前 {self.fusion_group_iou}")
        if not 0.0 <= self.fusion_chain_accept <= 1.0:
            raise ValueError(f"fusion_chain_accept 必须在 [0, 1]，当前 {self.fusion_chain_accept}")
        if self.fusion_min_support < 1:
            raise ValueError(f"fusion_min_support 至少为 1，当前 {self.fusion_min_support}")
        if self.fusion_trusted_weight < 0:
            raise ValueError(f"fusion_trusted_weight 不能为负，当前 {self.fusion_trusted_weight}")
        if (
            self.fusion_auto_clear_first != "auto"
            and self.fusion_auto_clear_first not in FUSION_MEMBER_CHOICES
        ):
            raise ValueError(
                f"未知 fusion_auto_clear_first: {self.fusion_auto_clear_first!r}"
                f"（可选: auto, {', '.join(FUSION_MEMBER_CHOICES)}）"
            )


@dataclass
class UIConfig:
    """界面相关。"""

    #: 窗口标题。**不带版本号** —— 版本号在状态栏显示（``MainWindow`` 会拼
    #: ``__version__``），写进标题的版本号必然随版本演进漂移（此处曾停在 v2.3.9）。
    window_title: str = "车牌识别系统"
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
    #: 日志级别（DEBUG / INFO / WARNING / ERROR / CRITICAL）
    log_level: str = "INFO"
    #: 是否把运行日志写入文件。关掉只剩控制台输出，关窗即丢
    log_to_file: bool = True
    #: 日志文件目录（相对当前工作目录）；为空串同样表示不写文件
    log_dir: str = "logs"
    #: 单个日志文件大小上限（MB），超出后轮转为 ``*.log.1`` 等
    log_max_mb: int = 5
    #: 保留的历史日志份数
    log_backups: int = 5

    def logging_kwargs(self, *, tag: str = "") -> dict:
        """转成 :func:`lpr.utils.logging_setup.setup_logging` 的关键字参数。

        CLI、界面、批量子进程三处入口共用，避免各自记一遍字段名
        （此前只有 ``log_level`` 一个字段，漏传也只是静默降级）。
        """
        return {
            "level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_dir": self.log_dir,
            "max_mb": self.log_max_mb,
            "backups": self.log_backups,
            "tag": tag,
        }

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
            _apply_log_fields(cfg, raw)

        # 环境变量优先级最高，便于临时调试
        if env_engine := os.getenv("LPR_ENGINE"):
            cfg.engine.default = env_engine
        if env_level := os.getenv("LPR_LOG_LEVEL"):
            cfg.log_level = env_level
        if env_dir := os.getenv("LPR_LOG_DIR"):
            cfg.log_dir = env_dir
        if env_file := os.getenv("LPR_LOG_FILE"):
            # 显式给了 LPR_LOG_FILE 就按它来（"0"/"off" 关掉），CI 里可彻底静音
            cfg.log_to_file = _as_bool(env_file, cfg.log_to_file)
        if env_mask := os.getenv("LPR_MASK_OUTPUT"):
            cfg.security.mask_output = _as_bool(env_mask, cfg.security.mask_output)

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
            "log_to_file": self.log_to_file,
            "log_dir": self.log_dir,
            "log_max_mb": self.log_max_mb,
            "log_backups": self.log_backups,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> Config:
        """从 :meth:`to_payload` 的结果还原配置。"""
        cfg = cls()
        if payload:
            _apply_overrides(cfg.engine, payload.get("engine", {}))
            _apply_overrides(cfg.ui, payload.get("ui", {}))
            _apply_overrides(cfg.security, payload.get("security", {}))
            _apply_log_fields(cfg, payload)
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


def _as_bool(value: Any, fallback: bool) -> bool:
    """宽松地读布尔：``config.json`` 是给人手改的，``"0"`` / ``"off"`` 也算假。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return fallback


def _as_int(value: Any, fallback: int, *, minimum: int) -> int:
    """宽松地读整数并夹住下界；读不出来时保持默认值（日志设置不值得让配置加载失败）。"""
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return fallback


def _apply_log_fields(cfg: Config, raw: dict) -> None:
    """落 5 个顶层日志字段（``log_level`` / ``log_to_file`` / ``log_dir`` / ...）。

    单列一个函数是因为它们既出现在 ``config.json`` 顶层，也出现在
    :meth:`Config.to_payload` 的字典里，两处必须用同一套宽松解析
    （``_apply_overrides`` 只认 dataclass 字段，不认这些扁平键）。
    """
    if not raw:
        return
    if "log_level" in raw:
        cfg.log_level = str(raw["log_level"])
    cfg.log_to_file = _as_bool(raw.get("log_to_file"), cfg.log_to_file)
    if "log_dir" in raw:
        cfg.log_dir = str(raw["log_dir"])
    cfg.log_max_mb = _as_int(raw.get("log_max_mb"), cfg.log_max_mb, minimum=1)
    cfg.log_backups = _as_int(raw.get("log_backups"), cfg.log_backups, minimum=0)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ENGINE_CHOICES",
    "FUSION_MEMBER_CHOICES",
    "FUSION_STRATEGIES",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "Config",
    "EngineConfig",
    "LocateConfig",
    "SecurityConfig",
    "UIConfig",
    "asdict",
]
