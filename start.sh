#!/bin/sh
# ============================================================
#  车牌识别系统 —— Linux / macOS 启动壳
#
#  直接运行      -> 打开界面
#  带参数运行    -> 转交命令行，例如
#      ./start.sh engines
#      ./start.sh predict car.jpg --engine deep
#
#  ./start.sh doctor   -> 只读环境体检：解释器、运行时依赖、模型权重、
#                         各引擎可用性。不建 venv、不装包、不启动界面。
#
#  与 start.bat 的分工：两者都是"薄引导壳"，真正的活儿（banner、版本检查、
#  .venv 创建与 pip、权重报告、中文错误）全在 tools/launcher.py 里，由 Python
#  打印 —— 中文输出因此不依赖终端编码。.bat 必须保持纯 ASCII 是因为 cmd.exe
#  的字节/字符偏移 bug（见该文件的注释）；sh 没有这个问题，所以这里可以有中文。
#
#  文件里不写版本号：写死的版本号必然随发布漂移（start.bat 曾停在 v2.3.6），
#  真实版本由 tools/launcher.py 启动时打印。
#
#  环境变量（由 tools/launcher.py 消费）：
#      LPR_PYTHON   强制指定解释器
#      LPR_NO_VENV  不使用/不创建项目内 .venv
# ============================================================
set -u

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
cd "$HERE" || exit 1

# ---------- 找一个能用的解释器（先命中先用） ----------
PY=""
if [ -n "${LPR_PYTHON:-}" ]; then
    PY=$LPR_PYTHON
fi
if [ -z "$PY" ] && [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
fi
if [ -z "$PY" ] && [ -x "$HERE/.venv/bin/python3" ]; then
    PY="$HERE/.venv/bin/python3"
fi

# 淘汰"存在但跑不起来"的候选（过期的 LPR_PYTHON、坏掉的 venv）
if [ -n "$PY" ] && ! "$PY" -c "import sys" >/dev/null 2>&1; then
    PY=""
fi

for cand in python3 python; do
    if [ -z "$PY" ] && command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c "import sys" >/dev/null 2>&1; then
            PY=$cand
        fi
    fi
done

if [ -z "$PY" ]; then
    cat >&2 <<'EOF'
[错误] 未找到 Python 解释器。

  请先安装 Python 3.10 或更高版本（64 位），然后重新运行 ./start.sh：
    https://www.python.org/downloads/
  也可以显式指定解释器：
    LPR_PYTHON=/path/to/python3 ./start.sh
EOF
    exit 1
fi

# ---------- 转交 tools/launcher.py ----------
"$PY" "$HERE/tools/launcher.py" "$@"
RC=$?

# 只在"失败 + 真终端"时停一下：双击启动的终端会立刻关闭，
# 不留下错误信息等于什么都没说；管道/脚本调用则不该被卡住。
if [ "$RC" -ne 0 ] && [ -t 0 ]; then
    printf '\n按回车键关闭…'
    read -r _
fi

exit "$RC"
