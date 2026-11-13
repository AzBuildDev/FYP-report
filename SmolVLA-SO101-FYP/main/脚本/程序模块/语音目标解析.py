"""实机窗口的有限口令解析；只输出 RC/BC，不猜同音词或生成关节动作。"""
import re


def normalized_text(text):
    return text.translate(str.maketrans("紅藍綠塊標區積動機臂換為選擇這個謝請", "红蓝绿块标区积动机臂换为选择这个谢请"))


def parse_target_command(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 160:
        raise ValueError("没有听到明确指令，请说“抓红方块”或“切换到蓝色”")
    text = normalized_text(text).lower().strip()
    if re.search(r"不|勿|别|禁止|\b(?:don't|dont|not|never|stop)\b", text):
        raise ValueError("指令含否定或停止语义，请只说一个明确目标；停止请单独说“停止”")
    compact = re.sub(r"[\s，,。.!！?？、；;：:]", "", text)
    # 常见自然表达的安全归一化；仍要求单一红/蓝颜色和明确抓取/切换动作。
    colors = re.findall(r"红|蓝", compact)
    if len(colors) > 1:
        raise ValueError("颜色不明确或包含多个颜色")
    natural = [
        r"(?:请|帮我|现在|接下来|麻烦|机器人|我想让你)?(?:把|将)?(?:那个|这个|目标)?(?P<color>红|蓝)(?:色)?(?:的)?(?:方块|积木|块|那个)?(?:给我)?(?:拿起来|拿起|抓起来|抓住|抓|取出来|取|夹起来|夹住|选中)",
        r"(?:请|帮我|现在|接下来|麻烦|机器人)?(?:去)?(?:抓|拿|取|夹)(?:一下)?(?:那个|这个)?(?P<color>红|蓝)(?:色)?(?:的)?(?:方块|积木|块)?",
        r"(?:请|帮我|现在|接下来|麻烦)?(?:把|将)?(?:目标)?(?:切换|换|改)(?:到|成|为)?(?P<color>红|蓝)(?:色)?(?:方块|积木|目标)?",
    ]
    for pattern in natural:
        match = re.fullmatch(pattern, compact)
        if match:
            return "RC" if match['color'] == '红' else "BC"
    noun = r"(?P<color>红|蓝)(?:色)?(?:的)?(?:方块|积木|块)?"
    chinese = (r"(?:请|麻烦|帮我|现在|马上|给我|机器人|机械臂|我要)*"
               r"(?:抓取|抓住|抓|拿起|拿|拾取|夹起|选择|切换目标为|切换目标到|把目标切换到|"
               r"切换到|切换为|切换成|切换|换成|换到|改成|改抓|改为)?"
               r"(?:一下|一个|那个|这个)?" + noun +
               r"(?:(?:并|然后|再)?(?:把它)?(?:放到|放入|放在)(?:演示|指定|固定)?(?:的)?(?:目标区域|目标区|a区))?"
               r"(?:一下|吧|谢谢)*")
    match = re.fullmatch(chinese, compact)
    if match:
        return "RC" if match['color'] == '红' else "BC"
    english = re.sub(r"[,.!?]", "", text)
    english = re.sub(r"\s+", " ", english)
    if english in ('rc', 'red'):
        return 'RC'
    if english in ('bc', 'blue'):
        return 'BC'
    match = re.fullmatch(r"(?:please )?(?:pick up|grab|take|select|switch to|change to|switch target to) "
                         r"(?:the )?(red|blue)(?: cube| block)?"
                         r"(?: and (?:place|put) it (?:in|into) the target area)?", english)
    if match:
        return 'RC' if match[1] == 'red' else 'BC'
    raise ValueError("未匹配明确口令，请说“抓红方块”“切换到蓝色”或“停止”；不支持多目标、其他物体或其他放置区")
