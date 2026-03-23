"""统一的Matplotlib字体与样式安全配置

在无中文字体的环境中优先采用跨平台安全字体；
如检测到可用的中文字体则优先使用，同时保留英文回退。
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt


def _detect_available_fonts() -> set[str]:
    try:
        import matplotlib.font_manager as fm
        return {f.name.lower() for f in fm.fontManager.ttflist}
    except Exception:
        return set()


def apply_safe_matplotlib_fonts(prefer_chinese: bool = False, base_font_size: int = 10) -> list[str]:
    """应用安全的Matplotlib字体与样式配置。

    Args:
        prefer_chinese: 若为True且检测到中文字体，则优先使用中文字体。
        base_font_size: 基础字体大小。

    Returns:
        实际生效的`sans-serif`字体候选列表。
    """
    available = _detect_available_fonts()

    # 候选中文字体关键词（lowercase对比）
    chinese_candidates = [
        "noto sans cjk sc",
        "noto sans cjk",
        "source han sans sc",
        "source han sans",
        "simhei",
        "microsoft yahei",
        "sarasa gothic sc",
        "wqy-microhei",
        "wqy-zenhei",
        "arial unicode ms",
        "droid sans fallback",
    ]

    # 英文/跨平台安全字体回退
    english_fallback = ["DejaVu Sans", "Arial", "Liberation Sans", "Droid Sans Fallback"]

    # 选取可用的中文字体
    selected_chinese = []
    if prefer_chinese:
        for name in chinese_candidates:
            for font in available:
                if name in font:
                    # 将原始大小写名称恢复为更通用形式
                    if "noto sans cjk sc" in font:
                        selected_chinese.append("Noto Sans CJK SC")
                    elif "source han sans sc" in font:
                        selected_chinese.append("Source Han Sans SC")
                    elif "wqy-microhei" in font:
                        selected_chinese.append("WenQuanYi Micro Hei")
                    elif "wqy-zenhei" in font:
                        selected_chinese.append("WenQuanYi Zen Hei")
                    elif "droid sans fallback" in font:
                        selected_chinese.append("Droid Sans Fallback")
                    else:
                        selected_chinese.append(font.title())
        # 去重但保持顺序
        seen = set()
        selected_chinese = [f for f in selected_chinese if not (f in seen or seen.add(f))]

    # 构造最终字体栈
    if selected_chinese:
        font_stack = selected_chinese + english_fallback
    else:
        font_stack = english_fallback

    # 应用rcParams
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = font_stack
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = base_font_size

    # 样式：尽量使用seaborn v0_8，否则回退
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        try:
            plt.style.use("seaborn")
        except Exception:
            plt.style.use("default")

    return font_stack


def get_safe_css_font_stack(prefer_chinese: bool = True) -> str:
    """返回统一的CSS字体栈字符串。"""
    # system-ui优先，兼容各平台；加入Noto/Source Han/DejaVu/Arial回退
    stack_ch = (
        "system-ui, -apple-system, 'Noto Sans', 'Noto Sans CJK SC', "
        "'Source Han Sans SC', 'DejaVu Sans', Arial, sans-serif"
    )
    stack_en = "system-ui, -apple-system, 'DejaVu Sans', Arial, 'Liberation Sans', sans-serif"
    return stack_ch if prefer_chinese else stack_en