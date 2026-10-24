"""The look shared by the README figures: warm paper, ink lines, one red accent, a
monospaced label font, and "chips" (small labelled boxes with a hard offset shadow).

Fonts: pass a .ttf with --font (the figures in docs/figures use Geist Mono), or set
UNO_FIG_FONT; otherwise DejaVu Sans Mono, which ships with matplotlib.
"""

import os
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

PAPER = (244, 239, 230)
PAPER_2 = (235, 228, 216)
INK = (21, 20, 18)
INK_2 = (94, 88, 80)
ACCENT = (210, 52, 69)        # UNO red: large shapes and fills
ACCENT_INK = (166, 31, 48)    # the same red, dark enough for text on paper

COLOR_WORDS = {"r": "RED", "g": "GREEN", "b": "BLUE", "y": "YELLOW"}
VALUE_WORDS = {"draw_2": "+2", "reverse": "REVERSE", "skip": "SKIP"}

_font_path: Optional[str] = os.environ.get("UNO_FIG_FONT") or None


def set_font(path: Optional[str]) -> None:
    global _font_path
    if path:
        _font_path = path


def font_path() -> str:
    if _font_path:
        return _font_path
    from matplotlib import font_manager
    return font_manager.findfont("DejaVu Sans Mono")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path(), size)


def card_label(code: str) -> str:
    """"r_8" -> "RED 8", "y_draw_2" -> "YELLOW +2", "draw_4" -> "+4", "yellow_token" -> "TOKEN"."""
    if code.endswith("_token"):
        return "TOKEN"
    if code == "draw_4":
        return "+4"
    if code == "wild":
        return "WILD"
    color, value = code.split("_", 1)
    return f"{COLOR_WORDS[color]} {VALUE_WORDS.get(value, value)}"


def chip_size(text: str, fnt: ImageFont.FreeTypeFont, pad: Tuple[int, int] = (10, 6)) -> Tuple[int, int]:
    left, top, right, bottom = fnt.getbbox(text)
    return right - left + 2 * pad[0], bottom - top + 2 * pad[1]


def draw_chip(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fnt: ImageFont.FreeTypeFont,
              fg=INK, bg=PAPER, border=INK, shadow: int = 3, pad: Tuple[int, int] = (10, 6),
              strike: bool = False) -> Tuple[int, int]:
    """A labelled box at xy (top-left) with a hard shadow; returns its (width, height)."""
    w, h = chip_size(text, fnt, pad)
    x, y = xy
    if shadow:
        draw.rectangle([x + shadow, y + shadow, x + w + shadow, y + h + shadow], fill=border)
    draw.rectangle([x, y, x + w, y + h], fill=bg, outline=border, width=2)
    left, top, _, _ = fnt.getbbox(text)
    draw.text((x + pad[0] - left, y + pad[1] - top), text, font=fnt, fill=fg)
    if strike:
        mid = y + h // 2
        draw.line([x + pad[0] - 2, mid, x + w - pad[0] + 2, mid], fill=fg, width=2)
    return w, h


def paper_canvas(width: int, height: int, color=PAPER) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def hex_color(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def matplotlib_style() -> None:
    """Charts on paper, ink axes, the label font."""
    import matplotlib
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path())
    name = font_manager.FontProperties(fname=font_path()).get_name()
    matplotlib.rcParams.update({
        "font.family": name, "font.size": 10,
        "figure.facecolor": hex_color(PAPER), "axes.facecolor": hex_color(PAPER),
        "savefig.facecolor": hex_color(PAPER),
        "axes.edgecolor": hex_color(INK), "axes.labelcolor": hex_color(INK),
        "xtick.color": hex_color(INK), "ytick.color": hex_color(INK), "text.color": hex_color(INK),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": hex_color(PAPER_2), "grid.linewidth": 1.0,
    })
