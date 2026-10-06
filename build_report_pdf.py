from __future__ import annotations

import html
import re
from pathlib import Path

from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "docs" / "cs521-part2-report.md"
OUTPUT = ROOT / "outputs" / "exp-0011-part2-report" / "cs521-part2-report.pdf"


def inline_markup(value: str) -> str:
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = value.replace("–", "-").replace("−", "-").replace("‑", "-")
    value = html.escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*(.+?)\*", r"<i>\1</i>", value)
    return re.sub(r"`(.+?)`", r'<font face="DejaVuSansMono">\1</font>', value)


def build() -> None:
    font_path = Path(get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(font_path / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(font_path / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", str(font_path / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSansMono", str(font_path / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("DejaVuSans", bold="DejaVuSans-Bold", italic="DejaVuSans-Oblique")
    navy = colors.HexColor("#18324d")
    body = ParagraphStyle(
        "body",
        fontName="DejaVuSans",
        fontSize=8.5,
        leading=12.3,
        textColor=navy,
        spaceAfter=7,
        alignment=TA_LEFT,
    )
    title = ParagraphStyle(
        "title",
        parent=body,
        fontName="DejaVuSans-Bold",
        fontSize=17.5,
        leading=23,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    heading = ParagraphStyle(
        "heading",
        parent=body,
        fontName="DejaVuSans-Bold",
        fontSize=11.5,
        leading=15,
        spaceBefore=15,
        spaceAfter=7,
    )
    caption = ParagraphStyle(
        "caption",
        parent=body,
        fontSize=7.7,
        leading=10.5,
        textColor=colors.HexColor("#4b5e70"),
        alignment=TA_CENTER,
        spaceAfter=11,
    )
    code = ParagraphStyle(
        "code",
        parent=body,
        fontName="DejaVuSansMono",
        fontSize=7.4,
        leading=10,
        leftIndent=10,
        spaceAfter=10,
    )
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, firstLineIndent=-10, spaceAfter=4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=(595.28, 841.89),
        leftMargin=48,
        rightMargin=48,
        topMargin=52,
        bottomMargin=48,
        title="Snow-Season Feature Graphs and Preliminary Spectral Divides",
        author="Haoyang Feng and Feiyang Zhong",
    )
    story = []
    paragraph_lines: list[str] = []
    bullet_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            value = " ".join(line.rstrip().strip() for line in paragraph_lines)
            story.append(Paragraph(inline_markup(value), body))
            paragraph_lines.clear()

    def flush_bullets() -> None:
        if bullet_lines:
            for item in bullet_lines:
                story.append(Paragraph("• " + inline_markup(item), bullet))
            bullet_lines.clear()

    for line in SOURCE.read_text().splitlines():
        if line.startswith("```"):
            flush_paragraph()
            flush_bullets()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), code))
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            flush_bullets()
            continue
        if line.startswith("# "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(inline_markup(line[2:]), title))
            continue
        if line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(inline_markup(line[3:]), heading))
            continue
        if line.startswith("!["):
            flush_paragraph()
            flush_bullets()
            match = re.fullmatch(r"!\[(.+)\]\((.+)\)", line)
            if match is None:
                raise ValueError(f"Invalid figure line: {line}")
            description, relative_path = match.groups()
            path = (SOURCE.parent / relative_path).resolve()
            image_width, image_height = ImageReader(str(path)).getSize()
            scale = min(499 / image_width, 250 / image_height)
            figure = Image(str(path), width=image_width * scale, height=image_height * scale)
            story.append(
                KeepTogether([figure, Spacer(1, 4), Paragraph(inline_markup(description), caption)])
            )
            continue
        if line.startswith("- "):
            flush_paragraph()
            bullet_lines.append(line[2:])
            continue
        flush_bullets()
        paragraph_lines.append(line)
        if line.endswith("  "):
            flush_paragraph()
    flush_paragraph()
    flush_bullets()

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("DejaVuSans", 7.5)
        canvas.setFillColor(colors.HexColor("#52667a"))
        canvas.drawString(48, 31, "CS-521 | Homework 1 Part II")
        canvas.drawRightString(547, 31, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
