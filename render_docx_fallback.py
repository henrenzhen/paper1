from __future__ import annotations

import html
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
DOCX_PATH = ROOT / "deliverables" / "ZDS论文拒稿复盘与SCI_Q2+重构路线_证据审计版.docx"
PDF_PATH = ROOT / "tmp" / "docx_render" / "zds_report" / "zds_report_fallback.pdf"

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2E5D8A")
TEAL = colors.HexColor("#177E89")
DARK = colors.HexColor("#22303C")
MID = colors.HexColor("#D9DEE3")
LIGHT = colors.HexColor("#F7F9FA")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Deng", r"C:\Windows\Fonts\Deng.ttf"))
    pdfmetrics.registerFont(TTFont("DengBold", r"C:\Windows\Fonts\Dengb.ttf"))


def iter_block_items(document: Document):
    parent = document.element.body
    for child in parent.iterchildren():
        if child.tag == qn("w:p"):
            yield DocxParagraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, document)


def paragraph_text(paragraph: DocxParagraph) -> str:
    text_nodes = paragraph._p.xpath(".//w:t")
    parts = [node.text or "" for node in text_nodes]
    return "".join(parts)


def has_page_break(paragraph: DocxParagraph) -> bool:
    return bool(paragraph._p.xpath(".//w:br[@w:type='page']"))


def page_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Deng", 7.2)
    canvas.setFillColor(colors.HexColor("#7C8994"))
    canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 9 * mm, "ZDS 论文证据审计  ·  2026-07-29")
    canvas.drawCentredString(A4[0] / 2, 8 * mm, f"内部研究决策材料   |   {doc.page}")
    canvas.restoreState()


def build_styles():
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=8.9,
            leading=12.1,
            textColor=DARK,
            spaceAfter=4,
            splitLongWords=True,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="DengBold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="DengBold",
            fontSize=11.3,
            leading=14,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName="DengBold",
            fontSize=9.7,
            leading=12,
            textColor=TEAL,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=8.7,
            leading=11.8,
            leftIndent=11,
            firstLineIndent=-7,
            bulletIndent=2,
            textColor=DARK,
            spaceAfter=2.5,
        ),
        "cover": ParagraphStyle(
            "cover",
            parent=base["Title"],
            fontName="DengBold",
            fontSize=23,
            leading=30,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceBefore=70,
            spaceAfter=12,
        ),
        "center": ParagraphStyle(
            "center",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=BLUE,
            spaceAfter=7,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=7.2,
            leading=9.3,
            textColor=DARK,
            alignment=TA_LEFT,
            splitLongWords=True,
        ),
        "cell_head": ParagraphStyle(
            "cell_head",
            parent=base["BodyText"],
            fontName="DengBold",
            fontSize=7.3,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
            splitLongWords=True,
        ),
    }


def safe_markup(text: str) -> str:
    return html.escape(text).replace("\n", "<br/>")


def paragraph_flowable(paragraph: DocxParagraph, styles, index: int):
    text = paragraph_text(paragraph).strip()
    if not text:
        return Spacer(1, 2)
    style_name = paragraph.style.name if paragraph.style else ""
    if index < 12 and "ZDS 论文拒稿复盘" in text:
        style = styles["cover"]
    elif index < 18 and ("基于论文 PDF" in text or "Prepared for" in text):
        style = styles["center"]
    elif style_name.startswith("Heading 1"):
        style = styles["h1"]
    elif style_name.startswith("Heading 2"):
        style = styles["h2"]
    elif style_name.startswith("Heading 3"):
        style = styles["h3"]
    elif style_name.startswith("List Bullet"):
        return Paragraph("• " + safe_markup(text), styles["bullet"])
    elif style_name.startswith("List Number"):
        return Paragraph(safe_markup(text), styles["bullet"])
    else:
        style = styles["body"]
    return Paragraph(safe_markup(text), style)


def table_flowable(table: DocxTable, styles, available_width: float):
    ncols = max(len(row.cells) for row in table.rows)
    data = []
    for r_idx, row in enumerate(table.rows):
        vals = []
        for cell in row.cells:
            text = "\n".join(p.text for p in cell.paragraphs).strip()
            vals.append(Paragraph(safe_markup(text), styles["cell_head"] if r_idx == 0 and ncols > 1 else styles["cell"]))
        data.append(vals)
    if ncols == 1:
        col_widths = [available_width]
    elif ncols == 2:
        col_widths = [available_width * 0.28, available_width * 0.72]
    elif ncols == 3:
        col_widths = [available_width * 0.18, available_width * 0.46, available_width * 0.36]
    elif ncols == 4:
        col_widths = [available_width * 0.18, available_width * 0.40, available_width * 0.18, available_width * 0.24]
    else:
        col_widths = [available_width / ncols] * ncols
    repeat = 1 if ncols > 1 else 0
    tbl = Table(data, colWidths=col_widths, repeatRows=repeat, hAlign="CENTER", splitByRow=1)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if ncols > 1:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
        for i in range(1, len(data)):
            if i % 2 == 0:
                commands.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    else:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF1F7")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
        ])
    tbl.setStyle(TableStyle(commands))
    return tbl


def build() -> Path:
    register_fonts()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    document = Document(DOCX_PATH)
    page_width, page_height = A4
    left = right = 18 * mm
    top = 18 * mm
    bottom = 16 * mm
    frame = Frame(left, bottom, page_width - left - right, page_height - top - bottom, id="normal")
    template = PageTemplate(id="A4", frames=[frame], onPage=page_header_footer)
    pdf = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="ZDS 论文拒稿复盘与 SCI Q2+ 重构路线",
        author="Codex",
    )
    pdf.addPageTemplates([template])

    story = []
    para_index = 0
    available = page_width - left - right
    for block in iter_block_items(document):
        if isinstance(block, DocxParagraph):
            if has_page_break(block):
                story.append(PageBreak())
                continue
            flowable = paragraph_flowable(block, styles, para_index)
            para_index += 1
            story.append(flowable)
        else:
            story.append(table_flowable(block, styles, available))
            story.append(Spacer(1, 4))
    pdf.build(story)
    return PDF_PATH


if __name__ == "__main__":
    print(build())
