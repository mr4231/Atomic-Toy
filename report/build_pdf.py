import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "FINAL_REPORT.md"
OUTPUT = ROOT / "FINAL_REPORT.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    fontSize=19, leading=24, alignment=TA_CENTER,
    spaceAfter=12
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading1"],
    fontSize=14, leading=18, spaceBefore=12, spaceAfter=7,
    keepWithNext=True
))
styles.add(ParagraphStyle(
    name="Subsection", parent=styles["Heading2"],
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=5,
    keepWithNext=True
))
styles.add(ParagraphStyle(
    name="BodyReport", parent=styles["BodyText"],
    fontSize=8.5, leading=12, spaceAfter=5,
    splitLongWords=1
))
styles.add(ParagraphStyle(
    name="Equation", parent=styles["Code"],
    fontName="Courier", fontSize=7.5, leading=10,
    leftIndent=8, rightIndent=8, spaceBefore=3, spaceAfter=6,
    wordWrap="CJK"
))
styles.add(ParagraphStyle(
    name="TableCell", parent=styles["BodyText"],
    fontSize=6.8, leading=8, splitLongWords=1
))
styles.add(ParagraphStyle(
    name="TableHeader", parent=styles["BodyText"],
    fontSize=7, leading=8, textColor=colors.white
))

def inline_markup(text):
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text

def parse_table(lines):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)

    if not rows:
        return None

    width = max(len(row) for row in rows)
    data = []
    for i, row in enumerate(rows):
        row += [""] * (width - len(row))
        style = styles["TableHeader"] if i == 0 else styles["TableCell"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])

    available = landscape(A4)[0] - 30 * mm
    col_width = available / width
    table = Table(data, colWidths=[col_width] * width, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F5F8")]),
    ]))
    return table

def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.drawString(15 * mm, 9 * mm, "3D Atomic Toy Model — Scientific Computing Report")
    canvas.drawRightString(doc.pagesize[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()

def build():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Report source not found: {SOURCE}")

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0

    while i < len(lines):
        raw = lines[i].strip()

        if not raw or raw == "---":
            story.append(Spacer(1, 3))
            i += 1
            continue

        if raw.startswith("$$"):
            equation = []
            if raw.count("$$") >= 2:
                equation.append(raw.strip("$"))
                i += 1
            else:
                equation.append(raw[2:])
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    equation.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    equation.append(lines[i].replace("$$", "").strip())
                    i += 1
            expr = " ".join(part for part in equation if part)
            story.append(Paragraph(escape(expr), styles["Equation"]))
            continue

        if raw.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            table = parse_table(table_lines)
            if table:
                story.append(Spacer(1, 4))
                story.append(table)
                story.append(Spacer(1, 6))
            continue

        if raw.startswith("# "):
            story.append(Paragraph(inline_markup(raw[2:]), styles["ReportTitle"]))
        elif raw.startswith("## "):
            story.append(Paragraph(inline_markup(raw[3:]), styles["Section"]))
        elif raw.startswith("### "):
            story.append(Paragraph(inline_markup(raw[4:]), styles["Subsection"]))
        elif raw.startswith("- "):
            story.append(Paragraph("&bull;&nbsp; " + inline_markup(raw[2:]), styles["BodyReport"]))
        elif re.match(r"^\d+\.\s", raw):
            story.append(Paragraph(inline_markup(raw), styles["BodyReport"]))
        else:
            story.append(Paragraph(inline_markup(raw), styles["BodyReport"]))

        i += 1

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="Final Scientific Computing Report: 3D Atomic Toy Model",
        author="IIT Hyderabad Atomic Toy Model Project",
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"PDF created: {OUTPUT}")
    print(f"Source: {SOURCE}")
    print(f"Size: {OUTPUT.stat().st_size:,} bytes")

if __name__ == "__main__":
    build()
