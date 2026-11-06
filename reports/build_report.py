"""Build the formal project report DOCX."""

from pathlib import Path
from datetime import date

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports" / "assets"
OUT = ROOT / "reports" / "Document_QA_Failure_Analysis_Project_Report.docx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x75, 0xB6)
TEAL = RGBColor(0x0F, 0x6E, 0x6E)
ORANGE = RGBColor(0xC5, 0x5A, 0x11)
GREEN = RGBColor(0x54, 0x82, 0x35)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT_GREY = RGBColor(0x7F, 0x7F, 0x7F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1F, 0x1F, 0x1F)

NAVY_HEX = "1F3864"
LIGHT_BLUE_HEX = "EAF1F8"
PALE_GREEN_HEX = "F3F6F4"
PALE_ORANGE_HEX = "FBE5D6"
PALE_GREY_HEX = "F2F2F2"
BORDER_HEX = "B4C6E7"

BODY_FONT = "Calibri"
HEAD_FONT = "Cambria"
TABLE_FONT = "Calibri"

FIGURE_COUNTER = 0
TABLE_COUNTER = 0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def set_run_font(run, name=BODY_FONT, size=10.5, bold=None, italic=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade_cell(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tc_pr.append(shd)


def set_cell_margins(cell, top=60, bottom=60, left=90, right=90):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for key, value in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = tc_mar.find(qn(f"w:{key}"))
        if node is None:
            node = OxmlElement(f"w:{key}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=BORDER_HEX, size=4):
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)
        borders.append(element)
    tbl_pr.append(borders)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    tr_pr.append(cant)


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_col_widths(table, widths_cm):
    table.autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths_cm):
            row.cells[idx].width = Cm(width)
    for idx, width in enumerate(widths_cm):
        table.columns[idx].width = Cm(width)


def add_page_field(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)
    set_run_font(run, size=8, color=LIGHT_GREY)


def add_num_pages_field(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " NUMPAGES "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)
    set_run_font(run, size=8, color=LIGHT_GREY)


def page_break(document):
    document.add_page_break()


def add_header_footer(section, title_short):
    header = section.header
    header.is_linked_to_previous = False
    table = header.add_table(rows=1, cols=2, width=Cm(16.6))
    table.autofit = False
    table.columns[0].width = Cm(12.0)
    table.columns[1].width = Cm(4.6)
    left = table.rows[0].cells[0].paragraphs[0]
    left.text = ""
    run = left.add_run(title_short)
    set_run_font(run, size=8, color=LIGHT_GREY)
    left.alignment = WD_ALIGN_PARAGRAPH.LEFT
    right = table.rows[0].cells[1].paragraphs[0]
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = right.add_run("IIT Roorkee  ·  PARAM Ganga")
    set_run_font(run, size=8, color=LIGHT_GREY)
    # Remove table borders and padding.
    for cell in table.rows[0].cells:
        set_cell_margins(cell, 0, 0, 0, 0)
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "nil")
        borders.append(element)
    tbl_pr.append(borders)

    # Thin rule under header
    p = header.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p_pr = p._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), BORDER_HEX)
    p_bdr.append(bottom)
    p_pr.append(p_bdr)

    footer = section.footer
    footer.is_linked_to_previous = False
    f_table = footer.add_table(rows=1, cols=2, width=Cm(16.6))
    f_table.autofit = False
    f_table.columns[0].width = Cm(12.0)
    f_table.columns[1].width = Cm(4.6)
    left = f_table.rows[0].cells[0].paragraphs[0]
    run = left.add_run("Confidential — for faculty review")
    set_run_font(run, size=8, color=LIGHT_GREY)
    right = f_table.rows[0].cells[1].paragraphs[0]
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = right.add_run("Page ")
    set_run_font(run, size=8, color=LIGHT_GREY)
    add_page_field(right)
    run = right.add_run(" of ")
    set_run_font(run, size=8, color=LIGHT_GREY)
    add_num_pages_field(right)
    for cell in f_table.rows[0].cells:
        set_cell_margins(cell, 0, 0, 0, 0)
    tbl_pr = f_table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "nil")
        borders.append(element)
    tbl_pr.append(borders)


# ---------------------------------------------------------------------------
# Document building helpers
# ---------------------------------------------------------------------------

def heading(document, text, level=1, number=None):
    p = document.add_paragraph()
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run((number + "  " if number else "") + text)
        set_run_font(run, HEAD_FONT, 15, bold=True, color=NAVY)
        p_pr = p._p.get_or_add_pPr()
        p_bdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:space"), "4")
        bottom.set(qn("w:color"), "2E75B6")
        p_bdr.append(bottom)
        p_pr.append(p_bdr)
    elif level == 2:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run((number + "  " if number else "") + text)
        set_run_font(run, HEAD_FONT, 11.5, bold=True, color=NAVY)
    else:
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run((number + "  " if number else "") + text)
        set_run_font(run, HEAD_FONT, 10.5, bold=True, color=TEAL)
    return p


def body(document, text="", bold_lead=None, size=10.5, color=DARK, space_after=5, italic=False):
    p = document.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.12
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead:
        run = p.add_run(bold_lead)
        set_run_font(run, size=size, bold=True, color=NAVY)
    run = p.add_run(text)
    set_run_font(run, size=size, color=color, italic=italic)
    return p


def bullet(document, text, bold_lead=None, level=0):
    p = document.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.55 + 0.45 * level)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    if bold_lead:
        run = p.add_run(bold_lead)
        set_run_font(run, size=10.2, bold=True, color=NAVY)
    run = p.add_run(text)
    set_run_font(run, size=10.2)
    return p


def numbered(document, text, bold_lead=None):
    p = document.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.first_line_indent = Cm(-0.4)
    p.paragraph_format.space_after = Pt(3)
    if bold_lead:
        run = p.add_run(bold_lead)
        set_run_font(run, size=10.2, bold=True, color=NAVY)
    run = p.add_run(text)
    set_run_font(run, size=10.2)
    return p


def note_box(document, title, text, fill=LIGHT_BLUE_HEX, border=NAVY_HEX):
    table = document.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Cm(16.4)
    shade_cell(cell, fill)
    set_cell_margins(cell, 140, 140, 180, 180)
    set_table_borders(table, color=border, size=8)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(title)
    set_run_font(run, HEAD_FONT, 10.5, bold=True, color=NAVY)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p2.add_run(text)
    set_run_font(run, size=9.5, color=DARK)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def caption(document, text, kind="Figure"):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(f"{kind} — ")
    set_run_font(run, size=8.5, bold=True, color=BLUE)
    run = p.add_run(text)
    set_run_font(run, size=8.5, italic=True, color=GREY)


def add_figure(document, path, width_cm, fig_caption):
    global FIGURE_COUNTER
    FIGURE_COUNTER += 1
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    caption(document, fig_caption, kind=f"Figure {FIGURE_COUNTER}")


def add_table(document, headers, rows, widths, table_caption, font_size=8.4,
              header_fill=NAVY_HEX, zebra=True, align_center_cols=None):
    global TABLE_COUNTER
    TABLE_COUNTER += 1
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_header(header)
    for idx, text in enumerate(headers):
        cell = header.cells[idx]
        shade_cell(cell, header_fill)
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        set_run_font(run, TABLE_FONT, font_size, bold=True, color=WHITE)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for r_idx, row_values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        for idx, value in enumerate(row_values):
            cell = row.cells[idx]
            set_cell_margins(cell)
            if zebra and r_idx % 2 == 1:
                shade_cell(cell, "F5F8FC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            if align_center_cols and idx in align_center_cols:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            set_run_font(run, TABLE_FONT, font_size, color=DARK)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_col_widths(table, widths)
    caption(document, table_caption, kind=f"Table {TABLE_COUNTER}")
    return table


def formula_box(document, formula, note=None):
    table = document.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Cm(16.2)
    shade_cell(cell, "F7F9FC")
    set_cell_margins(cell, 120, 120, 160, 160)
    set_table_borders(table, color="D6E4F0", size=4)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(formula)
    set_run_font(run, "Cambria Math", 10.5, color=NAVY)
    if note:
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p2.add_run(note)
        set_run_font(run, size=8, italic=True, color=GREY)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def add_toc(document):
    heading(document, "Contents", level=1)
    entries = [
        ("Executive Summary", "Purpose, status, resource request"),
        ("1. Introduction", "Problem, questions, objectives, contributions"),
        ("2. Background and Related Work", "Document QA, OCR/VLM, attribution, Indic scripts"),
        ("3. Research Design and Methodology", "Data, conditions, metrics, statistics"),
        ("4. Phase 1 Implementation Status", "What is built, pilot results, readiness"),
        ("5. Confirmatory Study and Compute Requirements", "Models, runs, PARAM Ganga, timeline"),
        ("6. Faculty Support Requested", "Access, data, annotators, oversight"),
        ("7. Publication and Dissemination Plan", "Venues, authorship, artifact, thesis"),
        ("8. Ethics, Data Governance, and Compliance", "Licences, privacy, responsible release"),
        ("9. Conclusion", "Summary and decision requested"),
        ("References", "Primary sources"),
        ("Appendix A", "Repository map and commands"),
        ("Appendix B", "Manifest and condition schemas"),
        ("Appendix C", "PARAM Ganga execution checklist"),
        ("Appendix D", "Endorsement and signature"),
    ]
    table = document.add_table(rows=1, cols=2)
    table.autofit = False
    set_table_borders(table, color="D6E4F0", size=4)
    table.rows[0].cells[0].width = Cm(7.2)
    table.rows[0].cells[1].width = Cm(9.2)
    for cell, text in zip(table.rows[0].cells, ("Section", "Scope")):
        shade_cell(cell, NAVY_HEX)
        p = cell.paragraphs[0]
        run = p.add_run(text)
        set_run_font(run, size=8.6, bold=True, color=WHITE)
    for idx, (section, scope) in enumerate(entries):
        row = table.add_row()
        row.cells[0].width = Cm(7.2)
        row.cells[1].width = Cm(9.2)
        if idx % 2 == 1:
            for cell in row.cells:
                shade_cell(cell, "F5F8FC")
        for cell, text, bold in (
            (row.cells[0], section, True),
            (row.cells[1], scope, False),
        ):
            set_cell_margins(cell, 50, 50, 90, 90)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(text)
            set_run_font(run, size=8.9, bold=bold, color=NAVY if bold else DARK)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------

doc = Document()
section = doc.sections[0]
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.left_margin = Cm(2.2)
section.right_margin = Cm(2.2)
section.top_margin = Cm(2.0)
section.bottom_margin = Cm(1.9)
section.header_distance = Cm(1.0)
section.footer_distance = Cm(0.9)

style = doc.styles["Normal"]
style.font.name = BODY_FONT
style.font.size = Pt(10.5)
style.font.color.rgb = DARK
style.paragraph_format.space_after = Pt(4)

core = doc.core_properties
core.title = "Where Does Document Understanding Break? Project Report and PARAM Ganga Compute Proposal"
core.author = "Siddhartha (sidg2hp)"
core.subject = "Multilingual document QA failure attribution — project report"
core.keywords = "document QA, OCR, VLM, Indic scripts, error attribution, PARAM Ganga"
core.comments = "Phase 1 complete; Phase 2 confirmatory study requires HPC access."

add_header_footer(section, "Document QA Failure Attribution  ·  Project Report")

# ---------------------------------------------------------------------------
# COVER
# ---------------------------------------------------------------------------
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(28)
p.paragraph_format.space_after = Pt(0)

cover = doc.add_table(rows=1, cols=1)
cover.alignment = WD_TABLE_ALIGNMENT.CENTER
cover.autofit = False
cell = cover.rows[0].cells[0]
cell.width = Cm(16.6)
shade_cell(cell, NAVY_HEX)
set_cell_margins(cell, 260, 260, 240, 240)
set_table_borders(cover, color=NAVY_HEX, size=4)

p = cell.paragraphs[0]
p.paragraph_format.space_after = Pt(10)
run = p.add_run("RESEARCH PROJECT REPORT")
set_run_font(run, size=9, bold=True, color=RGBColor(0xBD, 0xD7, 0xEE))
p2 = cell.add_paragraph()
p2.paragraph_format.space_after = Pt(8)
run = p2.add_run("Where Does Document Understanding Break?")
set_run_font(run, HEAD_FONT, 25, bold=True, color=WHITE)
p3 = cell.add_paragraph()
p3.paragraph_format.space_after = Pt(12)
run = p3.add_run("Counterfactual Error Attribution for Multilingual Indian Document QA")
set_run_font(run, HEAD_FONT, 14.5, italic=True, color=RGBColor(0xD9, 0xE2, 0xF3))
p4 = cell.add_paragraph()
p4.paragraph_format.space_after = Pt(0)
run = p4.add_run("Phase 1 Completion  ·  Confirmatory Study Plan  ·  PARAM Ganga Compute Request")
set_run_font(run, size=9.5, color=RGBColor(0xD9, 0xE2, 0xF3))

doc.add_paragraph().paragraph_format.space_after = Pt(18)

# Cover metadata table
meta = doc.add_table(rows=0, cols=2)
meta.autofit = False
set_table_borders(meta, color="FFFFFF", size=0)
meta_rows = [
    ("Prepared by", "Siddhartha (sidg2hp)"),
    ("Affiliation", "Indian Institute of Technology Roorkee"),
    ("Submitted to", "Faculty Sponsor and PARAM Ganga Resource Allocation Committee"),
    ("Document status", "Version 1.0  ·  Final for faculty review"),
    ("Date", "6 November 2026"),
    ("Project repository", "github.com/sidg2hp/Multilingual-document-QA-failure-analysis"),
]
for label, value in meta_rows:
    row = meta.add_row()
    row.cells[0].width = Cm(4.6)
    row.cells[1].width = Cm(11.8)
    for cell, text, bold in ((row.cells[0], label, True), (row.cells[1], value, False)):
        set_cell_margins(cell, 60, 60, 0, 80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(text)
        set_run_font(run, size=9.5, bold=bold, color=NAVY if bold else DARK)

doc.add_paragraph().paragraph_format.space_after = Pt(18)

note_box(
    doc,
    "Decision requested",
    "Endorsement of PARAM Ganga access for the confirmatory study: one GPU node with 2×V100 16 GB, "
    "up to 300 GPU-hours over 13 weeks, 50 GB scratch storage, and faculty support for data access and "
    "two volunteer bilingual annotators. No paid annotation or additional hardware purchase is requested.",
    fill="F3F6F4",
    border="0F6E6E",
)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Prepared for faculty review  ·  Not for external circulation")
set_run_font(run, size=8.5, italic=True, color=LIGHT_GREY)

page_break(doc)

# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------
heading(doc, "Executive Summary", level=1)

body(
    doc,
    "Indian public administration, finance, health, and citizen services depend on documents whose content is "
    "multilingual, form-like, and frequently degraded by scanning, compression, or mobile capture. Modern "
    "document question answering (DocQA) systems can answer questions over such documents, but an aggregate "
    "score such as ANLS or exact match does not reveal why a system failed. A wrong answer may arise because "
    "the page was blurred, because the answer region was never localised, because an Indic matra was misread, "
    "or because the model could not reason across languages. Each failure calls for a different intervention.",
)
body(
    doc,
    "This project develops a counterfactual, stage-wise measurement framework that separates these failure "
    "sources for two architectures: a cascaded OCR + text-LLM pipeline and an OCR-free vision-language model "
    "(VLM). The framework combines controlled visual degradations, oracle layout and transcript interventions, "
    "a placebo crop, a shuffled-transcript control, a transcript-scored read–reason probe, matra-aware metrics, "
    "document-clustered statistics, and a leakage-free cost-aware router.",
)

heading(doc, "Phase 1 — completed", level=2)
body(
    doc,
    "The protocol, data layer, model runners, evaluation engine, router, reporting pipeline, and reproducibility "
    "infrastructure are implemented and tested. The repository contains 31 incremental commits (at the time of writing), 58 passing "
    "offline tests, a green GitHub Actions build on Python 3.10 and 3.11, a frozen data manifest builder, a "
    "provenance-aware prediction cache, and an analysis pipeline that generates the paper tables and figures "
    "from released predictions. A hardware-constrained pilot on the available RTX 3050 validated the measurement "
    "loop and informed perturbation calibration; it is explicitly a pilot, not the confirmatory study.",
)

heading(doc, "Phase 2 — the confirmatory study", level=2)
body(
    doc,
    "Phase 2 will execute the frozen protocol on PARAM Ganga using 3B and 7B VLMs, Tesseract and PaddleOCR "
    "cascades, and 3B/7B text readers. The target is a 600-item diagnostic set drawn from IndicVisionBench "
    "OCR pages and a Latin-script FUNSD control, with a Patram-Bench extension if access is granted. The study "
    "will produce condition-level accuracy, stage-wise attribution, degradation curves, a read–reason "
    "dissociation analysis, a quantization × script analysis, and a routing Pareto frontier. Estimated demand "
    "is 250 GPU-hours, with a 300 GPU-hour envelope to cover calibration, retries, and the 7B scale check.",
)

heading(doc, "Support requested", level=2)
body(
    doc,
    "The project requests: (i) faculty endorsement for a PARAM Ganga account and 300 GPU-hours on one 2×V100 "
    "node; (ii) a faculty letter supporting the Patram-Bench data-access request; (iii) help recruiting two "
    "bilingual volunteer annotators for a 100-item validation set (approximately 6–8 hours each, no payment); "
    "(iv) faculty oversight of data ethics and release; and (v) mentorship through submission. The expected "
    "outputs are one workshop or Findings-track paper, one thesis chapter, and a reproducible public artifact.",
)

note_box(
    doc,
    "Why this is a good use of institutional compute",
    "The project is compute-bounded, not data-annotation-bounded. The protocol, code, manifests, and analysis "
    "are already built; the remaining cost is GPU time to run a controlled, pre-registered study that produces "
    "a reusable diagnostic benchmark, a public artifact, and a publishable result. A 2×V100 node matches the "
    "FP16 memory requirement of the 7B models and reduces a multi-week laptop experiment to days.",
)

page_break(doc)
add_toc(doc)

heading(doc, "Figures", level=2)
for text in [
    "Figure 1 — End-to-end measurement pipeline",
    "Figure 2 — Architecture-specific counterfactual ladders",
    "Figure 3 — Perturbation calibration pilot",
    "Figure 4 — Controlled pipeline validation",
    "Figure 5 — Estimated compute envelope",
    "Figure 6 — Thirteen-week execution plan",
]:
    bullet(doc, text)

# ---------------------------------------------------------------------------
# 1. INTRODUCTION
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Introduction", level=1, number="1")

heading(doc, "1.1  Context and motivation", level=2)
body(
    doc,
    "Document understanding in India is a multilingual problem. A single citizen-facing workflow may contain "
    "Devanagari, Bengali, Tamil, Telugu, Gujarati, or Odia text, embedded English terms, Devanagari digits, "
    "currency symbols, tables, stamps, and handwritten annotations. Documents are often scanned at low DPI, "
    "re-compressed for messaging platforms, photographed at an angle, or archived as poor-quality photocopies. "
    "The same page can be easy for a human reader and hard for an OCR engine or a VLM.",
)
body(
    doc,
    "Recent DocQA systems follow two broad designs. A cascaded pipeline extracts text with an OCR engine and "
    "then answers with a language model. An OCR-free VLM reads the page image directly and answers end-to-end. "
    "Both can fail, but for different reasons. The cascade can lose information in character recognition or "
    "reading order; the VLM can hallucinate text that it never perceived. Aggregate accuracy hides this "
    "distinction, and a deployment team cannot tell whether to invest in a better OCR engine, a better layout "
    "model, a larger language model, or a different prompt.",
)

heading(doc, "1.2  Problem statement", level=2)
body(
    doc,
    "Given a document image I, a question Q, and a gold answer A*, the project asks how much of the end-to-end "
    "error is attributable to (a) visual degradation, (b) layout/localisation, (c) script and character "
    "recognition, and (d) semantic or cross-lingual reasoning; whether the answer differs between cascaded OCR "
    "and OCR-free VLMs; and whether the pattern changes for Indic abugidas, code-mixed text, and quantized models.",
)

heading(doc, "1.3  Research questions", level=2)
add_table(
    doc,
    ["ID", "Research question", "Primary evidence"],
    [
        ("RQ1", "Which failure stage dominates for OCR cascades versus OCR-free VLMs?", "Architecture-specific counterfactual decomposition with clustered CIs."),
        ("RQ2", "Do Indic VLMs read a span correctly but fail to use it when answering directly?", "C1 transcript CER/MER versus C0 and C1 QA, stratified by script."),
        ("RQ3", "Do recognition errors concentrate in matras relative to base characters under degradation?", "Matra Error Rate versus base-character CER across calibrated severity levels."),
        ("RQ4", "Does 4-bit quantization disproportionately degrade Indic recognition relative to Latin?", "Matched NF4/INT8/FP16 comparison with script interaction."),
        ("RQ5", "Can a text-free image router capture the OCR/VLM oracle gap at lower measured cost?", "Out-of-fold, page-grouped routing with measured latency and bootstrap gain."),
    ],
    widths=[1.4, 8.4, 6.8],
    table_caption="Research questions and the evidence that will answer them.",
    align_center_cols=[0],
)

heading(doc, "1.4  Objectives", level=2)
for text in [
    ("Build a reproducible diagnostic set. ", "Freeze a 600-item manifest with IDs, hashes, dataset/model revisions, and question templates."),
    ("Implement the full counterfactual protocol. ", "Run every supported condition on both architectures, including placebo and sham controls."),
    ("Quantify stage-wise error. ", "Report identified decompositions, clustered confidence intervals, paired tests, and coverage."),
    ("Test one mechanistic claim. ", "Prioritise read–reason dissociation, with matra vulnerability and quantization as secondary claims."),
    ("Deliver an artifact. ", "Release code, prompts, manifests, predictions, analysis scripts, and a reproduction guide."),
]:
    bullet(doc, text[1], bold_lead=text[0])

heading(doc, "1.5  Contributions", level=2)
body(
    doc,
    "The project makes four contributions. First, it defines an architecture-specific counterfactual ladder: the "
    "same five-condition template is not valid for both pipelines because C2 and C3 carry different information "
    "in a cascade. Second, it introduces a transcript-scored read–reason probe, distinguishing a model that "
    "cannot read from one that reads but cannot use what it read. Third, it provides matra-aware metrics and a "
    "confident-hallucination/abstention analysis for Indic scripts. Fourth, it contributes a leakage-free, "
    "cost-aware router evaluated out-of-fold and grouped by document.",
)

heading(doc, "1.6  Scope and non-goals", level=2)
body(
    doc,
    "The study is a measurement and attribution study, not a new benchmark or a new foundation model. It does "
    "not claim to be the first Indian document benchmark; Patram-Bench and IndicVisionBench already exist. It "
    "uses English questions over multilingual documents, which reflects a realistic cross-lingual retrieval "
    "setting but is not a study of in-language question generation. Layout claims are restricted to datasets "
    "with gold boxes; the Indic arm uses generated extractive and cloze questions over real transcripts. The "
    "Latin-script FUNSD control separates script effects from document effects.",
)

heading(doc, "1.7  Report organisation", level=2)
body(
    doc,
    "Section 2 positions the work against prior DocQA, OCR/VLM, attribution, and hallucination research. "
    "Section 3 describes the protocol and statistics. Section 4 documents what is already implemented and the "
    "pilot evidence. Section 5 specifies the confirmatory study, compute envelope, timeline, and risks. Section 6 "
    "lists the support requested from the faculty sponsor. Sections 7–9 cover publication, ethics, and the "
    "conclusion. Appendices provide the repository map, schemas, execution checklist, and endorsement form.",
)

heading(doc, "1.8  Project at a glance", level=2)
add_table(
    doc,
    ["Item", "Current state"],
    [
        ("Phase 1", "Protocol, data layer, model runners, evaluation, router, reporting, and CI complete"),
        ("Engineering evidence", "31 incremental commits (at the time of writing); 58 passing tests; GitHub Actions green on Python 3.10/3.11"),
        ("Data", "IndicVisionBench OCR (876 pages, 10 languages), FUNSD (50 forms), Patram-Bench access request"),
        ("Pilot", "RTX 3050 instrumentation pilot; used for calibration, not hypothesis testing"),
        ("Confirmatory target", "600 items, 3B/7B VLMs, Tesseract/Paddle cascades, all supported conditions"),
        ("Resource request", "1 node × 2×V100, up to 300 GPU-hours, 50 GB scratch, 13 weeks"),
        ("Human validation", "Two bilingual volunteers, 100 items, approximately 6–8 hours each"),
    ],
    widths=[4.4, 12.2],
    table_caption="Project status summary.",
)

heading(doc, "1.9  Expected outcomes", level=2)
for text in [
    ("Scientific: ", "a condition-level and stage-wise account of where multilingual document QA fails, with one mechanistically tested claim."),
    ("Artifact: ", "a reproducible diagnostic harness with manifests, prompts, prediction records, and analysis scripts."),
    ("Training: ", "a thesis chapter and a workshop or Findings submission."),
    ("Institutional: ", "a reusable multilingual document-QA evaluation capability for IIT Roorkee."),
]:
    bullet(doc, text[1], bold_lead=text[0])

# ---------------------------------------------------------------------------
# 2. BACKGROUND
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Background and Related Work", level=1, number="2")

heading(doc, "2.1  Document question answering", level=2)
body(
    doc,
    "DocQA requires locating and reading an answer in a visually rich document, then producing a short span. "
    "Standard benchmarks include DocVQA, InfographicVQA, and FUNSD; multilingual Indian resources include "
    "IndicVisionBench and Patram-Bench. Evaluation usually reports ANLS or exact match. These metrics are useful "
    "for leaderboards but do not identify the stage at which a system lost the answer.",
)

heading(doc, "2.2  OCR cascades versus OCR-free VLMs", level=2)
body(
    doc,
    "A cascade exposes an explicit text interface: OCR output can be inspected, cached, and corrected. It can "
    "be brittle on low-quality scans, complex layout, and combining marks. An OCR-free VLM avoids the hard text "
    "interface but must ground language in pixels; it can produce fluent answers unsupported by the image. "
    "DISCO (Benkirane et al., 2026) compares the two families across document types and finds that the better "
    "choice depends on document characteristics. The present study asks a complementary question: where inside "
    "each family does the error arise?",
)

heading(doc, "2.3  Error attribution", level=2)
body(
    doc,
    "Xu et al. (2026) perform interventional attribution for multi-page visually rich document understanding, "
    "separating representation, evidence selection, and reasoning failures on English documents. Their design "
    "demonstrates the value of controlled interventions but does not cover Indic scripts, OCR cascades, or "
    "abugida-specific recognition errors. This project adapts the intervention logic to two architectures and "
    "to multilingual documents, with explicit negative controls.",
)

heading(doc, "2.4  Perceptual hallucination under degradation", level=2)
body(
    doc,
    "Hwang et al. (Findings of ACL 2026) define perceptual hallucination as generating information as if it were "
    "perceived despite absent or damaged visual evidence. Their DocHallu benchmark shows higher hallucination "
    "rates for numerical content and links hallucinations to vision-encoder errors. The present study measures "
    "the hallucination-to-abstention ratio under controlled degradation and separates confident fabrication from "
    "explicit refusal, while treating the automatic proxy as a screening signal rather than ground truth.",
)

heading(doc, "2.5  Indic scripts and matras", level=2)
body(
    doc,
    "Indic scripts are abugidas: vowel diacritics (matras), virama/halant, and conjunct ligatures carry lexical "
    "and numerical meaning. A one-character edit can change a word, a date, or an amount. Standard CER treats all "
    "characters equally; a matra-aware metric is needed to test whether degradation and quantization destroy "
    "diacritics before base consonants. The Matra Error Rate (MER) in this project restricts combining-mark "
    "extraction to Indic blocks and reports it alongside base-character CER, so a Latin diacritic cannot "
    "contaminate the Indic metric.",
)

page_break(doc)
heading(doc, "2.6  Positioning", level=2)
add_table(
    doc,
    ["Prior work", "What it establishes", "What this project adds"],
    [
        ("DISCO (2026)", "OCR vs VLM performance depends on document type.", "Stage-wise attribution inside each architecture, not only aggregate comparison."),
        ("Xu et al. (2026)", "Interventional attribution for English multi-page VRDU.", "Indic scripts, OCR cascades, abugida-aware metrics, read–reason probe."),
        ("DocHallu (2026)", "Degradation induces perceptual hallucination, especially numerical.", "HAR under a controlled severity spectrum, multilingual documents, OCR vs VLM."),
        ("IndicVisionBench (2026)", "Multilingual cultural VQA and OCR benchmark.", "Uses OCR transcripts as a diagnostic substrate; generated QA is explicitly labelled."),
        ("Patram-Bench (2026)", "Large-scale Indian document benchmark.", "Optional extension; public fallback keeps the study executable."),
    ],
    widths=[3.4, 6.5, 6.7],
    table_caption="Relationship to the closest prior work.",
)

heading(doc, "2.7  Research gap and positioning statement", level=2)
body(
    doc,
    "The literature establishes that OCR and VLMs fail differently across document types, that controlled "
    "interventions can localise failures in English multi-page documents, and that degraded images induce "
    "perceptual hallucination. It does not establish how these findings transfer to Indic abugidas, where matras "
    "and conjuncts carry meaning; nor does it provide a single protocol that compares a cascade and a VLM under "
    "the same conditions with architecture-specific decompositions.",
)
body(
    doc,
    "This project fills that gap as a measurement study. It does not claim a new benchmark or a new model. Its "
    "contribution is the protocol, the abugida-aware metrics, the transcript-scored read–reason probe, and a "
    "reproducible public artifact that other groups can apply to their own documents and models.",
)

heading(doc, "2.8  Summary of the gap", level=2)
body(
    doc,
    "In short, the field has strong benchmarks and strong aggregate comparisons, but it lacks a controlled, "
    "architecture-aware account of failure for Indic documents. This project supplies that account as a measurement "
    "study, with a public protocol and artifact, rather than as another leaderboard entry.",
)

# ---------------------------------------------------------------------------
# 3. METHODOLOGY
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Research Design and Methodology", level=1, number="3")

heading(doc, "3.1  End-to-end measurement pipeline", level=2)
body(
    doc,
    "Figure 1 shows the measurement pipeline. A verified dataset is converted into a frozen manifest; the "
    "manifest fixes sample IDs, page IDs, question templates, answers, hashes, and capability flags. The runner "
    "executes the supported counterfactual conditions, records raw outputs and provenance, and the analysis "
    "layer computes metrics, clustered intervals, paired tests, and stage-wise decompositions.",
)
add_figure(doc, ASSETS / "fig_pipeline.png", 16.0, "End-to-end measurement pipeline, from verified data to stage-wise attribution and routing.")

heading(doc, "3.2  Data sources", level=2)
body(
    doc,
    "The study uses two public sources and one optional benchmark. All sources are verified against their "
    "Hugging Face schemas and pinned revisions. The public fallback is designed so that the study remains "
    "executable even if Patram-Bench access is delayed.",
)
add_table(
    doc,
    ["Dataset", "Content and role", "Size", "Text oracle", "Layout oracle"],
    [
        ("IndicVisionBench `ocr`", "Real Indic document pages with gold transcripts; used for the multilingual recognition and read–reason arm.", "876 pages, 10 languages", "Yes", "No"),
        ("FUNSD test", "English forms with word boxes and BIO entity tags; used as the Latin-script layout control.", "50 documents", "Yes", "Yes"),
        ("Patram-Bench", "ICDAR 2026 Indian document benchmark; optional primary extension subject to author access.", "Access requested", "Yes", "Yes"),
    ],
    widths=[4.2, 7.6, 3.2, 1.8, 1.8],
    table_caption="Data sources, roles, and condition support.",
    align_center_cols=[3, 4],
)
body(
    doc,
    "IndicVisionBench `vqa_indic` is a cultural VQA benchmark over natural images. It is not used as document QA; "
    "doing so would be a category error. The OCR split is the document substrate. Because the OCR split has no "
    "QA pairs and no bounding boxes, the project generates extractive and cloze questions from the gold "
    "transcript and does not run the layout conditions on that arm.",
)

heading(doc, "3.3  Question generation", level=2)
body(
    doc,
    "Questions are generated deterministically from the gold transcript using versioned templates. The generator "
    "extracts unique amount, date, year, identifier, number, and name spans; if a page contains no such span, it "
    "falls back to a cloze item built from a sentence with a unique content word. Uniqueness is required so that "
    "the answer is not ambiguous without the gold box. The generator records a template ID and generator version "
    "for every item.",
)
for text in [
    ("Structured field questions: ", "“What is the total amount stated in the document?”, “What date is stated?”, “What is the reference number?”."),
    ("Cloze reading probe: ", "“Fill in the blank with the exact word from the document: ‘… _____ …’.” This is a reading and grounding probe, not a natural information need."),
    ("FUNSD field questions: ", "“What is the value of ‘<field>’?”, built from BIO question/answer entities with a spatial pairing heuristic and filtered to unique label/answer pairs."),
    ("Verification: ", "a stratified sample of 50 generated items is checked by the author before the confirmatory run; no paid annotation is required for generation."),
]:
    bullet(doc, text[1], bold_lead=text[0])

heading(doc, "3.4  Counterfactual conditions", level=2)
body(
    doc,
    "The protocol uses seven conditions. The same condition label does not always mean the same intervention "
    "across architectures; this is the central methodological point. For example, C2 is a gold-region crop for "
    "a VLM but coordinated gold text for a cascade.",
)
add_table(
    doc,
    ["Condition", "VLM input", "Cascade input", "What it isolates"],
    [
        ("C0", "Clean image + question", "Clean image → OCR → LLM", "Baseline end-to-end performance"),
        ("Cδ", "Degraded image + question", "Degraded image → OCR → LLM", "Visual degradation sensitivity"),
        ("C1", "Transcribe, then answer from own transcript", "Not defined", "Read–reason dissociation (VLM only)"),
        ("C2", "Gold answer-region crop", "Coordinated gold text", "Localisation + clutter removal (VLM); full extraction oracle (cascade)"),
        ("C2b", "Placebo crop of matched size", "Not defined", "Crop/context negative control (VLM)"),
        ("C3", "Gold transcript", "Flat gold transcript", "Recognition oracle"),
        ("C3b", "Shuffled gold transcript", "Shuffled gold transcript", "Reading-order / transcript-content control"),
    ],
    widths=[1.5, 5.2, 5.2, 5.0],
    table_caption="Counterfactual conditions and their architecture-specific meaning.",
    align_center_cols=[0],
)
body(
    doc,
    "C1 is deliberately undefined for the cascade: C0 already performs recognition followed by reasoning, so a "
    "second “read first” prompt would duplicate C0 rather than create an independent intervention. C2 for the "
    "cascade is a full extraction oracle; the layout-representation effect C2_coord − C3_flat is reported as a "
    "separate diagnostic, not added to the error identity.",
)
add_figure(
    doc,
    ASSETS / "fig_ladders.png",
    16.2,
    "Architecture-specific counterfactual ladders. The same condition label carries different information in the two pipelines; C1 is undefined for the cascade.",
)

heading(doc, "3.5  Identified decompositions", level=2)
body(doc, "For an OCR-free VLM, the nested sequence is degraded → clean → gold crop → gold transcript:")
formula_box(
    doc,
    "E_total  =  (C0 − Cδ)  +  (C2 − C0)  +  (C3 − C2)  +  (1 − C3)",
    "perturbation + localisation + recognition + semantic reasoning",
)
body(doc, "For a cascaded OCR + text-LLM pipeline, the identified sequence is degraded → clean → gold transcript:")
formula_box(
    doc,
    "E_total  =  (C0 − Cδ)  +  (C3 − C0)  +  (1 − C3)",
    "perturbation + joint extraction (recognition and reading order) + semantic reasoning",
)
body(
    doc,
    "The cascade's middle term is not split into recognition and order because the available interventions cannot "
    "identify the two separately. Missing conditions are never imputed as zero or as the gold answer; they are "
    "returned as None, excluded from aggregates, and counted in a coverage table. Negative components are retained "
    "as findings: an oracle intervention that hurts accuracy is informative, not a numerical error.",
)

heading(doc, "3.5.1  Identification assumptions", level=2)
body(
    doc,
    "The decomposition assumes that the oracle sequence is nested and that the intervention is the only difference "
    "between adjacent conditions. Prompts, decoding parameters, image resolution limits, and answer parsing are "
    "held constant; only the supplied information changes. The placebo crop controls for the non-specific benefit "
    "of removing page clutter, and the shuffled transcript controls for the benefit of having any text at all.",
)
body(
    doc,
    "The cascade's extraction term is intentionally left joint. Splitting recognition from reading order would "
    "require replacing OCR characters while preserving OCR layout, which is not identifiable without a second "
    "independent layout annotation. The paper reports the joint term and the separate coordinate-text diagnostic, "
    "rather than overstating what the intervention identifies.",
)

page_break(doc)
heading(doc, "3.6  Models and baselines", level=2)
add_table(
    doc,
    ["Role", "Model / engine", "Precision", "Purpose"],
    [
        ("Primary VLM", "Qwen2.5-VL-3B-Instruct", "NF4, INT8", "Main OCR-free arm on 6–16 GB devices"),
        ("Scale VLM", "Qwen2.5-VL-7B-Instruct", "FP16", "Scale check on 2×V100"),
        ("Text reader", "Qwen2.5-3B-Instruct", "NF4", "Cascade reader"),
        ("Text reader scale check", "Qwen2.5-7B-Instruct", "FP16", "Reading-scale check"),
        ("OCR engines", "Tesseract 5, PaddleOCR PP-OCRv4, Surya", "CPU/GPU", "Traditional and neural cascades"),
        ("Secondary VLM", "SmolVLM2-2.2B / InternVL2.5-8B", "FP16/4-bit", "Architecture robustness (optional)"),
    ],
    widths=[4.0, 6.0, 3.0, 4.0],
    table_caption="Planned model matrix. Exact revisions and licences are pinned in the repository.",
)
body(
    doc,
    "Qwen2.5-VL is chosen because it has strong document and multilingual capability at 3B and 7B and fits the "
    "available memory envelope. Tesseract and PaddleOCR provide complementary failure profiles: Tesseract is a "
    "classical engine with known Indic language packs; PaddleOCR has stronger detection and neural recognition. "
    "Surya is an optional third engine if installation is stable. The study does not depend on a proprietary API.",
)

heading(doc, "3.7  Perturbation spectrum", level=2)
body(
    doc,
    "Five families are applied at five levels: Gaussian blur, salt-and-pepper noise, JPEG re-compression, "
    "resolution downscaling, and perspective skew. The levels are calibrated so that induced OCR CER covers "
    "roughly 5–60%, not fixed by an arbitrary pixel radius. The calibration is a manipulation check, not a "
    "hyperparameter search; the final table is frozen before the confirmatory run.",
)

heading(doc, "3.8  Metrics", level=2)
add_table(
    doc,
    ["Metric", "Definition and use"],
    [
        ("ANLS", "Normalized Levenshtein similarity with the standard 0.5 cutoff; primary QA metric."),
        ("Exact Match (EM)", "Normalized exact match against any accepted answer."),
        ("Field F1", "Token-level F1, used for form field extraction."),
        ("CER / WER", "Character and word error rate for OCR output against the gold transcript."),
        ("Base CER", "CER after removing combining marks; separates consonant errors from diacritic errors."),
        ("Matra Error Rate (MER)", "Edit distance on Indic combining marks only; Latin diacritics excluded."),
        ("HAR", "Confident hallucinations divided by explicit abstentions, reported with counts and a saturation flag."),
        ("Latency", "Measured wall-clock milliseconds per condition; used for the router cost model."),
    ],
    widths=[3.6, 13.0],
    table_caption="Evaluation metrics.",
)
body(
    doc,
    "ANLS is reported with conservative normalization by default. Numeric or punctuation canonicalization is an "
    "explicit secondary analysis and is labelled, never silently applied. The hallucination proxy is treated as "
    "a screening signal; causal claims about “matra-blind hallucination” require human inspection.",
)

heading(doc, "3.9  Statistical analysis", level=2)
body(
    doc,
    "The item is the unit of observation, but multiple questions can share a page. All intervals therefore use a "
    "document-clustered percentile bootstrap. Paired comparisons use a paired bootstrap, a sign-flip permutation "
    "test, and McNemar's test on exact correctness. Effect sizes and minimum detectable effects are reported for "
    "every cell. The confirmatory family is Holm-corrected; exploratory slices are labelled as exploratory.",
)
for text in [
    ("Clustered bootstrap: ", "10,000 resamples by page ID for means and paired differences."),
    ("Paired tests: ", "permutation and McNemar for condition contrasts; Wilcoxon as a sensitivity check."),
    ("Multiplicity: ", "Holm–Bonferroni over the pre-registered confirmatory family; no correction for exploratory slices."),
    ("Power: ", "minimum detectable effect reported per language/script cell; cells below 30 items are descriptive."),
    ("Missing conditions: ", "excluded with explicit coverage; never imputed as zero or as the gold answer."),
]:
    bullet(doc, text[1], bold_lead=text[0])

heading(doc, "3.10  Human validation", level=2)
body(
    doc,
    "A 100-item failure sample is stratified by script and model. Two bilingual volunteers label each case as "
    "correct, abstention, confident hallucination, or ambiguous; disagreements are adjudicated. Cohen's kappa is "
    "reported. The protocol requires approximately 6–8 hours per volunteer and does not require payment. If "
    "volunteers are unavailable, the paper reports an author audit and removes any claim of independent human "
    "validation.",
)

heading(doc, "3.11  Reproducibility and provenance", level=2)
body(
    doc,
    "Every prediction records the model ID and revision, quantization, prompt hash, dataset revision, image hash, "
    "parser version, latency, token counts, and raw output. The manifest records sample and page IDs, question "
    "templates, answers, content hashes, and capability flags. The cache key includes all of these fields, so a "
    "prompt or model change cannot silently reuse stale predictions. The report records the git commit and config "
    "hash. The analysis script refuses synthetic predictions unless explicitly overridden.",
)

heading(doc, "3.12  Threats to validity and safeguards", level=2)
body(
    doc,
    "Four threats are managed explicitly. Construct validity: generated questions may not reflect natural "
    "information needs, so the study uses real gold transcripts and a Latin-script field-extraction control, and "
    "labels generated items. Internal validity: prompt differences could confound conditions, so the system "
    "instruction and answer rule are held constant and only the intervention context changes. External validity: "
    "3B and 7B models do not represent every deployment, so the design includes a 7B FP16 scale check and a "
    "secondary VLM where feasible. Statistical validity: multiple questions can share a page, so all intervals "
    "are clustered by page, paired tests are used, and the confirmatory family is Holm-corrected.",
)
body(
    doc,
    "A separate risk is data access. The study is designed so that the public arm is executable without "
    "Patram-Bench, and the paper will state which arm was used. If the Patram extension is granted, it is reported "
    "as an extension, not silently mixed with the public results.",
)

# ---------------------------------------------------------------------------
# 4. IMPLEMENTATION
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Phase 1 Implementation Status", level=1, number="4")

heading(doc, "4.1  Repository architecture", level=2)
add_table(
    doc,
    ["Module", "Responsibility", "Status"],
    [
        ("src/data", "Schema, verified adapters, question generation, local cache, synthetic validation, manifests", "Complete"),
        ("src/models", "HF VLM/OCR runners, Ollama pilot runner, prompts, answer parsing, Windows OCR adapter, rule lower bound", "Complete for Phase 1"),
        ("src/evaluation", "Metrics, architecture-specific attribution, clustered statistics", "Complete"),
        ("src/routing", "Text-free features, leakage guard, measured-cost router, grouped CV", "Complete"),
        ("src/perturbations", "Five degradation families, oracle/placebo crops, text operations", "Complete"),
        ("scripts", "Experiment runner, manifest builder, provenance audit, analysis, figures, OCR pilot", "Complete"),
        ("tests", "58 offline tests covering metrics, statistics, data, cache, prompts, router, reporting", "Passing"),
    ],
    widths=[3.2, 10.6, 2.8],
    table_caption="Repository architecture and implementation status.",
)

heading(doc, "4.2  Data and provenance layer", level=2)
body(
    doc,
    "The data layer targets verified schemas rather than assumed ones. IndicVisionBench OCR, IndicVisionBench "
    "VQA, and FUNSD are pinned to revisions and validated by a live provenance audit. Patram-Bench is handled "
    "through a local-export adapter and fails loudly when the public path is unavailable; it never silently "
    "substitutes a different dataset. Capability flags (`supports_text_oracle`, `supports_layout_oracle`) gate "
    "the conditions, so a dataset without a transcript cannot run C3.",
)

heading(doc, "4.3  Model runners and conditions", level=2)
body(
    doc,
    "The Hugging Face runners support 4-bit NF4, 8-bit INT8, and FP16, with model-family-specific loading for "
    "Qwen2.5-VL and fallbacks for SmolVLM/InternVL. The Ollama runner provides a local pilot path with the same "
    "condition interface. Unsupported conditions raise an explicit error; they never fall back to C0. The "
    "cascade runner uses OCR text in the reader prompt and the VLM runner uses image prompts, with separate "
    "prompt IDs so the two cannot be confused.",
)

heading(doc, "4.4  Evaluation and reporting", level=2)
body(
    doc,
    "The attribution engine validates the nested condition sequence, computes signed components, and reports an "
    "identity check. Summaries include condition coverage, failure modes, HAR counts, and per-script/per-language "
    "breakdowns. The analysis script produces model × condition tables with clustered CIs, degradation curves, "
    "and paired comparisons joined by sample ID. The figure script reads only the analysis outputs, so a figure "
    "cannot diverge from a table.",
)

heading(doc, "4.5  Quality assurance", level=2)
add_table(
    doc,
    ["Check", "Result"],
    [
        ("Offline test suite", "58 tests passing (metrics, statistics, data adapters, cache, prompts, router, reporting)"),
        ("Continuous integration", "GitHub Actions green on Python 3.10 and 3.11"),
        ("Smoke pipeline", "Runs end-to-end; report is explicitly `reportable: false`"),
        ("Provenance audit", "Dataset/model revisions and access status checked against live APIs"),
        ("Cache invalidation", "Prompt, model revision, dataset revision, and image hash included in the key"),
        ("Synthetic guard", "Analysis refuses synthetic predictions unless explicitly overridden"),
    ],
    widths=[5.0, 11.6],
    table_caption="Phase 1 quality gates.",
)

heading(doc, "4.6  Pilot evidence", level=2)
body(
    doc,
    "A hardware-constrained pilot was run on the available RTX 3050 to validate the measurement loop, not to test "
    "the hypotheses. The pilot used an English-only Windows OCR engine and a 1.5B reader; it is deliberately not "
    "the confirmatory stack. Its purpose was to verify that the conditions separate, that the cache and report "
    "plumbing work, and that the perturbation spectrum produces measurable signal. Absolute pilot values are "
    "engine-specific and are not reported as paper findings.",
)
add_figure(
    doc,
    ASSETS / "fig_ocr_calibration.png",
    16.2,
    "Perturbation calibration pilot: relative CER increase over clean on 20 FUNSD documents. Blur and downscaling saturate; JPEG and impulse noise degrade gradually; perspective skew requires redesign before the confirmatory run.",
)
add_figure(
    doc,
    ASSETS / "fig_synthetic_validation.png",
    13.2,
    "Controlled pipeline validation on 40 rendered form items. The oracle conditions reach the ceiling and severe degradation collapses; this validates the measurement loop, not the research hypothesis.",
)
body(
    doc,
    "The pilot surfaced several subtle confounds that have been removed from the final protocol: the cascade "
    "reader prompt now receives OCR text rather than an image instruction; C3 cannot fall back to the gold "
    "answer; the VLM and cascade use separate decompositions; the router cannot use transcript-derived features; "
    "and the cache is keyed by prompt and revision. These are engineering hardening steps, and the confirmatory "
    "study starts from the corrected implementation.",
)

heading(doc, "4.7  Readiness assessment", level=2)
add_table(
    doc,
    ["Readiness dimension", "Status", "Evidence"],
    [
        ("Protocol", "Ready", "Pre-registration, condition table, decomposition, negative controls"),
        ("Data layer", "Ready for public arm; Patram pending", "Verified adapters, manifests, audit"),
        ("Model runners", "Ready for 3B/7B HF; local pilot validated", "NF4/INT8/FP16 paths, capability guards"),
        ("Evaluation", "Ready", "Metrics, clustered statistics, attribution"),
        ("Router", "Ready", "Leakage guard, measured cost, page-grouped CV"),
        ("Reporting", "Ready", "Tables, figures, provenance, synthetic guard"),
        ("Confirmatory evidence", "Pending HPC", "Phase 2"),
    ],
    widths=[4.2, 5.2, 7.2],
    table_caption="Phase 1 readiness against the confirmatory study.",
)

heading(doc, "4.8  From readiness to execution", level=2)
body(
    doc,
    "Three prerequisites remain before the confirmatory run: the PARAM Ganga allocation, the frozen data manifest "
    "for the selected arm, and the calibration table for the perturbation spectrum. The first is institutional; "
    "the other two are already scripted and can be completed in the first two weeks. The engineering work is "
    "therefore not the bottleneck. The bottleneck is access to the hardware and, for the preferred arm, to the "
    "Patram-Bench release.",
)
body(
    doc,
    "The execution order is deliberate: run FUNSD first as an integration test, then the 3B VLM, then the 7B "
    "scale check, then the OCR cascades and the sweep. Each stage has a report and a go/no-go check. A failure at "
    "any stage is diagnosed against the smoke test and the manifest, not by rerunning the whole matrix.",
)

heading(doc, "4.9  Immediate next actions", level=2)
for text in [
    ("Week 1: ", "confirm the PARAM Ganga allocation and submit the Patram-Bench access letter."),
    ("Week 1–2: ", "freeze the public manifest and the perturbation calibration table; verify 50 generated items."),
    ("Week 3–4: ", "stage the models and run the FUNSD integration test; publish an internal milestone report."),
    ("Week 4 onward: ", "start the 3B VLM main run; only then schedule the 7B scale check and the sweep."),
]:
    bullet(doc, text[1], bold_lead=text[0])

# ---------------------------------------------------------------------------
# 5. CONFIRMATORY STUDY / COMPUTE
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Confirmatory Study and Compute Requirements", level=1, number="5")

heading(doc, "5.1  Study matrix", level=2)
body(
    doc,
    "The confirmatory study targets 600 items: 500 Indic pages across at least five languages and 100 FUNSD "
    "items as the Latin-script layout control. Every supported condition is run on every item. The perturbation "
    "sweep applies five families at five levels to Cδ, with the perturbation-independent conditions computed once "
    "per item.",
)
add_table(
    doc,
    ["Block", "Items", "Conditions", "Purpose"],
    [
        ("Indic public arm", "500", "C0, Cδ, C1, C3, C3b", "Multilingual recognition, read–reason, degradation"),
        ("FUNSD control", "100", "C0, Cδ, C1, C2, C2b, C3, C3b", "Layout, crop placebo, Latin control"),
        ("Patram extension", "up to 300", "All supported", "Optional if access is granted"),
        ("Perturbation sweep", "600", "Cδ × 5 families × 5 levels", "Severity curves and calibration"),
    ],
    widths=[4.0, 2.4, 6.4, 3.8],
    table_caption="Confirmatory study blocks.",
    align_center_cols=[1],
)

heading(doc, "5.2  Compute envelope", level=2)
body(
    doc,
    "The estimate assumes 600 items, 7 VLM conditions plus a 25-setting Cδ sweep, one 3B model at two precisions, "
    "one 7B FP16 model, and two OCR cascades. It includes model loading, retries, calibration, and analysis "
    "overheads. The requested envelope is 300 GPU-hours, leaving approximately 20% headroom over the 250-hour "
    "working estimate.",
)
add_table(
    doc,
    ["Workload", "Calls / scope", "Estimated GPU-hours"],
    [
        ("3B VLM, NF4", "600 items × 32 condition-settings", "45"),
        ("3B VLM, INT8", "600 items × 32 condition-settings", "40"),
        ("7B VLM, FP16", "600 items × 32 condition-settings", "70"),
        ("OCR cascades + text readers", "600 items × 28 settings × 2 readers", "20"),
        ("Sweep, retries, calibration, analysis", "—", "75"),
        ("Total working estimate", "", "250"),
        ("Requested envelope", "20% contingency", "300"),
    ],
    widths=[6.6, 7.2, 2.8],
    table_caption="GPU-hour estimate and requested envelope.",
    align_center_cols=[2],
)
add_figure(doc, ASSETS / "fig_compute.png", 13.8, "Estimated compute envelope by workload; total working estimate 250 GPU-hours, request 300 GPU-hours.")

heading(doc, "5.3  PARAM Ganga technical requirements", level=2)
add_table(
    doc,
    ["Requirement", "Specification", "Justification"],
    [
        ("GPU node", "1 node, 2× NVIDIA Tesla V100 16 GB (32 GB total)", "7B FP16 weights plus activations and KV cache"),
        ("CPU / RAM", "12 CPUs, 64 GB host RAM", "Data loading, OCR, preprocessing, analysis"),
        ("Storage", "50 GB scratch + shared HF cache", "Models, page images, predictions, caches"),
        ("Job length", "24 h per job, checkpoint/resume", "Preemptible queue; cache allows restart"),
        ("Access window", "13 weeks", "Data, integration, runs, analysis, writing"),
        ("Software", "CUDA 12.x, PyTorch 2.x, transformers, bitsandbytes, qwen-vl-utils, Tesseract 5, PaddleOCR", "Pinned in the repository"),
        ("Scheduler", "Slurm `gpu` partition, `--gres=gpu:2`, job arrays", "Matches PARAM Ganga conventions"),
    ],
    widths=[3.8, 7.8, 5.0],
    table_caption="PARAM Ganga resource specification.",
)
body(
    doc,
    "The V100 is Volta (sm_70), so the profile uses FP16 and PyTorch SDPA; it does not enable native bfloat16 or "
    "FlashAttention-2. The 7B model is sharded across the two 16 GB GPUs with `device_map=\"auto\"`. A 24-hour job "
    "limit is sufficient because the prediction cache is resumable and per-model shards are independent.",
)

heading(doc, "5.4  Data access and fallback", level=2)
body(
    doc,
    "Patram-Bench is a recent ICDAR 2026 benchmark and is the preferred primary data source. An access request "
    "is prepared and will be submitted to the authors with faculty endorsement. In parallel, the public fallback is frozen: "
    "IndicVisionBench OCR for the multilingual arm and FUNSD for the layout control. The fallback is scientifically "
    "defensible but weaker on natural QA and layout; the paper will state exactly which arm is used.",
)

heading(doc, "5.5  Human validation", level=2)
body(
    doc,
    "Two bilingual volunteers are requested for a 100-item failure validation. The task is short (approximately "
    "6–8 hours per volunteer) and does not require payment. The validation supports the HAR analysis and the "
    "qualitative error taxonomy. If volunteers are not available, the paper reports an author audit and removes "
    "any claim of independent human validation; the read–reason claim does not depend on this annotation.",
)

heading(doc, "5.6  Timeline", level=2)
add_figure(doc, ASSETS / "fig_timeline.png", 16.0, "Thirteen-week execution plan after PARAM Ganga access.")

heading(doc, "5.7  Milestones and deliverables", level=2)
add_table(
    doc,
    ["Milestone", "Week", "Deliverable", "Acceptance criterion"],
    [
        ("M1  Frozen data + calibration", "W3", "Manifest, perturbation table, verification sample", "Hashes frozen; induced CER in target bands"),
        ("M2  FUNSD integration", "W4", "End-to-end report on 100 items", "All supported conditions present; no missing-condition imputation"),
        ("M3  3B VLM results", "W6", "Condition table + decomposition", "Coverage ≥ 95%; clustered CIs computed"),
        ("M4  7B / quantization", "W9", "Scale and precision comparison", "Matched prompts and decoding; seed variance reported"),
        ("M5  Validation + analysis", "W11", "Human labels, final tables, figures", "Agreement reported; every number traceable"),
        ("M6  Paper + artifact", "W13", "Workshop/Findings submission, public artifact", "Reproduction from clean environment succeeds"),
    ],
    widths=[4.2, 1.4, 5.8, 5.2],
    table_caption="Milestones, deliverables, and acceptance criteria.",
    align_center_cols=[1],
)

heading(doc, "5.8  Success criteria", level=2)
for text in [
    ("Protocol fidelity: ", "all supported conditions run with ≥95% item coverage; no gold-answer fallback."),
    ("Statistical completeness: ", "clustered CIs, paired tests, effect sizes, and MDE for every headline cell."),
    ("Mechanistic claim: ", "at least one pre-registered claim with a direction, effect size, and CI that survives Holm correction."),
    ("Reproducibility: ", "a clean environment can regenerate every table and figure from the released predictions."),
    ("Dissemination: ", "a workshop or Findings submission and a thesis chapter."),
]:
    bullet(doc, text[1], bold_lead=text[0])

heading(doc, "5.9  Risk register", level=2)
add_table(
    doc,
    ["Code", "Risk", "L", "I", "Mitigation"],
    [
        ("A", "Patram-Bench access delayed", "M", "H", "Faculty letter; public fallback frozen; paper states scope"),
        ("B", "PARAM Ganga queue delay", "M", "M", "24-hour resumable jobs; cache; prioritise 3B runs first"),
        ("C", "Model compatibility / OOM", "L", "M", "Pin revisions; FP16/SDPA; 3B fallback; CI smoke test"),
        ("D", "Generated-question validity", "M", "M", "Unique-answer constraint; 50-item verification; report limitation"),
        ("E", "Annotator availability", "L", "L", "Two volunteers; author-audit fallback; no paid annotation"),
        ("F", "Null or weak effect", "M", "M", "Pre-registered alternative claims; report null honestly"),
        ("G", "Reproducibility drift", "V", "H", "Manifests, hashes, CI, revision pins"),
    ],
    widths=[1.2, 4.2, 1.4, 1.4, 8.4],
    table_caption="Risk register. L = likelihood, I = impact, V = very low.",
    align_center_cols=[0, 2, 3],
)
body(
    doc,
    "The risk register is reviewed at each milestone. Codes A–G are referenced in the project tracker; the "
    "matrix in the working notes uses the same codes. The two critical-path risks are data access (A) and GPU "
    "queue time (B); both have executable fallbacks.",
)

# ---------------------------------------------------------------------------
# 6. SUPPORT
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Faculty Support Requested", level=1, number="6")

body(
    doc,
    "The project is ready to move from protocol and pilot to the confirmatory study. The remaining requirements "
    "are institutional access and a small amount of volunteer time; there is no request for paid annotation or "
    "new hardware. The table below separates critical path items from supporting items.",
)
add_table(
    doc,
    ["#", "Request", "Why it is needed", "Effort / cost", "Timing"],
    [
        ("1", "Endorse PARAM Ganga access: 1 node, 2×V100, up to 300 GPU-hours, 50 GB scratch, 13 weeks", "The 7B FP16 model and the full condition matrix cannot run on the 6 GB laptop", "Institutional compute only", "Immediate"),
        ("2", "Faculty letter supporting the Patram-Bench data-access request", "Patram is the preferred primary Indian document benchmark", "One letter", "Week 1"),
        ("3", "Help recruit two bilingual volunteer annotators (Hindi/Bengali/Tamil)", "100-item failure validation for HAR and qualitative analysis", "≈6–8 h per volunteer, unpaid", "Week 9"),
        ("4", "Faculty oversight of data ethics, licensing, and release", "Public datasets and model licences; responsible artifact release", "Review of a one-page protocol", "Week 1"),
        ("5", "Mentorship and co-authorship through submission", "Venue selection, framing, and revision", "Normal supervision", "Ongoing"),
        ("6", "Institutional storage/DOI for the artifact", "Long-term access to manifest, predictions, and analysis", "Small archive", "Week 13"),
    ],
    widths=[0.8, 6.4, 6.0, 3.4, 1.8],
    table_caption="Support requested from the faculty sponsor.",
    align_center_cols=[0, 4],
)
body(
    doc,
    "The critical path is items 1 and 2. If Patram access is not granted within two weeks, the public fallback is "
    "ready and the study proceeds without it. The compute request is the only item that cannot be substituted: "
    "the 7B FP16 scale check and the full 600-item matrix are not feasible on the laptop.",
)

heading(doc, "6.1  Why PARAM Ganga is the right resource", level=2)
for text in [
    ("Memory fit: ", "2×V100 16 GB provides 32 GB, enough for 7B FP16 with sharded weights and activations."),
    ("Throughput: ", "the full matrix is approximately 250 GPU-hours; a shared laptop would require weeks and cannot run 7B FP16."),
    ("Reproducibility: ", "a fixed node profile makes latency and memory measurements comparable across runs."),
    ("Institutional alignment: ", "the study targets Indian multilingual documents and contributes a reusable artifact to IIT Roorkee."),
]:
    bullet(doc, text[1], bold_lead=text[0])

heading(doc, "6.2  What the faculty sponsor receives", level=2)
body(
    doc,
    "The sponsor receives a complete, pre-registered study rather than an exploratory request: a tested codebase, "
    "a frozen protocol, a data manifest, a compute plan with a defensible estimate, a risk register, and a clear "
    "publication target. The project produces a workshop/Findings submission, a thesis chapter, and an open "
    "artifact. Progress reports will be provided at each milestone in the milestones table.",
)

# ---------------------------------------------------------------------------
# 7. PUBLICATION
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Publication and Dissemination Plan", level=1, number="7")

heading(doc, "7.1  Target venues", level=2)
body(
    doc,
    "The primary target is a strong document-intelligence or multilingual workshop (ICDAR workshops, WACI, "
    "CVPR VDU, ACL/EMNLP Document Intelligence) or the Findings track. A main-track submission is a stretch goal "
    "and depends on the strength and replication of the mechanistic claim. The paper will be written around one "
    "claim, not around the framework: read–reason dissociation, matra vulnerability, quantization × script, or "
    "routing gain.",
)
add_table(
    doc,
    ["Output", "Venue / form", "Timing"],
    [
        ("Main paper", "Document-intelligence or multilingual workshop, or Findings", "After M6"),
        ("Artifact", "Public repository + archived release with DOI", "With submission"),
        ("Thesis chapter", "IIT Roorkee thesis", "Same semester"),
        ("Faculty report", "This document + milestone updates", "Now and at M3/M5"),
    ],
    widths=[4.0, 9.0, 3.6],
    table_caption="Dissemination outputs.",
)

heading(doc, "7.2  Authorship and contributions", level=2)
body(
    doc,
    "Authorship will follow standard contribution criteria. The student leads protocol design, implementation, "
    "experiments, analysis, and writing. The faculty sponsor contributes supervision, framing, ethics oversight, "
    "and revision. Any additional collaborator who contributes to annotation, analysis, or infrastructure will be "
    "included according to the same criteria.",
)

heading(doc, "7.3  Artifact", level=2)
body(
    doc,
    "The artifact contains code, prompts, manifests, prediction JSONL, analysis scripts, figure scripts, and a "
    "reproduction guide. It does not redistribute restricted page images. Dataset access instructions and "
    "licences are documented. A clean-environment reproduction is a submission requirement, not an afterthought.",
)

heading(doc, "7.4  Review and submission workflow", level=2)
body(
    doc,
    "The paper will be written in four passes: (i) results tables and figures are frozen from the analysis script; "
    "(ii) the method and protocol sections are written against the pre-registration; (iii) the introduction and "
    "related work are positioned against DISCO, Xu et al., DocHallu, IndicVisionBench, and Patram-Bench; (iv) the "
    "limitations, ethics, and artifact sections are completed before internal review. The faculty sponsor reviews "
    "the draft at two points: after the main results table exists and before submission.",
)

# ---------------------------------------------------------------------------
# 8. ETHICS
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Ethics, Data Governance, and Compliance", level=1, number="8")

heading(doc, "8.1  Data", level=2)
body(
    doc,
    "The study uses public research datasets. IndicVisionBench and FUNSD have their own licences and terms; "
    "Patram-Bench will be used only under the terms granted by its authors. Page images are not redistributed; "
    "the artifact releases sample IDs, prompts, predictions, and analysis. No new personally identifiable "
    "information is collected. Forms in FUNSD are public research data; any examples in the paper will be "
    "redacted where necessary.",
)

heading(doc, "8.2  Human participants", level=2)
body(
    doc,
    "The validation task asks volunteers to label model outputs, not to disclose personal information. "
    "Participation is voluntary, uncompensated, and limited to publicly available document content. Volunteers "
    "may stop at any time. The protocol will be reviewed by the faculty sponsor before recruitment.",
)

heading(doc, "8.3  Model and compute use", level=2)
body(
    doc,
    "Model licences are recorded in the repository. Qwen2.5 models are used under their research licence; "
    "Tesseract and PaddleOCR under their open-source licences; Patram under its gated Apache-2.0 terms if access "
    "is granted. PARAM Ganga usage will follow the centre's acceptable-use policy, with no crypto-mining, no "
    "unrelated workloads, and no storage of sensitive data.",
)

heading(doc, "8.4  Responsible reporting", level=2)
body(
    doc,
    "The paper will report negative and null results, condition coverage, exclusions, and limitations. It will "
    "not claim a new benchmark, a first attribution study, or human-validated hallucination labels unless those "
    "claims are supported. Synthetic and pilot results will be labelled as such; they will not be mixed with "
    "confirmatory results.",
)

# ---------------------------------------------------------------------------
# 9. CONCLUSION
# ---------------------------------------------------------------------------
heading(doc, "Conclusion", level=1, number="9")
body(
    doc,
    "The project has completed its protocol, implementation, and pilot phase. It now has a tested measurement "
    "harness, a verified public data layer, a pre-registered analysis plan, a reproducible reporting pipeline, "
    "and a clear path to a confirmatory study. The unresolved question is empirical, not architectural: which "
    "failure stage dominates, and whether Indic OCR-free models exhibit a read–reason dissociation that can be "
    "measured and acted on.",
)
body(
    doc,
    "Answering that question requires the 3B/7B models and the full condition matrix that PARAM Ganga can run. "
    "The requested resources are modest relative to the output: 300 GPU-hours, one 2×V100 node, 50 GB of storage, "
    "a faculty letter for data access, and two volunteers for a short validation task. The project is ready to "
    "start as soon as access is granted.",
)
note_box(
    doc,
    "Decision requested",
    "Approve PARAM Ganga access (1 node, 2×V100, up to 300 GPU-hours, 13 weeks, 50 GB) and endorse the Patram-Bench "
    "data-access letter and the two-volunteer validation plan. With this support, the confirmatory study can begin "
    "immediately and the paper can be submitted within the 13-week plan.",
    fill="F3F6F4",
    border="0F6E6E",
)

# ---------------------------------------------------------------------------
# REFERENCES
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "References", level=1)
refs = [
    "Benkirane, K., Goldwater, D., Asenov, M., & Ghodsi, A. (2026). DISCO: Document Intelligence Suite for COmparative Evaluation. arXiv:2603.23511. ICLR 2026 Workshop on Multimodal Intelligence.",
    "Xu, L., Ding, Y., Xu, Z., Su, D. Y., Liu, D., Luo, S., Peng, Y., & Liu, W. (2026). Locating Failure in Multi-Page Visually Rich Document Understanding: An Empirical Attribution. arXiv:2608.07943.",
    "Hwang, T., Heo, I., Lee, S. J., & Jung, S. (2026). Perceptual Hallucination in Vision–Language Models: Definition, Analysis and Verification. Findings of ACL 2026, pp. 24710–24725.",
    "Faraz, A., Akash, Khan, S., Kolla, R., Patidar, A., Goswami, S., Ravi, A., Khatri, C., & Agarwal, S. (2025/2026). IndicVisionBench: Benchmarking Cultural and Multilingual Understanding in VLMs. arXiv:2511.04727. ICLR 2026.",
    "Patram-Bench: A Multi-task, Multi-domain and Multi-lingual Benchmark for Indian Document Image Understanding. ICDAR 2026. (Full author list and DOI to be taken from the camera-ready.)",
    "Jaume, G., Ekenel, H. K., & Thiran, J.-P. (2019). FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. ICDAR Workshops.",
    "Mathew, M., Karatzas, D., & Jawahar, C. V. (2021). DocVQA: A Dataset for VQA on Document Images. WACV.",
    "Biten, A. F., Tito, R., Mafla, A., Gomez, L., Rusinol, M., Jawahar, C. V., Valveny, E., & Karatzas, D. (2019). Scene Text Visual Question Answering. ICCV. (ANLS metric.)",
    "Qwen Team. (2025). Qwen2.5-VL Technical Report. arXiv:2502.13923.",
    "Smith, R. (2007). An Overview of the Tesseract OCR Engine. ICDAR.",
    "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics.",
    "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
]
for idx, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(f"[{idx}]  ")
    set_run_font(run, size=9.2, bold=True, color=NAVY)
    run = p.add_run(ref)
    set_run_font(run, size=9.2)

# ---------------------------------------------------------------------------
# APPENDIX A
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Appendix A — Repository Map and Commands", level=1)
body(
    doc,
    "The repository is public at github.com/sidg2hp/Multilingual-document-QA-failure-analysis. The table lists the "
    "main entry points used in the study.",
)
add_table(
    doc,
    ["Path", "Purpose"],
    [
        ("configs/experiment_config.yaml", "Local 3B model profile and dataset revisions"),
        ("configs/param_ganga_config.yaml", "2×V100 FP16 profile with 7B models"),
        ("configs/datasets.yaml", "Verified dataset revisions, licences, and condition support"),
        ("scripts/build_probe_set.py", "Build and freeze a stratified manifest"),
        ("scripts/run_experiment.py", "Run conditions, sweeps, and reports"),
        ("scripts/analyze_results.py", "Generate tables from predictions and a manifest"),
        ("scripts/plot_results.py", "Generate figures from analysis outputs"),
        ("scripts/audit_datasets.py", "Live provenance audit"),
        ("src/evaluation/attribution.py", "Architecture-specific decomposition"),
        ("src/routing/router.py", "Leakage-guarded, measured-cost router"),
        ("results/local_pilot/", "Pilot artifacts and limitations"),
    ],
    widths=[6.0, 10.6],
    table_caption="Principal repository entry points.",
)
heading(doc, "A.1  Representative commands", level=2)
for cmd in [
    "pytest -q",
    "python scripts/audit_datasets.py --out outputs/provenance/dataset_audit.json",
    "python scripts/build_probe_set.py --mode funsd --max-samples 200",
    "python scripts/build_probe_set.py --mode indic_ocr --max-pages 600 --per-group 100",
    "python scripts/run_experiment.py --config configs/param_ganga_config.yaml --mode funsd --max-samples 100 --resume --seed 42",
    "python scripts/run_experiment.py --config configs/param_ganga_config.yaml --mode indic_ocr --max-samples 500 --resume --seed 42",
    "python scripts/run_experiment.py --config configs/param_ganga_config.yaml --mode funsd --sweep --resume --seed 42",
    "python scripts/analyze_results.py --predictions outputs/predictions_funsd.jsonl --manifest manifests/probe_set_funsd.json",
    "python scripts/plot_results.py --analysis outputs/analysis_funsd/analysis.json --out-dir outputs/figures",
]:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(cmd)
    set_run_font(run, "Consolas", 8.2, color=RGBColor(0x33, 0x33, 0x33))

# ---------------------------------------------------------------------------
# APPENDIX B
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Appendix B — Manifest and Condition Schemas", level=1)
body(
    doc,
    "A manifest entry fixes the identity of one item. The fields below are mandatory for the confirmatory run.",
)
add_table(
    doc,
    ["Field", "Description"],
    [
        ("sample_id / page_id", "Item and document identifiers; page_id is the clustering unit"),
        ("dataset_name / revision / split", "Provenance and exact data version"),
        ("language / script_type", "Language and script grouping"),
        ("question / template_id", "Question text and the generator template"),
        ("answers / aliases", "Gold answer(s) and accepted aliases"),
        ("oracle_text_sha256", "Hash of the gold transcript, if available"),
        ("image_sha256", "Hash of the page image or crop"),
        ("supports_text_oracle / supports_layout_oracle", "Capability flags that gate C3 and C2"),
        ("answer_bbox", "Gold answer box when the layout condition is legal"),
        ("metadata", "Generator version, answer type, synthetic flag"),
    ],
    widths=[6.2, 10.4],
    table_caption="Manifest fields.",
)
body(
    doc,
    "Conditions are represented as an enum, not a free string. The VLM supports C0, Cδ, C1, C2, C2b, C3, C3b. "
    "The cascade supports C0, Cδ, C2, C3, C3b; C1 and C2b are rejected explicitly. A condition that is not "
    "supported by the dataset or model raises an error; it is never silently replaced by C0.",
)

# ---------------------------------------------------------------------------
# APPENDIX C
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Appendix C — PARAM Ganga Execution Checklist", level=1)
checks = [
    ("Account and allocation", "Confirm PARAM Ganga account, project code, and 300 GPU-hour allocation."),
    ("Modules", "Load Python 3.10+, CUDA 12.x, and the project virtual environment."),
    ("Repository", "Clone the pinned commit; run `pytest -q` and the smoke pipeline."),
    ("Models", "Pre-download Qwen2.5-VL-3B/7B, Qwen2.5-3B/7B, and OCR engines into the shared HF cache."),
    ("Data", "Build or copy the frozen manifest; verify SHA-256; do not stream mutable datasets."),
    ("Smoke test", "Run one FUNSD document through all conditions; confirm reportable=true and coverage."),
    ("Calibration", "Apply the frozen perturbation table; report induced CER; stop if non-monotonic."),
    ("Main runs", "3B NF4, 3B INT8, 7B FP16; checkpoint every model; use `--resume`."),
    ("Sweeps", "Run the 5×5 Cδ sweep; do not recompute baseline conditions."),
    ("Analysis", "Run `analyze_results.py` and `plot_results.py`; verify every table has a source file."),
    ("Human validation", "Export 100 stratified failures; collect labels; compute kappa."),
    ("Release", "Archive code, prompts, manifests, predictions, analysis, and environment record."),
]
for idx, (title, text) in enumerate(checks, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(f"☐  ")
    set_run_font(run, size=11, color=NAVY)
    run = p.add_run(f"{title}. ")
    set_run_font(run, size=10, bold=True, color=NAVY)
    run = p.add_run(text)
    set_run_font(run, size=10)

heading(doc, "C.1  Environment record to capture", level=2)
for text in [
    "Git commit, config SHA-256, dataset revision, model revision, prompt version.",
    "CUDA, PyTorch, transformers, bitsandbytes, Tesseract/PaddleOCR versions.",
    "GPU model, driver, VRAM, batch size, max pixels/tokens, decoding parameters.",
    "Wall-clock latency, input/output token counts, peak memory, and failure/retry counts.",
]:
    bullet(doc, text)

# ---------------------------------------------------------------------------
# APPENDIX D — ENDORSEMENT
# ---------------------------------------------------------------------------
page_break(doc)
heading(doc, "Appendix D — Endorsement and Signature", level=1)
body(
    doc,
    "This page is provided so that the faculty sponsor can record a decision and any conditions attached to the "
    "resource allocation.",
)
note_box(
    doc,
    "Resource requested",
    "PARAM Ganga: one GPU node with 2× NVIDIA Tesla V100 16 GB, up to 300 GPU-hours over 13 weeks, 12 CPUs, "
    "64 GB RAM, 50 GB scratch storage, and access to the shared Hugging Face cache. No paid annotation or "
    "hardware purchase is requested.",
    fill=LIGHT_BLUE_HEX,
)
body(doc, "")
add_table(
    doc,
    ["Decision", "Please mark one"],
    [
        ("Approved as requested", "☐"),
        ("Approved with modifications (specify below)", "☐"),
        ("Deferred; additional information required", "☐"),
        ("Not approved", "☐"),
    ],
    widths=[12.0, 4.4],
    table_caption="Faculty decision.",
    align_center_cols=[1],
)
body(doc, "")
for label in [
    "Faculty sponsor name: ________________________________________________",
    "Department / centre: _________________________________________________",
    "Signature: __________________________________    Date: ________________",
    "Comments / conditions:",
    "_______________________________________________________________________",
    "_______________________________________________________________________",
    "_______________________________________________________________________",
]:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(label)
    set_run_font(run, size=10, color=DARK)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(24)
run = p.add_run(
    "For administrative use: PARAM Ganga project code __________________  "
    "Allocation (GPU-hours) __________  Valid from __________ to __________"
)
set_run_font(run, size=9, italic=True, color=GREY)

doc.save(OUT)
print("saved", OUT)
