"""Rapor üretimi için python-docx yardımcıları (ders şablonunun stillerini kullanır)."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HEADER_FILL = "D9E2F3"


class Report:
    def __init__(self, template: Path):
        self.doc = Document(str(template))
        body = self.doc.element.body
        for el in list(body):
            if el.tag != qn("w:sectPr"):
                body.remove(el)
        self.fig_n = 0
        self.tab_n = 0
        st = self.doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        pf = st.paragraph_format
        pf.line_spacing = 1.15
        pf.space_after = Pt(4)
        self._page_numbers()

    # ---------------------------------------------------------------- temel öğeler
    def _page_numbers(self):
        for sec in self.doc.sections:
            p = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
                r = p.add_run()
                if t:
                    fc = OxmlElement("w:fldChar")
                    fc.set(qn("w:fldCharType"), t)
                    r._r.append(fc)
                else:
                    it = OxmlElement("w:instrText")
                    it.set(qn("xml:space"), "preserve")
                    it.text = txt
                    r._r.append(it)

    @staticmethod
    def tr_decimal(t: str) -> str:
        """Türkçe ondalık virgül: 0.309 -> 0,309 (sürüm numaraları 8.3.6 gibi çok noktalılar korunur); sayı önündeki
        kısa çizgi gerçek eksi işaretine (U+2212) çevrilir, böylece Word satırı eksiden sonra bölmez."""
        t = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", lambda m: f"{m.group(1)},{m.group(2)}", t)
        return re.sub(r"(?<![\w.,/-])-(?=\d)", "−", t)

    def _runs(self, p, text: str, size=None, convert: bool = True):
        # **kalın**, *italik* ve `kod` işaretlemesi
        for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
            if not part:
                continue
            if convert and not part.startswith("`"):
                part = self.tr_decimal(part)
            if part.startswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("*"):
                r = p.add_run(part[1:-1]); r.italic = True
            elif part.startswith("`"):
                r = p.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
            else:
                r = p.add_run(part)
            if size:
                r.font.size = Pt(size)
        return p

    def title(self, text):
        return self.doc.add_paragraph(text, style="Title")

    def h1(self, text, new_page: bool = False):
        para = self.doc.add_paragraph(text, style="Heading 1")
        para.paragraph_format.page_break_before = new_page
        return para

    def h2(self, text, new_page: bool = False):
        para = self.doc.add_paragraph(text, style="Heading 2")
        para.paragraph_format.page_break_before = new_page
        return para

    def p(self, text, align_justify=True, size=None, space_after=None, convert=True):
        para = self._runs(self.doc.add_paragraph(), text, size, convert)
        if align_justify:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if space_after is not None:
            para.paragraph_format.space_after = Pt(space_after)
        return para

    def bullets(self, items, size=None):
        for it in items:
            para = self.doc.add_paragraph(style="List Paragraph")
            para.paragraph_format.left_indent = Cm(0.6)
            para.paragraph_format.first_line_indent = Cm(-0.4)
            para.paragraph_format.space_after = Pt(1)
            self._runs(para, "• " + it, size)

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------------------------------------------------------- tablo / şekil
    def caption(self, kind: str, text: str):
        if kind == "Şekil":
            self.fig_n += 1; n = self.fig_n
        else:
            self.tab_n += 1; n = self.tab_n
        c = self.doc.add_paragraph(style="Caption")
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(f"{kind} {n}. "); r.bold = True
        c.add_run(self.tr_decimal(text)).bold = False
        c.paragraph_format.space_after = Pt(6)
        return n

    def figure(self, path: Path, caption: str, width_cm: float = 16.5):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_after = Pt(0)
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        return self.caption("Şekil", caption)

    def table(self, header, rows, widths_cm=None, caption=None, font=8.5, bold_last=False, highlight_rows=()):
        if caption:
            n = self.caption("Tablo", caption)
            self.doc.paragraphs[-1].paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        self._borders(t)
        for j, h in enumerate(header):
            self._cell(t.cell(0, j), str(h), font, bold=True, fill=HEADER_FILL)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                hl = i in highlight_rows
                self._cell(t.cell(i + 1, j), "" if v is None else str(v), font,
                           bold=hl or (bold_last and i == len(rows) - 1), fill="FCE9DF" if hl else None,
                           align_left=(j == 0))
        if widths_cm:
            for j, w in enumerate(widths_cm):
                for r in t.rows:
                    r.cells[j].width = Cm(w)
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(2)
        return t

    def _cell(self, cell, text, font, bold=False, fill=None, align_left=False):
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if align_left else WD_ALIGN_PARAGRAPH.CENTER
        self._runs(p, text, font)
        if bold:
            for r in p.runs:
                r.bold = True
        if fill:
            tcPr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
            tcPr.append(shd)

    @staticmethod
    def _borders(table):
        tblPr = table._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "8EA9DB")
            b.append(e)
        tblPr.append(b)

    def save(self, path: Path):
        self.doc.save(str(path))
