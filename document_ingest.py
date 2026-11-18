"""Universal, resource-conscious document-to-search-record conversion.

All non-PDF inputs become logical pages preserving source location (sheet/row,
slide, section, or text block). PDF OCR is handled by the existing app.
Never execute spreadsheet formulas or embedded document macros.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SUPPORTED = {"pdf", "docx", "doc", "xlsx", "xls", "csv", "tsv", "txt", "md", "markdown", "json", "html", "htm", "pptx", "rtf", "odt", "ods", "log"}
MAX_SIZE = 30 * 1024 * 1024  # limit independent of Streamlit's upload limit
MAX_TEXT = 2_000_000
MAX_ROWS_PER_SHEET = 15_000
MAX_ROWS_RENDERED = 15_000
PAGE_CHARS = 3200


def safe_decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            value = data.decode(encoding)
            if value:
                return value
        except UnicodeError:
            pass
    return data.decode("utf-8", errors="replace")


def squash(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text or "").strip()


def segments(text: str, title: str, target: int = PAGE_CHARS) -> list[dict]:
    """Logical page chunks; page numbers are local section references, not physical pages."""
    text = (text or "").strip()
    if not text:
        return []
    output = []
    for k in range(0, len(text), target):
        part = text[k:k+target]
        output.append({"kn": part, "en": "", "method": "Native text extraction", "translation_pending": True,
                       "source_location": title, "number": len(output) + 1})
    return output


def parse_csv(data: bytes, name: str, delimiter: str) -> list[dict]:
    raw = safe_decode(data)
    rows = csv.reader(io.StringIO(raw), delimiter=delimiter)
    header = None
    pages = []
    group = []
    count = 0
    start = 2
    for row in rows:
        if header is None:
            header = [str(x).strip() or f"Column {i+1}" for i, x in enumerate(row)]
            continue
        count += 1
        if count > MAX_ROWS_RENDERED:
            raise ValueError(f"More than {MAX_ROWS_RENDERED} rows. Split the file before uploading to avoid an incomplete index.")
        values = [f"{header[i] if i < len(header) else 'Column '+str(i+1)}: {squash(str(val))}" for i, val in enumerate(row) if str(val).strip()]
        if values:
            group.append(f"Row {count+1}: " + "; ".join(values))
        if len(group) >= 24:
            pages.append({"kn": f"Table: {name}\nHeader: {', '.join(header)}\n" + "\n".join(group), "en": "", "method": "Tabular extraction", "translation_pending": True,
                          "source_location": f"Rows {start}-{count+1}", "number": len(pages)+1})
            group, start = [], count+2
    if group:
        pages.append({"kn": f"Table: {name}\nHeader: {', '.join(header or [])}\n" + "\n".join(group), "en": "", "method": "Tabular extraction", "translation_pending": True,
                      "source_location": f"Rows {start}-{count+1}", "number": len(pages)+1})
    return pages


def extract_docx(data: bytes) -> list[dict]:
    from docx import Document
    doc = Document(io.BytesIO(data))
    blocks = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit('}', 1)[-1]
        if tag == 'p':
            from docx.text.paragraph import Paragraph
            t = Paragraph(child, doc).text.strip()
            if t:
                blocks.append(t)
        elif tag == 'tbl':
            from docx.table import Table
            table = Table(child, doc)
            for row_number, row in enumerate(table.rows, 1):
                blocks.append(f"Table row {row_number}: " + " | ".join(c.text.replace('\n','; ') for c in row.cells))
    return segments("\n".join(blocks), "Word document")


def extract_pptx(data: bytes) -> list[dict]:
    from pptx import Presentation
    deck = Presentation(io.BytesIO(data))
    output = []
    for i, slide in enumerate(deck.slides, 1):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                lines.append(shape.text)
            if shape.has_table:
                for row in shape.table.rows:
                    lines.append(' | '.join(cell.text for cell in row.cells))
        for group in segments('\n'.join(lines), f"Slide {i}"):
            group['number'] = len(output)+1
            output.append(group)
    return output


def extract_legacy_doc(data: bytes) -> list[dict]:
    try:
        result = subprocess.run(['antiword', '-w', '0', '-'], input=data,
                                capture_output=True, timeout=30, check=False)
    except FileNotFoundError as exc:
        raise ValueError('Legacy .doc requires the antiword system package; convert to DOCX if unavailable.') from exc
    if result.returncode:
        raise ValueError('Legacy .doc could not be read. Please convert it to DOCX or searchable PDF.')
    return segments(safe_decode(result.stdout), 'Legacy Word document')


def extract_rtf(data: bytes) -> list[dict]:
    from striprtf.striprtf import rtf_to_text
    return segments(rtf_to_text(safe_decode(data)), 'RTF')


def extract_html(data: bytes) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(safe_decode(data), 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return segments(soup.get_text('\n', strip=True), 'HTML document')


def extract_json(data: bytes) -> list[dict]:
    value = json.loads(safe_decode(data))
    pretty = json.dumps(value, ensure_ascii=False, indent=2)
    return segments(pretty, 'JSON document')


def _open_xml_docs(data: bytes, content_name: str) -> ET.Element:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return ET.fromstring(z.read(content_name))


def _xlsx_rows(data: bytes):
    """Read cell display values from Office Open XML without evaluating formulas."""
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
          'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
          'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        shared = []
        if 'xl/sharedStrings.xml' in names:
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('m:si', ns):
                shared.append(''.join(t.text or '' for t in si.findall('.//m:t', ns)))
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'].lstrip('/') for r in rels}
        for sh in wb.findall('m:sheets/m:sheet', ns):
            rel = sh.attrib.get('{'+ns['r']+'}id')
            target = targets.get(rel, '')
            if not target:
                continue
            path = target if target.startswith('xl/') else 'xl/'+target
            if path not in names:
                continue
            sheet_name = sh.attrib.get('name', 'Sheet')
            root = ET.fromstring(z.read(path))
            outrows = []
            for row in root.findall('.//m:sheetData/m:row', ns):
                cells = []
                for c in row.findall('m:c', ns):
                    v = c.find('m:v', ns)
                    ctype = c.get('t')
                    if ctype == 'inlineStr':
                        val = ''.join(t.text or '' for t in c.findall('.//m:t', ns))
                    elif v is not None:
                        val = v.text or ''
                        if ctype == 's':
                            try: val = shared[int(val)]
                            except (ValueError, IndexError): pass
                    else:
                        val = ''
                    ref = c.attrib.get('r', '')
                    col = re.match(r'[A-Z]+', ref)
                    if col and val != '':
                        cells.append((col.group(), val))
                if cells:
                    outrows.append((int(row.get('r', len(outrows)+1)), cells))
                    if len(outrows) > MAX_ROWS_PER_SHEET:
                        raise ValueError('Spreadsheet has too many rows for a safe complete index; split the workbook.')
            yield sheet_name, outrows


def extract_xlsx(data: bytes) -> list[dict]:
    output = []
    for sheetname, rows in _xlsx_rows(data):
        if not rows:
            continue
        header = dict(rows[0][1])
        groups = []
        group_first = rows[0][0]
        group_last = rows[0][0]
        for n, cells in rows[1:]:
            values = [f"{header.get(col, col)}: {val}" for col, val in cells]
            if values:
                groups.append(f"Row {n}: " + '; '.join(values))
                group_last = n
            if len(groups) >= 22:
                output.append({"kn": f"Sheet: {sheetname}\n"+'\n'.join(groups), "en": "", "method": "Excel extraction", "translation_pending": True,
                               "source_location": f"Sheet {sheetname}, rows {group_first}-{group_last}", "number":len(output)+1})
                groups = []
                group_first = n+1
        if groups:
            output.append({"kn": f"Sheet: {sheetname}\n"+'\n'.join(groups), "en": "", "method": "Excel extraction", "translation_pending": True,
                           "source_location": f"Sheet {sheetname}, rows {group_first}-{group_last}", "number":len(output)+1})
        if len(rows)==1:
            output.append({"kn": f"Sheet: {sheetname}\nHeaders: " + '; '.join(f"{k}: {v}" for k,v in header.items()),
                           "en":"", "method":"Excel extraction", "translation_pending":True,
                           "source_location":f"Sheet {sheetname}, row {rows[0][0]}","number":len(output)+1})
    return output


def extract_xls(data: bytes) -> list[dict]:
    try:
        from python_calamine import CalamineWorkbook
    except ImportError as e:
        raise ValueError('Legacy .xls needs python-calamine. Convert to .xlsx if not installed.') from e
    book = CalamineWorkbook.from_filelike(io.BytesIO(data))
    output = []
    for name in book.sheets:
        sheet = book.get_sheet_by_name(name)
        rows = sheet.to_python()
        if not rows:
            continue
        headers = [str(x) for x in rows[0]]
        for offset in range(1, len(rows), 24):
            lines = []
            for j, values in enumerate(rows[offset:offset+24], offset+1):
                lines.append('Row '+str(j+1)+': '+'; '.join(f'{headers[k] if k<len(headers) else "Column "+str(k+1)}: {v}' for k,v in enumerate(values) if v is not None and str(v).strip()))
            output.append({'kn':f'Sheet: {name}\n'+'\n'.join(lines),'en':'','method':'Excel extraction','translation_pending':True,
                           'source_location':f'Sheet {name}, rows {offset+1}-{min(len(rows),offset+24)}','number':len(output)+1})
    return output


def extract_odf(data: bytes) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read('content.xml'))
    text_ns = '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'
    texts = []
    for node in root.iter():
        if node.tag in {text_ns+'p', text_ns+'h'}:
            s = ''.join(node.itertext()).strip()
            if s: texts.append(s)
    return segments('\n'.join(texts), 'OpenDocument')


def extract_upload(filename: str, data: bytes) -> list[dict]:
    suffix = Path(filename).suffix.lower().lstrip('.')
    if suffix not in SUPPORTED:
        raise ValueError('Unsupported format. Upload PDF, Word, Excel, PowerPoint, CSV, TXT, Markdown, JSON, HTML, RTF, or ODF.')
    if not data:
        raise ValueError('The file is empty.')
    if len(data) > MAX_SIZE:
        raise ValueError('File exceeds 30 MB upload-processing limit; split it before uploading.')
    if suffix == 'pdf':
        raise ValueError('PDF processing uses the existing OCR pipeline in app.py.')
    if suffix in {'txt','md','markdown','log'}:
        pages = segments(safe_decode(data), 'Text file')
    elif suffix == 'docx': pages = extract_docx(data)
    elif suffix == 'doc': pages = extract_legacy_doc(data)
    elif suffix == 'pptx': pages = extract_pptx(data)
    elif suffix in {'html','htm'}: pages = extract_html(data)
    elif suffix == 'rtf': pages = extract_rtf(data)
    elif suffix == 'json': pages = extract_json(data)
    elif suffix == 'csv': pages = parse_csv(data,filename, ',')
    elif suffix == 'tsv': pages = parse_csv(data,filename, '\t')
    elif suffix == 'xlsx': pages = extract_xlsx(data)
    elif suffix == 'xls': pages = extract_xls(data)
    elif suffix in {'odt','ods'}: pages = extract_odf(data)
    else: raise ValueError(f'Unsupported format: {suffix}')
    if not pages:
        raise ValueError('No searchable text found. This may be an image-based document requiring OCR conversion.')
    total = sum(len(p['kn']) for p in pages)
    if total > MAX_TEXT:
        raise ValueError('Extracted text exceeds 2 million characters. Split the file for complete indexing.')
    for p in pages:
        p['translation_pending'] = bool(any('\u0c80'<=ch<='\u0cff' for ch in p['kn']))
        if not p['translation_pending']:
            p['en'] = p['kn']
    return pages
