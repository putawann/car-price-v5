"""Build a public, offline report from saved notebook outputs; never execute cells.

Run from this worktree: python -X utf8 make_report.py
Only curated aggregates, PASS/WARN messages and three plot types are exported.
Uses the standard library only. Input files are never modified.
"""
from __future__ import annotations

import base64
import csv
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'docs'
MISSING = 'ไม่มีผล'
DATASETS = ('new_cardb', 'merge50minC')
MODELS = ('DT', 'RF', 'XGB')
FORBIDDEN = re.compile(r'[A-Za-z]:[/\\]|putaw|(?:api[_-]?key|access[_-]?token|secret)\s*[:=]', re.I)


def joined(value):
    return ''.join(value) if isinstance(value, list) else str(value or '')


def output_text(cell):
    return '\n'.join(joined(o.get('text', o.get('data', {}).get('text/plain', '')))
                     for o in cell.get('outputs', []))


class Tables(HTMLParser):
    """Extract cell text only; never copy notebook HTML, CSS or attributes."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.rows = None
        self.row = None
        self.parts = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.rows = []
        elif tag == 'tr' and self.rows is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.parts = []

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.parts is not None:
            self.row.append(' '.join(''.join(self.parts).split()))
            self.parts = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == 'table' and self.rows is not None:
            self.tables.append(self.rows)
            self.rows = None


def cell_tables(cell):
    parser = Tables()
    for output in cell.get('outputs', []):
        parser.feed(joined(output.get('data', {}).get('text/html', '')))
    return parser.tables


def numeric(value):
    try:
        number = float(str(value).replace(',', '').replace('%', '').strip())
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def records(table):
    if len(table) < 2:
        return []
    headers = table[0]
    return [dict(zip(headers, row)) for row in table[1:] if len(row) == len(headers)]


def number_records(table):
    return [{key: numeric(value) if numeric(value) is not None else value
             for key, value in row.items()} for row in records(table)]


def json_line(text, prefix):
    for line in text.splitlines():
        if line.startswith(prefix + '='):
            value = json.loads(line[len(prefix) + 1:])
            if not isinstance(value, dict):
                raise ValueError('expected JSON object')
            return value
    return None


def clean_result(raw, run):
    """Whitelist fields; no arbitrary text or paths from RESULT_JSON are exported."""
    result = {key: run[key] for key in ('run_id', 'model', 'dataset', 'variant')}
    result['dataset'] = {'new_cardb': 'new_car_dataset.csv',
                         'merge50minC': 'merge_50min_clean.csv'}[run['dataset']]
    result['version'] = 'v5'
    result['features'] = [x for x in raw.get('features', [])
                          if x in ('engine', 'mileage', 'car_age', 'brand', 'model', 'car_type', 'fuel', 'gear')]
    for split in ('train', 'validation', 'test'):
        result[split] = {key: numeric(raw.get(split, {}).get(key))
                         for key in ('R2', 'RMSE', 'MAE', 'MAPE')}
    result['best_params'] = {}
    for key, value in raw.get('best_params', {}).items():
        if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
            if value is None or isinstance(value, (int, float, bool)):
                result['best_params'][key] = value
            elif value in ('sqrt', 'log2', 'auto', 'squared_error', 'absolute_error'):
                result['best_params'][key] = value
    return result


def clean_importance(raw, run_id):
    result = {'run_id': run_id, 'permutation': {}, 'model_based': {}}
    for variable in ('engine', 'mileage', 'car_age', 'brand', 'model', 'car_type', 'fuel', 'gear'):
        if variable in raw.get('permutation', {}):
            value = raw['permutation'][variable]
            result['permutation'][variable] = {k: numeric(value.get(k)) for k in ('mean', 'std')}
        if variable in raw.get('model_based', {}):
            result['model_based'][variable] = numeric(raw['model_based'][variable])
    return result


def extract_plot(cell, run, kind):
    for output in cell.get('outputs', []):
        encoded = output.get('data', {}).get('image/png')
        if encoded:
            blob = base64.b64decode(joined(encoded), validate=False)
            if not blob.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('invalid PNG')
            # Strip ancillary metadata; keep only image decoding chunks.
            # Plot pixels are preserved exactly, without requiring imaging packages.
            clean = bytearray(blob[:8])
            offset = 8
            while offset + 12 <= len(blob):
                size = struct.unpack('>I', blob[offset:offset + 4])[0]
                chunk = blob[offset + 4:offset + 8]
                end = offset + size + 12
                if end > len(blob):
                    raise ValueError('truncated PNG')
                if chunk in (b'IHDR', b'PLTE', b'IDAT', b'IEND', b'tRNS'):
                    clean.extend(blob[offset:end])
                offset = end
                if chunk == b'IEND':
                    break
            rel = f"img/{run['run_id']}_{kind}.png"
            (OUT / rel).write_bytes(clean)
            return rel
    return None


def parse_notebook(path, log):
    match = re.fullmatch(r'(DT|RF|XGB)/(\1)_(new_cardb|merge50minC)(_adj)?\.ipynb', path)
    if not match:
        raise ValueError('run_order.txt contains an unexpected notebook name')
    model, _, dataset, adj = match.groups()
    run = {'run_id': Path(path).stem, 'notebook': path, 'model': model,
           'dataset': dataset, 'variant': 'adj' if adj else 'base',
           'result': None, 'importance': None, 'cv': None, 'cv_tables': [],
           'ablation': [], 'overfit': [], 'plots': {}, 'data': {}, 'missing': []}
    if log:
        run['runtime'] = {k: log.get(k) for k in ('start', 'end', 'status')}
        run['runtime']['seconds'] = numeric(log.get('seconds'))
    else:
        run['runtime'] = None
        run['missing'].append('run_log')
    try:
        notebook = json.loads((ROOT / path).read_text(encoding='utf-8-sig'))
        cells = notebook['cells']
    except (OSError, ValueError, KeyError):
        run['missing'].append('อ่าน notebook ไม่ได้')
        return run
    failed = any(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', []))
    failed = failed or bool(log and log.get('status', '').lower() not in ('ok', 'success', 'completed'))
    if failed:
        run['missing'].append('notebook รันไม่สำเร็จ (ไม่ใช้ผลบางส่วน)')
        return run
    for cell in cells:
        text = output_text(cell)
        header = re.search(r'^=== \[\d+\] (.*?) ===', text, re.M)
        title = header.group(1) if header else ''
        for prefix, field in (('RESULT_JSON', 'result'), ('IMPORTANCE_JSON', 'importance')):
            try:
                raw = json_line(text, prefix)
                if raw is not None:
                    if raw.get('run_id') != run['run_id']:
                        run['missing'].append(prefix + ': run_id ไม่ตรง')
                        continue
                    run[field] = clean_result(raw, run) if field == 'result' else clean_importance(raw, run['run_id'])
            except (ValueError, TypeError, AttributeError):
                run['missing'].append(prefix + ': รูปแบบไม่ถูกต้อง')
        if title.startswith('เตรียมข้อมูล:'):
            m = re.search(r'ขนาดตาราง df\s*:\s*([\d,]+) แถว', text)
            if m:
                run['data']['rows'] = int(m[1].replace(',', ''))
            m = re.search(r'หลังรวมชื่อรุ่น:\s*([\d,]+)', text)
            if m:
                run['data']['models'] = int(m[1].replace(',', ''))
        if title.startswith('แบ่ง train /'):
            for table in cell_tables(cell):
                split = number_records(table)
                if split and 'แถว' in split[0]:
                    run['data']['split'] = split
                    if 'models' not in run['data']:
                        run['data']['models'] = split[0].get('จำนวนรุ่น')
        if title.startswith('ตาราง CV train'):
            for table in cell_tables(cell):
                rows = number_records(table)
                if rows and 'validation MAE' in rows[0]:
                    run['cv_tables'].append({'kind': 'fold_scores', 'rows': rows})
                    means = next((r for r in rows if r.get('') == 'mean'), None)
                    sd = next((r for r in rows if r.get('') == 'sd'), None)
                    if means and sd:
                        run['cv'] = {'mean': means['validation MAE'], 'sd': sd['validation MAE'],
                                     'source': title, 'precision': 'printed integer baht'}
        if title.startswith('ตาราง 10 อันดับแรก'):
            for table in cell_tables(cell):
                rows = number_records(table)
                if rows and 'cv_mae' in rows[0]:
                    run['cv_tables'].append({'kind': 'search_candidates', 'rows': rows})
        if title.startswith('สรุปการตรวจ overfit'):
            run['overfit'] = [line.strip() for line in text.splitlines()
                              if re.match(r'^\s*(PASS|WARN)\b', line)]
        if title.lower().startswith('ablation: เปรียบเทียบ'):
            for table in cell_tables(cell):
                rows = number_records(table)
                if rows and 'CV MAE mean' in rows[0]:
                    run['ablation'] = [{k: v for k, v in row.items() if k} for row in rows]
            if not run['ablation']:
                columns = ['CV MAE mean', 'CV MAE SD', 'test MAE', 'test RMSE', 'test R2', 'test MAPE',
                           'change test MAE (baht)', 'change test MAE (%)',
                           'change CV MAE mean (baht)', 'change CV MAE mean (%)']
                for line in text.splitlines():
                    m = re.match(r'^\s*(none|scale only|log only|both)\s+(.+)$', line)
                    if m:
                        values = m[2].split()
                        if len(values) == len(columns) and all(numeric(v) is not None for v in values):
                            run['ablation'].append({'variant': m[1], **dict(zip(columns, map(numeric, values)))})
        kind = None
        if title.lower().startswith('learning curve: ตารางและกราฟ'):
            kind = 'learning'
        elif title.startswith('ความสำคัญแบบ permutation'):
            kind = 'permutation'
        elif title.startswith('ความสำคัญจากโมเดลรวมกลับ'):
            kind = 'model_based'
        if kind:
            try:
                plot = extract_plot(cell, run, kind)
                if plot:
                    run['plots'][kind] = plot
            except (ValueError, struct.error):
                run['missing'].append('รูป ' + kind + ': อ่านไม่ได้')
    for field, label in [('result', 'RESULT_JSON'), ('importance', 'IMPORTANCE_JSON'),
                         ('cv', 'ตาราง CV'), ('overfit', 'PASS/WARN overfit')]:
        if not run[field]:
            run['missing'].append(label)
    if run['result']:
        for split in ('train', 'validation', 'test'):
            for key, value in run['result'][split].items():
                if value is None:
                    run['missing'].append(f'{split} {key}')
    for kind in ('learning', 'permutation', 'model_based'):
        if kind not in run['plots']:
            run['missing'].append('รูป ' + kind)
    if adj and len(run['ablation']) != 4:
        run['missing'].append('ablation ไม่ครบ 4 แบบ')
    for field in ('rows', 'models', 'split'):
        if field not in run['data']:
            run['missing'].append('ข้อมูลรวม ' + field)
    return run


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=0):
    return MISSING if value is None else f'{value:,.{digits}f}'


def pct(value, fraction=False):
    return MISSING if value is None else f'{value * (100 if fraction else 1):,.2f}%'


def metric(run, key='MAE'):
    return (run.get('result') or {}).get('test', {}).get(key)


def label(run):
    return f"{run['model']} · {run['dataset']} · {run['variant']}"


def cv_text(run):
    cv = run.get('cv')
    return f"{fmt(cv['mean'])} ± {fmt(cv['sd'])}" if cv else MISSING


def table(headers, rows, best=()):
    body = []
    for i, row in enumerate(rows):
        css = ' class="best"' if i in best else ''
        body.append(f'<tr{css}>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>')
    return '<div class="table-wrap" role="region" aria-label="ตารางเลื่อนแนวนอนได้" tabindex="0"><table><thead><tr>' + ''.join(
        f'<th scope="col">{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


def figure(run, kind):
    rel = run['plots'].get(kind)
    names = {'learning': 'Learning curve', 'permutation': 'Permutation importance', 'model_based': 'Model-based importance'}
    alt = names[kind] + ' — ' + label(run)
    if not rel:
        return f'<p>{esc(alt)}: {MISSING}</p>'
    return f'<figure><a href="{esc(rel)}" aria-label="เปิดรูปขนาดเต็ม: {esc(alt)}"><img src="{esc(rel)}" alt="{esc(alt)}" loading="lazy" decoding="async"></a><figcaption>{esc(alt)} · แตะเพื่อดูเต็มขนาด</figcaption></figure>'


CSS = """
:root{color-scheme:light dark;--bg:#f4f6f8;--card:#fff;--ink:#172c3b;--muted:#526572;--line:#d5dfe5;--accent:#006a64;--best:#e0f4e9;--warn:#8d410b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 system-ui,-apple-system,'Tahoma',sans-serif}main{max-width:1120px;margin:auto;padding:24px 16px 64px}header{padding:16px 0 24px}h1{font-size:clamp(1.8rem,5vw,2.7rem);line-height:1.3;letter-spacing:-.03em}h2{font-size:1.45rem;line-height:1.5;margin:0 0 16px}h3{font-size:1.12rem}p{margin:10px 0}a{color:var(--accent);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}nav a{padding:6px 12px;border:1px solid var(--line);border-radius:20px;text-decoration:none;font-size:.9rem}section{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin:16px 0;scroll-margin-top:12px}.muted,figcaption{color:var(--muted);font-size:.88rem}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.1em}.cards,.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{padding:18px;border:1px solid var(--line);border-radius:12px;min-width:0}.value{font-size:1.8rem;font-weight:750;color:var(--accent)}.note{border-left:4px solid var(--accent);padding:8px 14px;background:var(--bg)}.table-wrap{max-width:100%;overflow:auto;border:1px solid var(--line);border-radius:10px;margin:14px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.9rem}th,td{padding:10px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}th{background:var(--bg);font-weight:650}th:first-child,td:first-child{text-align:left}tr:last-child td{border-bottom:0}.best{background:var(--best);font-weight:700}details{border-top:1px solid var(--line);padding:12px 0}summary{cursor:pointer;font-weight:650;overflow-wrap:anywhere}figure{margin:8px 0;min-width:0}img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}figcaption{margin:6px 0;overflow-wrap:anywhere}ul{padding-left:22px}li{margin:7px 0;overflow-wrap:anywhere}.pass{color:var(--accent)}.warn{color:var(--warn)}code{overflow-wrap:anywhere}footer{padding:14px 0;color:var(--muted)}
@media(max-width:600px){main{padding:12px 10px 40px}section{padding:16px 12px;border-radius:12px}.cards,.plots{grid-template-columns:1fr}th,td{padding:9px}.value{font-size:1.5rem}h2{font-size:1.25rem}}
@media(prefers-color-scheme:dark){:root{--bg:#101a23;--card:#182630;--ink:#e0ecf3;--muted:#a4b8c5;--line:#344957;--accent:#73d7c4;--best:#173e35;--warn:#ffbf85}}
"""


def render(runs, missing):
    winners = {}
    for dataset in DATASETS:
        eligible = [r for r in runs if r['dataset'] == dataset and metric(r) is not None]
        winners[dataset] = min(eligible, key=metric) if eligible else None
    parts = ['<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             '<meta name="description" content="ผลฝึกโมเดลราคารถ v5 จาก output ที่บันทึกไว้ 12 งาน"><title>ผลการฝึกโมเดลราคารถ · v5</title>',
             '<style>' + CSS + '</style></head><body><main><header><p class="eyebrow">TRAINING RESULTS / V5</p><h1>ผลการฝึกโมเดลราคารถ</h1>',
             '<p>12 การทดลอง · DT / RF / XGB · 2 ชุดข้อมูล · base และ adj</p><p class="muted">อ่านจาก output ที่บันทึกไว้เท่านั้น ไม่มีการฝึกหรือรัน notebook ใหม่ · หน่วยเงิน: บาท</p><nav aria-label="สารบัญ">']
    sections = [('summary', 'สรุปสั้น'), ('data', 'ข้อมูลที่ใช้'), ('results', 'ผลหลัก'), ('adjustment', 'ผลการปรับ'),
                ('importance', 'ความสำคัญตัวแปร'), ('overfit', 'Overfit'), ('learning', 'Learning curves'), ('runtime', 'เวลารัน')]
    parts.extend(f'<a href="#{key}">{name}</a>' for key, name in sections)
    parts.append('</nav></header><section id="summary"><h2>1. สรุปสั้น</h2><div class="cards">')
    for dataset, winner in winners.items():
        parts.append(f'<article class="card"><h3>{dataset}</h3>')
        if winner:
            imp = (winner['importance'] or {}).get('permutation', {})
            top = sorted((k for k in imp if imp[k]['mean'] is not None), key=lambda k: imp[k]['mean'], reverse=True)[:3]
            parts.append(f'<p><strong>{winner["model"]} / {winner["variant"]}</strong> มี test MAE ต่ำสุด</p><div class="value">{fmt(metric(winner))} บาท</div>')
            parts.append('<p>ตัวแปรสำคัญ 3 อันดับของโมเดลนี้: ' + esc(' → '.join(top) or MISSING) + '</p>')
        else:
            parts.append(f'<p>{MISSING}</p>')
        pairs = adjustment_pairs(runs, dataset)
        helped = [m for m, b, a in pairs if b is not None and a is not None and a < b]
        worse = [m for m, b, a in pairs if b is not None and a is not None and a >= b]
        parts.append(f'<p>adj ลด test MAE: {esc(", ".join(helped) or "ไม่มี")}; ไม่ลด: {esc(", ".join(worse) or "ไม่มี")}</p></article>')
    parts.append('</div><p class="muted">เลือกผู้ชนะจาก test MAE ในผลที่มีอยู่ เป็นการสรุปเชิงพรรณนา ไม่ใช่หลักฐานยืนยันผลบนข้อมูลใหม่ ความสำคัญแบบ permutation วัด MAE ที่เพิ่มขึ้นเมื่อสลับตัวแปร ไม่ได้บอกเหตุและผล</p></section>')
    parts.append('<section id="data"><h2>2. ข้อมูลที่ใช้</h2>')
    rows = []
    for dataset in DATASETS:
        data = next((r['data'] for r in runs if r['dataset'] == dataset and r['data'].get('split')), {})
        splits = {s['ชุด']: s['แถว'] for s in data.get('split', [])}
        rows.append([dataset, fmt(data.get('rows')), fmt(data.get('models')), *(fmt(splits.get(k)) for k in ('train', 'validation', 'test'))])
    parts.append(table(['ชุดข้อมูล', 'แถว', 'รุ่น (brand|model)', 'Train', 'Validation', 'Test'], rows))
    parts.append('<ul><li><strong>merge50minC / merge_50min_clean</strong>: ทำความสะอาดด้วยการลบแถวเท่านั้น ไม่รวมชื่อรุ่น; แบ่งแบบ stratify ตาม brand|model</li><li><strong>new_cardb</strong>: กฎ v3 กรองอายุรถและรวมชื่อรุ่นที่สะกดต่างกัน; forced-train บังคับ 1 แถวต่อ brand|model เข้า train ก่อนสุ่มเติมโควตาและแบ่ง validation/test</li><li>ทั้งสองชุดใช้ car_age แทน year และตัด color; เป้าหมายคือ price และใช้ 8 ตัวแปร</li></ul>')
    parts.append('<p class="note"><strong>เทียบภายในชุดข้อมูลก่อน:</strong> สองชุดใช้แถว test คนละชุดและวิธีแบ่งต่างกัน ค่า error ข้ามชุดจึงไม่ใช่การแข่งบนข้อสอบเดียวกัน</p></section>')
    parts.append('<section id="results"><h2>3. ผลหลักทั้ง 12 งาน</h2><p class="muted">เลื่อนตารางแนวนอนได้ · แถวสีเน้น = test MAE ต่ำสุดในชุดข้อมูล · CV คือ validation MAE ของโมเดลที่เลือก บน 5 folds ใน train; mean ± SD อ่านจากตารางที่พิมพ์และปัดเป็นบาท</p>')
    main_rows = [[r['model'], r['dataset'], r['variant'], fmt(metric(r, 'R2'), 4), fmt(metric(r, 'RMSE')),
                  fmt(metric(r)), pct(metric(r, 'MAPE'), True), cv_text(r)] for r in runs]
    parts.append(table(['โมเดล', 'ชุดข้อมูล', 'แบบ', 'Test R²', 'RMSE (บาท)', 'MAE (บาท)', 'MAPE', 'CV MAE ± SD (บาท)'], main_rows,
                       [i for i, r in enumerate(runs) if winners[r['dataset']] is r]))
    parts.append('<p><a href="results.json">ดาวน์โหลดตัวเลขทั้งหมด (JSON)</a> รวม train / validation / test, best params, CV และ importance</p></section>')
    parts.append('<section id="adjustment"><h2>4. ผลของการปรับข้อมูล</h2><p>adj = scale ตัวแปรตัวเลข + log ราคา แล้วแปลงกลับเป็นบาทเมื่อวัดผล · Δ = adj − base; ค่าติดลบแปลว่าความคลาดเคลื่อนลดลง</p>')
    changes = []
    for dataset in DATASETS:
        for model, base, adj in adjustment_pairs(runs, dataset):
            delta = adj - base if base is not None and adj is not None else None
            changes.append([model, dataset, fmt(base), fmt(adj), fmt(delta), pct(delta / base * 100) if delta is not None and base else MISSING])
    parts.append(table(['โมเดล', 'ชุดข้อมูล', 'Base MAE', 'Adj MAE', 'Δ บาท', 'Δ %'], changes))
    parts.append('<p class="note">Ablation ด้านล่างตรึงพารามิเตอร์ที่ปรับสำหรับ both ไม่ได้ค้นหาพารามิเตอร์ใหม่ให้แต่ละแบบ ดังนั้น none ใน ablation อาจไม่เท่ากับ base ด้านบน · scale only = scale ตัวเลข; log only = log ราคา</p>')
    for r in runs:
        if r['variant'] != 'adj':
            continue
        parts.append(f'<details><summary>{esc(label(r))} · ablation 4 แบบ</summary>')
        if r['ablation']:
            headers = list(r['ablation'][0])
            rows = []
            for item in r['ablation']:
                row = []
                for key in headers:
                    v = item.get(key)
                    row.append(v if key == 'variant' else pct(v, True) if key == 'test MAPE' else pct(v) if '(%)' in key else fmt(v, 4) if key == 'test R2' else fmt(v))
                rows.append(row)
            parts.append(table(headers, rows))
        else:
            parts.append(f'<p>{MISSING}</p>')
        parts.append('</details>')
    parts.append('</section><section id="importance"><h2>5. ความสำคัญของตัวแปร</h2><p>Permutation: MAE บน test ที่เพิ่มขึ้น (บาท), mean ± std จากการสลับ 10 รอบ; ค่าสูงแปลว่าโมเดลพึ่งตัวแปรนั้นมาก · Model-based: สัดส่วน impurity (DT/RF) หรือ gain (XGB) รวม dummy กลับเป็นตัวแปรเดิม แต่ละวิธีใช้คนละสเกลและอาจให้อันดับต่างกัน</p>')
    for r in runs:
        parts.append(f'<details><summary>{esc(label(r))}</summary>')
        imp = r['importance'] or {}
        perm = imp.get('permutation', {})
        shares = imp.get('model_based', {})
        variables = sorted(perm.keys() | shares.keys(), key=lambda k: (perm.get(k, {}).get('mean') or 0), reverse=True)
        rows = []
        for v in variables:
            p = perm.get(v, {})
            rows.append([v, f"{fmt(p.get('mean'))} ± {fmt(p.get('std'))}", pct(shares.get(v), True)])
        parts.append(table(['ตัวแปร', 'Permutation MAE ± std (บาท)', 'Model-based share'], rows) if rows else f'<p>{MISSING}</p>')
        parts.append('<div class="plots">' + figure(r, 'permutation') + figure(r, 'model_based') + '</div></details>')
    parts.append('</section><section id="overfit"><h2>6. ตรวจ Overfit</h2><p class="muted">คัดเฉพาะ PASS/WARN จากเซลล์สรุป overfit เดิม เกณฑ์ต่างกันตามโมเดล; WARN อาจเกิดจาก test ง่ายกว่า CV จึงควรอ่านทิศของความต่าง ไม่ใช่ตัดสินว่า overfit เสมอไป</p>')
    for r in runs:
        warns = sum(line.startswith('WARN') for line in r['overfit'])
        parts.append(f'<details><summary>{esc(label(r))} · {str(warns) + " WARN" if r["overfit"] else MISSING}</summary><ul>')
        parts.extend(f'<li class="{"warn" if line.startswith("WARN") else "pass"}">{esc(line)}</li>' for line in r['overfit'])
        if not r['overfit']:
            parts.append(f'<li>{MISSING}</li>')
        parts.append('</ul></details>')
    parts.append('</section><section id="learning"><h2>7. Learning curves</h2><p class="muted">หนึ่งรูปต่อ notebook · แตะภาพเพื่อเปิดขนาดเต็ม · ใช้กราฟที่บันทึกไว้เดิม</p>')
    for model in MODELS:
        parts.append(f'<h3>{model}</h3><div class="plots">')
        parts.extend(figure(r, 'learning') for r in runs if r['model'] == model)
        parts.append('</div>')
    parts.append('</section><section id="runtime"><h2>8. เวลารัน</h2>')
    total = sum((r['runtime'] or {}).get('seconds') or 0 for r in runs)
    parts.append(f'<p>รวมเวลาตาม run_log.csv: <strong>{duration(total)}</strong> ({fmt(total)} วินาที)</p><p class="muted">เวลาเริ่ม/จบเป็นค่าตาม log ไม่ได้ระบุเขตเวลา; ผลรวมเป็นเวลาของแต่ละงาน ไม่รวมช่องว่างระหว่างงาน</p>')
    rows = []
    for r in runs:
        runtime = r['runtime'] or {}
        rows.append([r['run_id'], runtime.get('start') or MISSING, runtime.get('end') or MISSING,
                     duration(runtime.get('seconds')), runtime.get('status') or MISSING])
    parts.append(table(['งาน', 'เริ่ม', 'จบ', 'ระยะเวลา', 'สถานะ'], rows))
    parts.append('</section><section id="missing"><h2>รายการที่ขาด</h2>')
    parts.append('<ul>' + ''.join(f'<li>{esc(item)}</li>' for item in missing) + '</ul>' if missing else '<p>ไม่มี · พบผลและรูปที่ต้องใช้ครบทั้ง 12 งาน</p>')
    parts.append('</section><footer>รายงานสรุปจาก saved outputs · สร้างซ้ำด้วย make_report.py · ไม่มีข้อมูลรถรายแถวในรายงาน</footer></main></body></html>')
    return ''.join(parts), winners, main_rows


def adjustment_pairs(runs, dataset):
    for model in MODELS:
        pair = {r['variant']: metric(r) for r in runs if r['dataset'] == dataset and r['model'] == model}
        yield model, pair.get('base'), pair.get('adj')


def duration(seconds):
    if seconds is None:
        return MISSING
    hours, rest = divmod(round(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'img').mkdir(exist_ok=True)
    paths = [line.strip().replace('\\', '/') for line in (ROOT / 'run_order.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    logs = {}
    global_missing = []
    try:
        with (ROOT / 'run_log.csv').open(encoding='utf-8-sig', newline='') as source:
            logs = {row['notebook'].replace('\\', '/'): row for row in csv.DictReader(source)}
    except (OSError, KeyError):
        global_missing.append('อ่าน run_log.csv ไม่ได้')
    runs = [parse_notebook(p, logs.get(p)) for p in paths]
    expected = {f'{m}/{m}_{d}{suffix}.ipynb' for m in MODELS for d in DATASETS for suffix in ('', '_adj')}
    for path in sorted(expected - set(paths)):
        run = parse_notebook(path, logs.get(path))
        run['missing'].append('ไม่มีใน run_order.txt')
        runs.append(run)
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate notebook in run_order.txt')
    missing = global_missing + [r['run_id'] + ': ' + item for r in runs for item in r['missing']]
    page, winners, main_rows = render(runs, missing)
    payload = {'schema_version': 1, 'source': 'saved notebook outputs; no execution',
               'cv_precision': 'mean and SD from printed fold summary, rounded to integer baht',
               'runs': runs, 'total_seconds': sum((r['runtime'] or {}).get('seconds') or 0 for r in runs), 'missing': missing}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    readme = ['# ผลฝึกโมเดลราคารถ v5', '', 'สรุปจาก output ที่บันทึกไว้ 12 งาน (DT / RF / XGB × 2 ชุดข้อมูล × base / adj) เลือกผู้ชนะด้วย test MAE ภายในแต่ละชุดข้อมูล', '',
              '| ชุดข้อมูล | ผู้ชนะ | Test MAE (บาท) | Test R² |', '|---|---|---:|---:|']
    for dataset, winner in winners.items():
        readme.append(f'| {dataset} | {winner["model"] + " / " + winner["variant"] if winner else MISSING} | {fmt(metric(winner)) if winner else MISSING} | {fmt(metric(winner, "R2"), 4) if winner else MISSING} |')
    readme.extend(['', '**รายงานฉบับเต็ม:** [docs/index.html](docs/index.html) · [GitHub Pages]({{PAGES_URL}})', '',
                   'สองชุดใช้แถว test ต่างกัน ควรเทียบภายในชุดก่อน รายงานมีผลการปรับข้อมูล, ablation, importance, overfit, learning curves และเวลารัน', '',
                   'สร้างซ้ำจากโฟลเดอร์นี้: `python -X utf8 make_report.py` (อ่าน saved outputs เท่านั้น ไม่รัน notebook; ใช้ Python stdlib)', '',
                   'เผยแพร่ GitHub Pages: เลือก branch ที่มีรายงานและโฟลเดอร์ `/docs` แล้วแทน `{{PAGES_URL}}` ด้วย URL จริง เผยแพร่เฉพาะไฟล์สคริปต์/รายงานที่อนุญาต และอย่านำ notebook หรือไฟล์ข้อมูลเข้ารายงานสาธารณะ', ''])
    readme = '\n'.join(readme)
    for name, content in [('index.html', page), ('results.json', encoded), ('README.md', readme)]:
        if FORBIDDEN.search(content):
            raise ValueError('public output safety check failed: ' + name)
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    (OUT / 'results.json').write_text(encoded, encoding='utf-8')
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    # Remove only stale plots owned by this generator, never unrelated files.
    used = {rel for r in runs for rel in r['plots'].values()}
    for plot in (OUT / 'img').glob('*.png'):
        if re.fullmatch(r'(DT|RF|XGB)_(new_cardb|merge50minC)(_adj)?_(learning|permutation|model_based)\.png', plot.name) and 'img/' + plot.name not in used:
            plot.unlink()
    size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if size >= 15_000_000:
        raise ValueError('docs must be smaller than 15 MB')
    print(f'Created report: {len(runs)} runs, {len(used)} figures, {size:,} bytes; missing items: {len(missing)}')
    print('Model | Dataset | Variant | Test R2 | RMSE | MAE | MAPE | CV MAE +/- SD')
    for row in main_rows:
        print(' | '.join(map(str, row)))


if __name__ == '__main__':
    main()
