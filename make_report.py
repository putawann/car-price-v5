"""Build a public, offline report from saved notebook outputs; never execute cells.

Run from this worktree: python -X utf8 make_report.py
Only curated aggregates, PASS/WARN messages and three plot types are exported.
Reads installed estimator defaults without fitting. Input files are never modified.
"""
from __future__ import annotations

import base64
import ast
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
PAGES_URL = 'https://putawann.github.io/car-price-v5/'
FORBIDDEN = re.compile(r'[A-Za-z]:[/\\]|putaw|(?:api[_-]?key|access[_-]?token|secret)\s*[:=]', re.I)

PARAM_MISSING = 'ไม่พบในผลรัน'
OLD = 'old'  # generation of notebooks that select with the 1-SE rule
NEW_SOURCE = {m: 'GridSearchCV, ผู้ชนะ = CV MAE ต่ำสุด (refit=MAE)' for m in ('DT', 'RF')}
NEW_SOURCE['XGB'] = 'ค้นทุกชุดด้วย 5-fold CV (โค้ดเอง), ผู้ชนะ = CV MAE ต่ำสุด'
ALPHA_SOURCE = 'เลือกจากทุก alpha บน pruning path ร่วมกับ grid ด้วย 5-fold CV (CART)'
PATH_ALPHAS = 'ทุก alpha บน pruning path'
ES_SOURCE = 'จำนวนรอบจาก early stopping ต่อ fold (ค่าเฉลี่ย)'
FIXED_SOURCE = 'กำหนดคงที่ในโค้ด'
PARAM_MEANINGS = {
    'criterion': 'เกณฑ์วัดคุณภาพการแตกกิ่ง', 'splitter': 'วิธีเลือกจุดแตกกิ่ง',
    'max_depth': 'ความลึกสูงสุดของต้นไม้', 'min_samples_split': 'จำนวนแถวขั้นต่ำก่อนแตกกิ่ง',
    'min_samples_leaf': 'จำนวนแถวขั้นต่ำต่อใบ', 'min_weight_fraction_leaf': 'สัดส่วนน้ำหนักขั้นต่ำต่อใบ',
    'max_features': 'จำนวนหรือสัดส่วนตัวแปรที่ใช้เลือกจุดแตกกิ่ง',
    'random_state': 'เลขสุ่มสำหรับทำผลซ้ำ', 'max_leaf_nodes': 'จำนวนใบสูงสุด',
    'min_impurity_decrease': 'การลดความไม่บริสุทธิ์ขั้นต่ำก่อนแตกกิ่ง',
    'ccp_alpha': 'ความแรงของการตัดกิ่ง', 'monotonic_cst': 'ข้อบังคับทิศทางความสัมพันธ์',
    'n_estimators': 'จำนวนต้นไม้หรือรอบ boosting', 'bootstrap': 'การสุ่มแถวแบบใส่คืน',
    'oob_score': 'การวัดผลด้วยแถวที่ไม่ถูกสุ่มเข้าแต่ละต้น', 'n_jobs': 'จำนวนงานหรือ thread ของโมเดล',
    'verbose': 'ระดับข้อความความคืบหน้า', 'warm_start': 'การต่อเติมโมเดลเดิม',
    'max_samples': 'ขนาดตัวอย่าง bootstrap', 'objective': 'ฟังก์ชันเป้าหมายในการฝึก',
    'learning_rate': 'ขนาดก้าวของ boosting', 'min_child_weight': 'น้ำหนัก Hessian ขั้นต่ำของโหนดลูก',
    'reg_lambda': 'บทลงโทษ L2 ของค่าในใบ', 'reg_alpha': 'บทลงโทษ L1 ของค่าในใบ',
    'gamma': 'การลด loss ขั้นต่ำก่อนแตกกิ่ง', 'max_delta_step': 'ขนาดการปรับค่าในใบสูงสุด',
    'subsample': 'สัดส่วนแถวที่สุ่มใช้แต่ละรอบ', 'colsample_bytree': 'สัดส่วนคอลัมน์ที่สุ่มต่อหนึ่งต้น',
    'colsample_bylevel': 'สัดส่วนคอลัมน์ที่สุ่มต่อระดับ', 'colsample_bynode': 'สัดส่วนคอลัมน์ที่สุ่มต่อโหนด',
    'tree_method': 'วิธีสร้างต้นไม้', 'max_bin': 'จำนวนช่วง histogram สูงสุด',
    'grow_policy': 'ลำดับการขยายต้นไม้', 'booster': 'ชนิดตัวเรียนรู้ย่อย',
    'eval_metric': 'ตัววัดระหว่างฝึกเมื่อมี eval_set', 'early_stopping_rounds': 'จำนวนรอบรอก่อนหยุดฝึก',
    'base_score': 'ค่าทำนายเริ่มต้น', 'callbacks': 'ฟังก์ชันเสริมระหว่างฝึก',
    'device': 'อุปกรณ์ประมวลผล', 'enable_categorical': 'การอนุญาตข้อมูลหมวดหมู่แบบ native',
    'feature_types': 'ชนิดข้อมูลของตัวแปร', 'feature_weights': 'น้ำหนักในการสุ่มตัวแปร',
    'importance_type': 'วิธีสรุปความสำคัญตัวแปร', 'interaction_constraints': 'ข้อจำกัดปฏิสัมพันธ์ของตัวแปร',
    'max_cat_threshold': 'จำนวนหมวดหมู่สูงสุดที่พิจารณาจุดแบ่ง',
    'max_cat_to_onehot': 'เกณฑ์เลือก one-hot สำหรับข้อมูลหมวดหมู่ native',
    'max_leaves': 'จำนวนใบสูงสุด', 'missing': 'ค่าที่ใช้แทนข้อมูลสูญหาย',
    'monotone_constraints': 'ข้อบังคับทิศทางความสัมพันธ์', 'multi_strategy': 'วิธีเรียนรู้หลายเป้าหมาย',
    'num_parallel_tree': 'จำนวนต้นไม้ขนานต่อรอบ', 'sampling_method': 'วิธีสุ่มแถวฝึก',
    'scale_pos_weight': 'น้ำหนักสัมพัทธ์ของคลาสบวก', 'validate_parameters': 'การตรวจชื่อพารามิเตอร์',
    'verbosity': 'ระดับข้อความของ XGBoost',
}


def estimator_class(model):
    # Imports and construction only: never load training data or fit a model.
    if model == 'XGB':
        import xgboost
        return xgboost.XGBRegressor, 'xgboost ' + xgboost.__version__
    import sklearn
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    return {'DT': DecisionTreeRegressor, 'RF': RandomForestRegressor}[model], 'sklearn ' + sklearn.__version__


def static_value(node, names):
    """Small allowlist of value expressions; no notebook code is evaluated."""
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.UnaryOp)):
        return ast.literal_eval(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = static_value(node.left, names), static_value(node.right, names)
        return [v / right for v in left] if isinstance(left, list) else left / right
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'np'
            and node.func.attr == 'arange' and not node.keywords):
        return list(range(*(static_value(a, names) for a in node.args)))
    raise ValueError('unsupported static parameter expression')


def param_display(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else str(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def collect_params(cells, run):
    """Read each notebook's AST and saved summaries, never execute its cells."""
    sources = [joined(c.get('source', '')) for c in cells if c.get('cell_type') == 'code']
    constants, comments, fixed, fixed_comments, grid = {}, {}, {}, {}, {}
    trees = [(s, ast.parse(s)) for s in sources]
    for source, tree in trees:
        if 'ค่าคงที่ของการทดลอง' in source and re.search(r'^SEED\s*=', source, re.M):
            lines = source.splitlines()
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                targets = node.targets[0]
                names = [targets.id] if isinstance(targets, ast.Name) else [n.id for n in targets.elts]
                try:
                    value = static_value(node.value, constants)
                    values = [value] if isinstance(targets, ast.Name) else value
                except (ValueError, KeyError, TypeError):
                    values = [PARAM_MISSING] * len(names)
                for name, value in zip(names, values):
                    constants[name] = value
                    comments[name] = lines[node.end_lineno - 1].partition('#')[2].strip()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PARAM_GRID' for t in node.targets):
                grid.update(static_value(node.value, constants))
            if isinstance(node, ast.FunctionDef) and node.name == 'make_model':
                call = next(n.value for n in node.body if isinstance(n, ast.Return))
                for kw in call.keywords:
                    try:
                        fixed[kw.arg] = static_value(kw.value, constants)
                    except (ValueError, KeyError, TypeError):
                        fixed[kw.arg] = PARAM_MISSING
                    fixed_comments[kw.arg] = source.splitlines()[kw.end_lineno - 1].partition('#')[2].strip()
    best = dict((run['result'] or {}).get('best_params', {}))
    edge = {}
    alpha_printed = None
    new = run['generation'] != OLD
    alpha_info = ''
    for cell in cells:
        text = output_text(cell)
        if re.search(r'^=== \[\d+\] สรุปผลกฎ 1-SE', text, re.M):
            selected = text.partition('ชุดที่กฎ 1-SE เลือก')[2]
            found = re.search(r'^\s*params\s*:\s*(\{[^\n]+\})', selected, re.M)
            if found:
                for key, value in ast.literal_eval(found[1]).items():
                    if key in best and best[key] != value:
                        raise ValueError('RESULT_JSON and 1-SE summary disagree: ' + run['run_id'] + ' ' + key)
                    best.setdefault(key, value)
        if re.search(r'^=== \[\d+\] ccp 5:', text, re.M):
            block = text.partition('ccp_alpha ที่ใส่ใน PARAM_GRID:')[2]
            alpha_printed = [float(v) for v in re.findall(r'^\s+([\d.]+e[+-]\d+)\s*$', block, re.M | re.I)]
        if re.search(r'^=== \[\d+\] ตรวจขอบ grid', text, re.M):
            if new:
                block = text.partition('ผู้ชนะ (CV MAE ต่ำสุด)')[2].partition('สรุป:')[0]
            else:
                block = text.partition('ผู้ชนะ 1-SE')[2].partition('ชุด CV MAE ต่ำสุด')[0]
            edge.update(re.findall(r'^\s*(\w+)\s*:.*?->\s*(.+)$', block, re.M))
            found = re.search(r'^(?:PASS|WARN): early stopping[^\n]*', text, re.M)
            if new and found:
                edge['n_estimators'] = found[0]
            for table_ in cell_tables(cell):
                for row in records(table_):
                    if 'parameter' in row and '1-SE winner check' in row:
                        edge[row['parameter']] = row['1-SE winner check'] + ' (' + row['1-SE winner position'] + ')'
                    elif new and 'parameter' in row and 'check' in row:
                        edge[row['parameter']] = row['check'] + ' (' + row['position'] + ')'
        found = re.search(r'alpha ที่ลอง\s*:\s*([\d,]+) ถึง ([\d,]+) ต่อคู่, รวม ([\d,]+)', text)
        if new and found:
            alpha_info = f' (ลอง {found[1]} ถึง {found[2]} ค่าต่อคู่ รวม {found[3]} ค่า)'
    if run['model'] == 'DT':
        grid['ccp_alpha'] = PATH_ALPHAS if new else (alpha_printed or PARAM_MISSING)
    cls, version = estimator_class(run['model'])
    # Missing explicit settings must never silently become library defaults.
    params = cls(**{k: v for k, v in fixed.items() if v != PARAM_MISSING}).get_params()
    rows = []

    def add(name, value, source, explanation, kind='fixed', **extra):
        if value == PARAM_MISSING:
            run['missing'].append('params: ' + name + ' · ' + PARAM_MISSING)
        rows.append({'parameter': name, 'value': value, 'source': source,
                     'explanation': explanation, 'kind': kind, **extra})

    for name, default in params.items():
        meaning = PARAM_MEANINGS.get(name, 'การตั้งค่า ' + name)
        if new and name == 'n_estimators' and run['model'] == 'XGB':
            status = edge.get(name, PARAM_MISSING)
            if status == PARAM_MISSING:
                run['missing'].append('params: ตรวจเพดานรอบ ' + name + ' · ' + PARAM_MISSING)
            add(name, best.get(name, PARAM_MISSING), ES_SOURCE,
                meaning + f'; early stopping ต่อ fold (เพดาน MAX_ROUNDS, หยุดเมื่อ MAE บนแถวตรวจของ fold ไม่ดีขึ้นต่อเนื่อง EARLY_STOPPING_ROUNDS รอบ) จำนวนรอบของชุด = ค่าเฉลี่ยของ fold ปัดเป็นจำนวนเต็ม; CV MAE แบบนี้อาจต่ำกว่าจริงเล็กน้อยเพราะแถวตรวจเป็นตัวกำหนดจุดหยุด; ตรวจเพดาน: {status}।',
                'tuned', search_range='early stopping ต่อ fold', grid_edge=status, candidate_source='early stopping')
        elif name in grid:
            candidates = grid[name]
            if new:
                source = NEW_SOURCE[run['model']] + ' (ช่วงที่ค้น: ' + param_display(candidates) + ')'
                explanation = meaning + '; เลือกค่าของผู้ชนะที่ CV MAE เฉลี่ยต่ำสุด'
                if name == 'ccp_alpha':
                    source = NEW_SOURCE['DT'] + '; ' + ALPHA_SOURCE
                    explanation += '; ' + ALPHA_SOURCE + alpha_info
            else:
                source = 'GridSearchCV + กฎ 1-SE (ช่วงที่ค้น: ' + param_display(candidates) + ')'
                explanation = meaning + '; เลือกค่าของผู้ชนะที่มีใบน้อยที่สุดในช่วง 1-SE'
                if name == 'ccp_alpha':
                    explanation += '; ตัวเลือกจาก ccp path บน train และ CV (ช่วงแสดงตามผลพิมพ์ 4 หลักนัยสำคัญ)'
            status = edge.get(name, PARAM_MISSING)
            if status == PARAM_MISSING:
                run['missing'].append('params: ตรวจขอบ grid ' + name + ' · ' + PARAM_MISSING)
            explanation += '; ตรวจขอบ: ' + status + '।'
            add(name, best.get(name, PARAM_MISSING), source, explanation, 'tuned',
                search_range=candidates, grid_edge=status,
                candidate_source=('pruning path (CART)' if new else 'เลือกจาก ccp path') if name == 'ccp_alpha' else 'PARAM_GRID')
        elif name in fixed:
            source = 'กราฟ OOB' if name == 'n_estimators' and run['model'] == 'RF' and 'OOB' in fixed_comments[name] else FIXED_SOURCE
            comment = fixed_comments[name]
            explanation = (comment if comment else meaning) + '; กำหนดคงที่ ไม่ได้ค้นใน grid ของไฟล์นี้।'
            add(name, fixed[name], source, explanation)
        else:
            value = 'NaN' if isinstance(default, float) and math.isnan(default) else default
            explanation = meaning + '; คงค่าเริ่มต้นของไลบรารีโดยไม่ได้ปรับจูน'
            if default is None and run['model'] == 'XGB':
                explanation += ' (None คือไม่ได้ระบุใน Python wrapper ให้ backend ตัดสิน)'
            if name in ('enable_categorical', 'max_cat_threshold', 'max_cat_to_onehot'):
                explanation += '; pipeline ใช้ FoldOneHot จึงไม่ได้ส่งหมวดหมู่ native'
            add(name, value, 'ค่าเริ่มต้นของไลบรารี (' + version + ')', explanation + '।', 'default')
    for name, value in constants.items():
        meaning = {'RUN_ID': 'ชื่อการทดลอง', 'DATASET': 'ชื่อชุดข้อมูล', 'VARIANT': 'รูปแบบ base หรือ adj',
                   'MAX_CAR_AGE': 'อายุรถสูงสุดที่ยอมรับในการกรองข้อมูล',
                   'MODEL_ALIASES': 'กฎรวมชื่อรุ่นที่สะกดต่างกัน',
                   'PERM_AXIS': 'ช่วงแกนกราฟ permutation importance',
                   'SHARE_AXIS': 'ช่วงแกนสัดส่วนความสำคัญตัวแปร'}.get(name, 'ค่าประกอบการทดลอง ' + name)
        explanation = comments[name] or meaning + '; กำหนดคงที่โดยไม่ได้ปรับจูน'
        add(name, value, FIXED_SOURCE, explanation + '।')
    if run['model'] == 'DT' and new:
        add('ccp_alpha.candidates', PATH_ALPHAS, 'pruning path (CART)',
            'ใช้ทุก alpha บน pruning path ของแต่ละคู่ (max_depth, min_samples_leaf) ที่สร้างจาก train ทั้งชุด ไม่จำกัดจำนวนและไม่ใช้เกณฑ์ใบขั้นต่ำ แล้ววัดทุก alpha ด้วย 5-fold CV ร่วมกับ grid' + alpha_info + '।')
    elif run['model'] == 'DT':
        add('ccp_alpha.candidates', grid['ccp_alpha'], 'เลือกจาก ccp path',
            'เลือก alpha บวกบนสเกล log ที่ต้นไม้ยังเกิน MIN_LEAVES แล้วใช้ CV คัดเพื่อนบ้าน CCP_NEIGHBOURS พร้อมเพิ่ม 0 ก่อนค้น grid; ค่าตัวเลือกแสดงตามผลพิมพ์ 4 หลักนัยสำคัญ।')

    # Pipeline/search arguments are read from the final pipeline, not diagnostics/ablation.
    final_pipeline = next((tree for source, tree in trees if any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pipeline' for t in n.targets)
        for n in tree.body)), None) or next((tree for source, tree in trees if new and any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'best_pipeline' for t in n.targets)
        for n in tree.body)), None)
    calls = list(ast.walk(final_pipeline)) if final_pipeline else []
    encoder = next((n for n in calls if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'FoldOneHot'), None)
    target = next((n for n in calls if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'TransformedTargetRegressor'), None)
    scaled = next((static_value(k.value, constants) for k in encoder.keywords if k.arg == 'scale_numeric'), False) if encoder else PARAM_MISSING
    add('pipeline.target_transform', 'np.log / np.exp' if target else ('identity (บาท)' if final_pipeline else PARAM_MISSING), FIXED_SOURCE,
        'แปลงราคา log แล้วคืนด้วย exp ก่อนวัดผลเป็นบาท; กำหนดคงที่ไม่ได้ปรับจูน।' if target else 'ใช้ราคาเป็นบาทโดยไม่แปลงเป้าหมาย; กำหนดคงที่ไม่ได้ปรับจูน।')
    if target:
        for kw in target.keywords:
            if kw.arg in ('func', 'inverse_func') and ast.unparse(kw.value) not in ('np.log', 'np.exp'):
                raise ValueError('unexpected target transform')
        from sklearn.compose import TransformedTargetRegressor
        for name, value in TransformedTargetRegressor().get_params(deep=False).items():
            if name not in ('regressor', 'func', 'inverse_func'):
                add('target.' + name, value, 'ค่าเริ่มต้นของไลบรารี (' + estimator_class('DT')[1] + ')',
                    ('ตรวจว่าฟังก์ชันแปลงและแปลงกลับสอดคล้องกัน' if name == 'check_inverse' else 'ตัวแปลงเป้าหมายแบบ estimator ซึ่งไม่ได้ส่งเพราะใช้คู่ฟังก์ชัน log/exp') + '; คงค่าเริ่มต้นโดยไม่ได้ปรับจูน।', 'default')
    add('FoldOneHot.scale_numeric', scaled, FIXED_SOURCE, 'กำหนดว่าจะปรับสเกลเฉพาะตัวเลขในแต่ละ fold หรือไม่ตาม pipeline; กำหนดคงที่ไม่ได้ปรับจูน।')
    if scaled is True:
        add('scaler', '(x - mean_train_fold) / std_train_fold; ddof=0; std=0 → 1', FIXED_SOURCE,
            'ใช้ mean/std จากแถวฝึกของแต่ละ fold เท่านั้นและไม่ scale dummy ตามโค้ดเพื่อป้องกันข้อมูลรั่ว।')
    else:
        add('scaler', 'ไม่มี', FIXED_SOURCE, 'ส่งค่าตัวเลขเดิมเข้าโมเดลตามโค้ดโดยไม่ได้ปรับจูนการ scale।')
    add('encoder', 'FoldOneHot; numeric=float; dummy=float; dense', FIXED_SOURCE, 'แปลงหมวดหมู่เป็น 0/1 และต่อกับตัวเลขโดย fit ใหม่ในแต่ละ fold เพื่อป้องกันข้อมูลรั่ว।')
    add('FoldOneHot.reference', 'most frequent; ties: name ascending, stable', FIXED_SOURCE, 'ทิ้งระดับที่พบบ่อยที่สุดเพื่อลดคอลัมน์ซ้ำซ้อนและใช้ชื่อเรียงตัดสินเมื่อความถี่เสมอกันตามโค้ด।')
    add('FoldOneHot.unseen', 'all zeros', FIXED_SOURCE, 'ระดับที่ไม่เคยพบในแถวฝึกได้ศูนย์ทุกคอลัมน์ตามโค้ดเพื่อให้แปลงข้อมูลใหม่ได้।')
    split_source = next((s for s in sources if 'train_test_split(' in s), '')
    add('split.strategy', 'forced-train 1 / brand|model' if 'forced =' in split_source else ('stratify brand|model' if 'stratify=model_key' in split_source else PARAM_MISSING), FIXED_SOURCE,
        'ใช้กติกาแบ่งชุดของไฟล์นี้ร่วมกับ TRAIN_SIZE/VAL_SIZE/TEST_SIZE และ SEED ตามโค้ดโดยไม่ได้ปรับจูน।')
    for source, tree in trees:
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id in ('GridSearchCV', 'KFold'):
                for kw in call.keywords:
                    name = call.func.id + '.' + kw.arg
                    if kw.arg in ('param_grid', 'cv'):
                        continue
                    try:
                        value = static_value(kw.value, constants)
                    except (ValueError, KeyError, TypeError):
                        value = ast.unparse(kw.value)
                    meaning = {'shuffle': 'สลับลำดับแถวก่อนแบ่ง fold เพื่อให้กระจายข้อมูล',
                               'random_state': 'เลขสุ่มของการแบ่ง fold เพื่อทำผลซ้ำ',
                               'scoring': ('ตัววัด CV หน่วยบาท ใช้ MAE เลือกผู้ชนะ' if new
                                           else 'ตัววัด CV โดยใช้ MAE หน่วยบาทเลือกผู้ชนะและใช้ใบวัดความซับซ้อน'),
                               'refit': ('เลือกผู้ชนะที่ mean CV MAE ต่ำสุดแล้วฝึกผู้ชนะบน train ทั้งชุด' if new
                                         else 'ฟังก์ชันเลือกชุดค่าด้วยกฎ 1-SE แล้วฝึกผู้ชนะบน train ทั้งชุด'),
                               'return_train_score': 'เก็บคะแนน train เพื่อเทียบกับ CV',
                               'error_score': 'ให้หยุดเมื่อการฝึกเกิดข้อผิดพลาด',
                               'n_jobs': 'จำนวนงานค้นหาที่ทำพร้อมกัน',
                               'pre_dispatch': 'จำนวนงานที่ส่งล่วงหน้าเพื่อคุมหน่วยความจำ',
                               'verbose': 'ระดับข้อความความคืบหน้าของการค้นหา'}.get(kw.arg, kw.arg)
                    add(name, value, FIXED_SOURCE, meaning + '; กำหนดคงที่ตามโค้ดโดยไม่ได้ปรับจูน।')
    if new:
        add('selection.minCV', 'min mean CV MAE; เสมอ → index แรก', FIXED_SOURCE,
            'ผู้ชนะคือชุดที่ CV MAE เฉลี่ยต่ำสุด (refit=MAE) ไม่มีเกณฑ์จำนวนใบ; จำนวนใบแสดงเป็นข้อมูลประกอบเท่านั้น।')
    else:
        rule = 'min CV MAE + SD(ddof=0)/sqrt(N_FOLDS); min leaves → CV MAE → index'
        add('selection.1SE', rule, FIXED_SOURCE, 'เลือกใบน้อยที่สุดในช่วงหนึ่ง SE จาก MAE ต่ำสุด; DT นับใบบน train ทั้งชุด ส่วน RF/XGB ใช้ค่าเฉลี่ยใบรวมจาก CV ตามโค้ด।')
    rows.sort(key=lambda row: {'tuned': 0, 'fixed': 1, 'default': 2}[row['kind']])
    return rows


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
           'dataset': dataset, 'variant': 'adj' if adj else 'base', 'generation': None,
           'result': None, 'importance': None, 'cv': None, 'cv_tables': [],
           'ablation': [], 'overfit': [], 'plots': {}, 'data': {}, 'params': [], 'missing': []}
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
    # Generation comes from the saved outputs: only the old runs print the 1-SE selection rule.
    run['generation'] = OLD if any('1-SE' in output_text(c) for c in cells) else ((log or {}).get('start') or 'new')[:10]
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
                        for key, col in (('rmse', 'validation RMSE'), ('mape', 'validation MAPE')):
                            if col in means and col in sd:
                                run['cv'].update({key + '_mean': means[col], key + '_sd': sd[col]})
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
    run['params'] = collect_params(cells, run)
    return run


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=0):
    return MISSING if value is None else f'{value:,.{digits}f}'


def pct(value, fraction=False):
    return MISSING if value is None else f'{value * (100 if fraction else 1):,.2f}%'


def metric(run, key='MAE'):
    return (run.get('result') or {}).get('test', {}).get(key)


def is_new(run):
    return run['generation'] not in (None, OLD)


def gen_label(run):
    return MISSING if run['generation'] is None else ('รอบใหม่ ' + run['generation'] if is_new(run) else 'รอบเก่า (กฎ 1-SE)')


def label(run):
    return f"{run['model']} · {run['dataset']} · {run['variant']} · {gen_label(run)}"


def latest_generation(runs, dataset):
    gens = {r['generation'] for r in runs if r['dataset'] == dataset and metric(r) is not None} - {OLD}
    return max(gens, default=OLD)


def old_note(runs):
    old = [r for r in runs if not is_new(r)]
    rows = ' / '.join(fmt(v) for v in sorted({r['data'].get('rows') for r in old} - {None}))
    return (f'<p class="note"><strong>ผลรอบเก่า {len(old)} งาน:</strong> ใช้ข้อมูลรุ่นเก่า ({rows} แถว) และเลือกพารามิเตอร์ด้วยกฎ 1-SE '
            'เทียบกับรอบใหม่ตรงๆ ไม่ได้และไม่นำมาจัดอันดับร่วมกัน · <a href="index.html">กลับหน้าหลัก (ผลรอบใหม่)</a></p>')


def winner_map(runs):
    winners = {}
    for dataset in DATASETS:
        # Never rank across generations: only the newest generation present for the dataset competes.
        latest = latest_generation(runs, dataset)
        eligible = [r for r in runs if r['dataset'] == dataset and metric(r) is not None and r['generation'] == latest]
        winners[dataset] = min(eligible, key=metric) if eligible else None
    return winners


def cv_text(run):
    cv = run.get('cv')
    return f"{fmt(cv['mean'])} ± {fmt(cv['sd'])}" if cv else MISSING


def cv_extra_text(run, key):
    cv = run.get('cv')
    if not cv:
        return MISSING
    if key + '_mean' not in cv:
        return PARAM_MISSING
    if key == 'mape':
        return f"{pct(cv['mape_mean'], True)} ± {pct(cv['mape_sd'], True)}"
    return f"{fmt(cv[key + '_mean'])} ± {fmt(cv[key + '_sd'])}"


def cv_fold_table(run):
    rows = next((t['rows'] for t in run['cv_tables'] if t['kind'] == 'fold_scores'), None)
    if not rows:
        return f'<p>CV รายfold: {MISSING}</p>'
    columns = [c for c in ('train MAE', 'validation MAE', 'train RMSE', 'validation RMSE', 'train MAPE', 'validation MAPE') if c in rows[0]]
    body = [[f"fold {fmt(r.get(''))}" if isinstance(r.get(''), float) else r.get('')] +
            [pct(r.get(c), True) if 'MAPE' in c else fmt(r.get(c)) for c in columns] for r in rows]
    html_ = '<p class="muted">CV 5 folds ใน train: ค่าเฉลี่ยและ SD (ddof=0) ตามตารางที่พิมพ์ในไฟล์</p>' + table(['', *columns], body)
    if 'validation RMSE' not in rows[0]:
        html_ += f'<p>CV RMSE / MAPE: {PARAM_MISSING}</p>'
    return html_


def table(headers, rows, best=()):
    body = []
    for i, row in enumerate(rows):
        css = ' class="best"' if i in best else ''
        body.append(f'<tr{css}>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>')
    return '<div class="table-wrap" role="region" aria-label="ตารางเลื่อนแนวนอนได้" tabindex="0"><table><thead><tr>' + ''.join(
        f'<th scope="col">{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


def latest_results(runs):
    latest = [r for r in runs if is_new(r) and metric(r) is not None
              and r['generation'] == latest_generation(runs, r['dataset'])]
    if not latest:
        return ''
    parts = ['<section id="latest-results"><h2>ผลรันรอบใหม่ · DT / RF / XGB</h2>']
    for dataset in DATASETS:
        group = [r for r in latest if r['dataset'] == dataset]
        if not group:
            continue
        best = min(group, key=metric)
        parts.append(f'<h3>{esc(dataset)} · {esc(gen_label(group[0]))}</h3>')
        parts.append(f'<p>ข้อมูล {fmt(group[0]["data"].get("rows"))} แถว · '
                     'เลือกพารามิเตอร์ด้วย CV MAE ต่ำสุดบน train · '
                     'ผลด้านล่างอ่านจากการรันที่บันทึกในโน้ตบุ๊ก</p>')
        parts.append('<h3>5-fold cross-validation บน train</h3>'
                     '<p class="muted">คะแนนบนแถวตรวจของแต่ละ fold · ค่าเฉลี่ย ± SD ข้าม 5 folds (ddof=0) · '
                     'MAE/RMSE หน่วยบาท; MAPE แสดงเป็นเปอร์เซ็นต์ · ค่ายิ่งต่ำยิ่งดี</p>')
        parts.append(table(['โมเดล / แบบ', 'CV MAE ± SD (บาท)', 'CV RMSE ± SD (บาท)', 'CV MAPE ± SD'],
                           [[f'{r["model"]} / {r["variant"]}', cv_text(r),
                             cv_extra_text(r, 'rmse'), cv_extra_text(r, 'mape')] for r in group]))
        parts.append('<h3>ผลบนชุด test ที่กันไว้</h3>')
        parts.append(table(['โมเดล / แบบ', 'Test MAE (บาท)', 'Test RMSE (บาท)', 'Test MAPE', 'Test R²'],
                           [[f'{r["model"]} / {r["variant"]}', fmt(metric(r)), fmt(metric(r, 'RMSE')),
                             pct(metric(r, 'MAPE'), True), fmt(metric(r, 'R2'), 4)] for r in group],
                           [group.index(best)]))
        for r in group:
            parts.append(f'<details><summary>{esc(r["model"])} / {esc(r["variant"])} · ผลครบทั้ง 5 folds</summary>'
                         + cv_fold_table(r) + '</details>')
    parts.append('<p class="muted">ผลรอบเก่าแยกไว้ในเมนู "ผลรอบเก่า" ด้านบน เพื่อไม่เทียบข้อมูลและวิธีเลือกโมเดลคนละรอบ</p></section>')
    return ''.join(parts)


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
@media(max-width:600px){main{padding:12px 16px 40px}section{padding:16px 12px;border-radius:12px}.cards,.plots{grid-template-columns:1fr}th,td{padding:9px}.value{font-size:1.5rem}h2{font-size:1.25rem}}
@media(prefers-color-scheme:dark){:root{--bg:#101a23;--card:#182630;--ink:#e0ecf3;--muted:#a4b8c5;--line:#344957;--accent:#73d7c4;--best:#173e35;--warn:#ffbf85}}
#parameters table{table-layout:fixed;min-width:740px}#parameters th,#parameters td{text-align:left;white-space:normal;overflow-wrap:anywhere;vertical-align:top}#parameters th:nth-child(1){width:19%}#parameters th:nth-child(2){width:20%}#parameters th:nth-child(3){width:27%}#parameters th:nth-child(4){width:34%}
"""


def render(all_runs, missing, old=False):
    """One page: the new-generation runs (index.html) or every other run (old.html)."""
    runs = [r for r in all_runs if (not is_new(r)) == old]
    n_old = sum(not is_new(r) for r in all_runs)
    winners = {d: w for d, w in winner_map(runs).items() if any(r['dataset'] == d for r in runs)}
    title = 'ผลรอบเก่า · ผลการฝึกโมเดลราคารถ v5' if old else 'ผลการฝึกโมเดลราคารถ · v5'
    desc = (f'ผลรอบเก่า {len(runs)} งาน (ข้อมูลรุ่นเก่า กฎ 1-SE) จาก output ที่บันทึกไว้' if old
            else f'ผลฝึกโมเดลราคารถ v5 รอบล่าสุด {len(runs)} งาน (DT / RF / XGB) จาก output ที่บันทึกไว้')
    parts = ['<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<meta name="description" content="{esc(desc)}"><title>{esc(title)}</title>',
             '<style>' + CSS + '</style></head><body><main><header>']
    if old:
        parts.append(f'<p class="eyebrow">TRAINING RESULTS / V5 / OLD</p><h1>ผลรอบเก่า</h1><p>{len(runs)} งาน · DT / RF / XGB · base และ adj</p>')
        parts.append('<p class="muted">อ่านจาก output ที่บันทึกไว้เท่านั้น ไม่มีการฝึกหรือรัน notebook ใหม่ · หน่วยเงิน: บาท</p>' + old_note(runs) + '<nav aria-label="สารบัญ"><a href="index.html">หน้าหลัก (ผลรอบใหม่)</a>')
        sections = [('summary', 'สรุปสั้น'), ('data', 'ข้อมูลที่ใช้'), ('results', f'ผลทั้ง {len(runs)} งาน'), ('parameters', 'พารามิเตอร์ของแต่ละไฟล์'), ('adjustment', 'ผลการปรับ'),
                    ('importance', 'ความสำคัญตัวแปร'), ('overfit', 'Overfit'), ('learning', 'Learning curves'), ('runtime', 'เวลารัน')]
    else:
        rows_new = ' / '.join(fmt(v) for v in sorted({r['data'].get('rows') for r in runs} - {None}))
        parts.append(f'<p class="eyebrow">TRAINING RESULTS / V5</p><h1>ผลการฝึกโมเดลราคารถ</h1><p>ผลรอบล่าสุด · {len(runs)} งาน · DT / RF / XGB บน merge50minC · {rows_new} แถว · ผู้ชนะ = CV MAE ต่ำสุด</p>')
        parts.append('<p class="muted">อ่านจาก output ที่บันทึกไว้เท่านั้น ไม่มีการฝึกหรือรัน notebook ใหม่ · หน่วยเงิน: บาท</p>')
        parts.append('<p>DT เลือก <code>ccp_alpha</code> จากทุก alpha บน pruning path ร่วมกับ grid ด้วย 5-fold CV; XGB เลือกจำนวนรอบด้วย early stopping ต่อ fold ซึ่งทำให้ CV MAE ของ XGB อาจต่ำกว่าความจริงเล็กน้อย (แถวตรวจเป็นตัวกำหนดจุดหยุด) '
                     f'ผลรอบเก่า {n_old} งานแยกไว้ในเมนู "ผลรอบเก่า ({n_old} งาน)" ด้านล่าง</p><nav aria-label="สารบัญ">')
        sections = [('latest-results', 'รอบใหม่ / 5-fold CV'), ('summary', 'สรุปสั้น'), ('data', 'ข้อมูลที่ใช้'), ('results', f'ผลทั้ง {len(runs)} งาน'), ('parameters', 'พารามิเตอร์ของแต่ละไฟล์'),
                    ('importance', 'ความสำคัญตัวแปร'), ('overfit', 'Overfit'), ('learning', 'Learning curves'), ('runtime', 'เวลารัน')]
    parts.extend(f'<a href="#{key}">{name}</a>' for key, name in sections)
    if not old:
        parts.append(f'<a href="old.html">ผลรอบเก่า ({n_old} งาน)</a>')
    parts.append('</nav></header>' + ('' if old else latest_results(runs)))
    parts.append('<section id="summary"><h2>1. สรุปสั้น</h2><div class="cards">')
    for dataset, winner in winners.items():
        parts.append(f'<article class="card"><h3>{dataset}</h3>')
        if winner:
            imp = (winner['importance'] or {}).get('permutation', {})
            top = sorted((k for k in imp if imp[k]['mean'] is not None), key=lambda k: imp[k]['mean'], reverse=True)[:3]
            parts.append(f'<p><strong>{winner["model"]} / {winner["variant"]}</strong> มี test MAE ต่ำสุด · {esc(gen_label(winner))}</p><div class="value">{fmt(metric(winner))} บาท</div>')
            parts.append('<p>ตัวแปรสำคัญ 3 อันดับของโมเดลนี้: ' + esc(' → '.join(top) or MISSING) + '</p>')
        else:
            parts.append(f'<p>{MISSING}</p>')
        if old:
            pairs = [(m, metric(b) if b else None, metric(a) if a else None, comparable(b, a)) for m, b, a in adjustment_pairs(runs, dataset)]
            helped = [m for m, b, a, ok in pairs if ok and b is not None and a is not None and a < b]
            worse = [m for m, b, a, ok in pairs if ok and b is not None and a is not None and a >= b]
            if any(ok for *_, ok in pairs):
                parts.append(f'<p>adj ลด test MAE: {esc(", ".join(helped) or "ไม่มี")}; ไม่ลด: {esc(", ".join(worse) or "ไม่มี")}</p>')
            else:
                parts.append('<p>adj: เทียบกับ base ไม่ได้ เพราะไม่มี base ของชุดนี้ในรอบเดียวกัน</p>')
        parts.append('</article>')
    parts.append('</div><p class="muted">เลือกผู้ชนะจาก test MAE ภายใน' + ('ผลรอบเก่าของแต่ละชุดข้อมูล' if old else 'รอบล่าสุด') + ' เป็นการสรุปเชิงพรรณนา ไม่ใช่หลักฐานยืนยันผลบนข้อมูลใหม่ ความสำคัญแบบ permutation วัด MAE ที่เพิ่มขึ้นเมื่อสลับตัวแปร ไม่ได้บอกเหตุและผล</p></section>')
    parts.append('<section id="data"><h2>2. ข้อมูลที่ใช้</h2>')
    rows = []
    for dataset in DATASETS:
        for generation in sorted({r['generation'] for r in runs if r['dataset'] == dataset and r['generation']}, key=lambda g: (g != OLD, g), reverse=True):
            group = [r for r in runs if r['dataset'] == dataset and r['generation'] == generation]
            data = next((r['data'] for r in group if r['data'].get('split')), {})
            splits = {s['ชุด']: s['แถว'] for s in data.get('split', [])}
            rows.append([dataset, gen_label(group[0]), fmt(data.get('rows')), fmt(data.get('models')), *(fmt(splits.get(k)) for k in ('train', 'validation', 'test'))])
    parts.append(table(['ชุดข้อมูล', 'รอบ', 'แถว', 'รุ่น (brand|model)', 'Train', 'Validation', 'Test'], rows))
    merge_li = '<li><strong>merge50minC / merge_50min_clean</strong>: ทำความสะอาดด้วยการลบแถวเท่านั้น ไม่รวมชื่อรุ่น; แบ่งแบบ stratify ตาม brand|model</li>'
    cardb_li = '<li><strong>new_cardb</strong>: กฎ v3 กรองอายุรถและรวมชื่อรุ่นที่สะกดต่างกัน; forced-train บังคับ 1 แถวต่อ brand|model เข้า train ก่อนสุ่มเติมโควตาและแบ่ง validation/test</li>'
    parts.append('<ul>' + (merge_li + cardb_li if old else merge_li) + f'<li>{"ทั้งสองชุด" if old else "ชุดนี้"}ใช้ car_age แทน year และตัด color; เป้าหมายคือ price และใช้ 8 ตัวแปร</li></ul>')
    if old:
        parts.append('<p class="note"><strong>เทียบภายในชุดข้อมูลก่อน:</strong> สองชุดใช้แถว test คนละชุดและวิธีแบ่งต่างกัน ค่า error ข้ามชุดจึงไม่ใช่การแข่งบนข้อสอบเดียวกัน</p></section>')
    else:
        parts.append('<p class="note"><strong>ข้อมูลรอบใหม่:</strong> ใช้ merge_50min_clean จำนวน 46,074 แถว เทียบกับ 47,001 แถวของรอบก่อน ถูกลบออก 927 แถวใน 3 ขั้นตอน คือ แถวที่เครื่องยนต์เกิน 3,500 cc จำนวน 2 แถว, แถวจากการตรวจรอบแรก 305 แถว และแถวที่ถูกระบุว่าผิดปกติ 620 แถว ส่วน sha256 ของไฟล์ข้อมูลไม่ได้ตรวจแล้ว</p></section>')
    if old:
        intro = 'เลื่อนตารางแนวนอนได้ · แถวสีเน้น = test MAE ต่ำสุดของแต่ละชุดข้อมูลในผลรอบเก่า (ไม่จัดอันดับร่วมกับรอบใหม่) · CV คือ validation MAE ของโมเดลที่เลือก บน 5 folds ใน train; mean ± SD อ่านจากตารางที่พิมพ์ (ปัดเป็นบาท) · CV RMSE/MAPE ไม่พบในผลรันของรอบเก่า'
    else:
        intro = 'เลื่อนตารางแนวนอนได้ · แถวสีเน้น = test MAE ต่ำสุด · CV คือ validation MAE / RMSE / MAPE ของโมเดลที่เลือก บน 5 folds ใน train; mean ± SD อ่านจากตารางที่พิมพ์ (MAE/RMSE ปัดเป็นบาท, MAPE 4 ทศนิยมของสัดส่วน)'
    parts.append(f'<section id="results"><h2>3. ผลหลักทั้ง {len(runs)} งาน</h2><p class="muted">{intro}</p>')
    main_rows = [[r['model'], r['dataset'], r['variant'], gen_label(r), fmt(metric(r, 'R2'), 4), fmt(metric(r, 'RMSE')),
                  fmt(metric(r)), pct(metric(r, 'MAPE'), True), cv_text(r), cv_extra_text(r, 'rmse'), cv_extra_text(r, 'mape')] for r in runs]
    parts.append(table(['โมเดล', 'ชุดข้อมูล', 'แบบ', 'รอบ', 'Test R²', 'RMSE (บาท)', 'MAE (บาท)', 'MAPE', 'CV MAE ± SD (บาท)', 'CV RMSE ± SD (บาท)', 'CV MAPE ± SD'], main_rows,
                       [i for i, r in enumerate(runs) if winners[r['dataset']] is r]))
    if any(is_new(r) and r['model'] == 'XGB' for r in runs):
        parts.append('<p class="note">ข้อควรระวังของ XGB รอบใหม่: จำนวนรอบเลือกด้วย early stopping บนแถวตรวจของแต่ละ fold ดังนั้น CV MAE ของ XGB อาจต่ำกว่าความจริงเล็กน้อย (แถวตรวจเป็นตัวกำหนดจุดหยุด) ส่วน test MAE ไม่ได้ใช้ตัดสินจุดหยุด</p>')
    parts.append('<p><a href="results.json">ดาวน์โหลดตัวเลขทั้งหมด (JSON)</a> รวม train / validation / test, best params, CV และ importance</p></section>')
    parts.append('<section id="parameters"><h2>พารามิเตอร์ของแต่ละไฟล์</h2><p>ค่าของโมเดลสุดท้ายจาก RESULT_JSON หรือสรุปผู้ชนะของแต่ละไฟล์' + (' (อ่านจากสรุปกฎ 1-SE)' if old else '') + '; ค่าคงที่อ่านจาก source และค่าที่ไม่ได้ระบุอ่านจาก get_params() ของไลบรารีที่ติดตั้งขณะสร้างรายงาน โดยไม่ฝึกโมเดล</p>')
    if not old:
        parts.append('<p class="note">รอบใหม่: ที่มาของค่าที่ปรับคือ <strong>GridSearchCV, ผู้ชนะ = CV MAE ต่ำสุด (refit=MAE)</strong> ผู้ชนะคือชุดที่ mean CV MAE ต่ำสุด เสมอกันเลือกชุดแรก ไม่มีเกณฑ์จำนวนใบ · DT <code>ccp_alpha</code>: ' + ALPHA_SOURCE + ' (ไม่มีรายการ alpha ตายตัวหรือกฎจำกัดจำนวน alpha) · XGB <code>n_estimators</code>: ' + ES_SOURCE + ' (เพดาน 1,000 รอบ หยุดเมื่อ MAE บนแถวตรวจไม่ดีขึ้น 50 รอบ; XGB รอบใหม่ใช้โค้ดค้นเองที่ทำงานเหมือน GridSearchCV แต่ไม่ได้เรียก GridSearchCV)</p>')
    if old:
        parts.append('<p class="note">ที่มา (รอบเก่า): <strong>GridSearchCV + กฎ 1-SE</strong> คือค้นในช่วงที่ระบุแล้วเลือกชุดที่ใบน้อยที่สุดเมื่อ CV MAE ไม่เกินค่าต่ำสุด + SD/√จำนวน fold (SD แบบ ddof=0); เสมอกันเลือก CV MAE ต่ำกว่าแล้ว index ต่ำกว่า · DT นับใบบน train ทั้งชุด ส่วน RF/XGB ใช้ค่าเฉลี่ยใบรวมใน CV; กฎนี้เป็น heuristic ไม่ใช่หลักฐานว่าผลเท่ากันทางสถิติ</p>')
    if old:
        parts.append('<p><strong>เลือกจาก ccp path</strong> (DT รอบเก่า): สร้างตัวเลือก alpha จาก train แล้วคัดเพื่อนบ้านด้วย CV ก่อนเข้า grid; ช่วง alpha อ่านจากผลพิมพ์ 4 หลักนัยสำคัญ แต่ค่าผู้ชนะเก็บความละเอียดจาก JSON · <strong>กราฟ OOB / MAE รายรอบ</strong>: ใช้เป็นที่มาเฉพาะเมื่อโค้ดระบุว่าตรึงจำนวนต้นจากกราฟ (RF ระบุ OOB; XGB ค้นจำนวนรอบใน grid และกราฟรายรอบใช้ตรวจภายหลัง) · <strong>กำหนดคงที่ในโค้ด</strong>: ส่งค่าเอง แม้เท่ากับ default · <strong>ค่าเริ่มต้นของไลบรารี</strong>: ไม่ส่งค่าเอง ระบุเวอร์ชันที่ติดตั้งขณะสร้างรายงาน ซึ่งไม่ใช่การยืนยันเวอร์ชันขณะฝึก; XGB None หมายถึงไม่ได้ระบุใน wrapper และ NaN คือค่าข้อมูลสูญหาย</p>')
    else:
        parts.append('<p><strong>กราฟ OOB</strong>: ใช้เป็นที่มาเฉพาะเมื่อโค้ดระบุว่าตรึงจำนวนต้นจากกราฟ · <strong>กำหนดคงที่ในโค้ด</strong>: ส่งค่าเอง แม้เท่ากับ default · <strong>ค่าเริ่มต้นของไลบรารี</strong>: ไม่ส่งค่าเอง ระบุเวอร์ชันที่ติดตั้งขณะสร้างรายงาน ซึ่งไม่ใช่การยืนยันเวอร์ชันขณะฝึก; XGB None หมายถึงไม่ได้ระบุใน wrapper และ NaN คือค่าข้อมูลสูญหาย</p>')
    parts.append('<p class="muted">เรียงค่าที่ค้นหา → ค่าคงที่ → ค่าเริ่มต้น; ค่าที่ไม่มีหลักฐานแสดงว่า ไม่พบในผลรัน และอยู่ในรายการที่ขาด · ตารางเลื่อนแนวนอนได้บนจอเล็ก</p>')
    for model in MODELS:
        parts.append(f'<h3>{model}</h3>')
        for run in runs:
            if run['model'] != model:
                continue
            parts.append(f'<details id="params-{esc(run["run_id"])}"><summary>{esc(run["notebook"])} · {esc(gen_label(run))}</summary>')
            rows = [[p['parameter'], param_display(p['value']), p['source'], p['explanation']] for p in run['params']]
            parts.append(table(['พารามิเตอร์', 'ค่าที่ใช้', 'ที่มา', 'คำอธิบาย'], rows) if rows else f'<p>{PARAM_MISSING}</p>')
            parts.append('</details>')
    parts.append('</section>')
    if old:
        parts.append('<section id="adjustment"><h2>4. ผลของการปรับข้อมูล</h2><p>adj = scale ตัวแปรตัวเลข + log ราคา แล้วแปลงกลับเป็นบาทเมื่อวัดผล · Δ = adj − base; ค่าติดลบแปลว่าความคลาดเคลื่อนลดลง</p>')
        changes = []
        for dataset in DATASETS:
            for model, b_run, a_run in adjustment_pairs(runs, dataset):
                base, adj = (metric(b_run) if b_run else None), (metric(a_run) if a_run else None)
                ok = comparable(b_run, a_run)
                delta = adj - base if ok and base is not None and adj is not None else None
                changes.append([model, dataset, ' / '.join(gen_label(x) if x else MISSING for x in (b_run, a_run)), fmt(base), fmt(adj),
                                fmt(delta) if ok else 'เทียบไม่ได้ (คนละรอบ)', pct(delta / base * 100) if delta is not None and base else ('เทียบไม่ได้' if not ok else MISSING)])
        parts.append(table(['โมเดล', 'ชุดข้อมูล', 'รอบ base / adj', 'Base MAE', 'Adj MAE', 'Δ บาท', 'Δ %'], changes))
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
    parts.append('</section>' if old else '')
    parts.append('<section id="importance"><h2>5. ความสำคัญของตัวแปร</h2><p>Permutation: MAE บน test ที่เพิ่มขึ้น (บาท), mean ± std จากการสลับ 10 รอบ; ค่าสูงแปลว่าโมเดลพึ่งตัวแปรนั้นมาก · Model-based: สัดส่วน impurity (DT/RF) หรือ gain (XGB) รวม dummy กลับเป็นตัวแปรเดิม แต่ละวิธีใช้คนละสเกลและอาจให้อันดับต่างกัน</p>')
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
        parts.append('</ul>' + cv_fold_table(r) + '</details>')
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
    parts.append('<ul>' + ''.join(f'<li>{esc(item)}</li>' for item in missing) + '</ul>' if missing else f'<p>ไม่มี · พบผลและรูปที่ต้องใช้ครบทั้ง {len(runs)} งาน</p>')
    parts.append('</section><footer>' + ('<a href="index.html">กลับหน้าหลัก (ผลรอบใหม่)</a> · ' if old else '') + 'รายงานสรุปจาก saved outputs · สร้างซ้ำด้วย make_report.py · ไม่มีข้อมูลรถรายแถวในรายงาน</footer></main></body></html>')
    return ''.join(parts), main_rows


def adjustment_pairs(runs, dataset):
    for model in MODELS:
        pair = {r['variant']: r for r in runs if r['dataset'] == dataset and r['model'] == model}
        yield model, pair.get('base'), pair.get('adj')


def comparable(base, adj):
    # Δ adj - base is only meaningful when both runs share data and selection method.
    return bool(base and adj and base['generation'] is not None and base['generation'] == adj['generation'])


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
    # Show current runs before historical runs in every report section.
    runs.sort(key=lambda r: (not is_new(r), r['dataset'], r['variant'], MODELS.index(r['model'])))
    missing = global_missing + [r['run_id'] + ': ' + item for r in runs for item in r['missing']]
    runs_new = [r for r in runs if is_new(r)]
    runs_old = [r for r in runs if not is_new(r)]
    # Each page lists only the missing items of its own runs.
    scoped = lambda group: global_missing + [r['run_id'] + ': ' + item for r in group for item in r['missing']]
    page, main_rows = render(runs, scoped(runs_new))
    page_old, rows_old = render(runs, scoped(runs_old), old=True)
    winners = winner_map(runs)
    payload = {'schema_version': 4, 'pages': {'main': 'index.html', 'old': 'old.html'},
               'source': 'saved notebook outputs and static source; installed estimator defaults; no notebook execution',
               'cv_precision': 'mean and SD from printed fold summary; MAE/RMSE rounded to integer baht; MAPE stored as a fraction to 4 decimal places and displayed as percent',
               'generations': 'old = 1-SE selection (older data); other value = date of the standard CV-MAE-minimum generation; winners are chosen only within the latest generation of a dataset',
               'winners': {d: (w['run_id'] if w else None) for d, w in winners.items()},
               'runs': runs, 'total_seconds': sum((r['runtime'] or {}).get('seconds') or 0 for r in runs), 'missing': missing}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    best = {d: w for d, w in winner_map(runs_new).items() if w}
    readme = ['# ผลฝึกโมเดลราคารถ v5', '',
              f'ผลรอบล่าสุด {len(runs_new)} งาน (DT / RF / XGB บน merge50minC, ผู้ชนะ = CV MAE ต่ำสุด) จาก output ที่บันทึกไว้', '',
              '| โมเดล / แบบ | ชุดข้อมูล | Test MAE (บาท) | Test R² | CV MAE ± SD (บาท) |', '|---|---|---:|---:|---:|']
    for r in runs_new:
        readme.append(f'| {r["model"]} / {r["variant"]} | {r["dataset"]} | {fmt(metric(r))} | {fmt(metric(r, "R2"), 4)} | {cv_text(r)} |')
    for dataset, winner in best.items():
        readme.append('')
        readme.append(f'ผู้ชนะของ {dataset}: **{winner["model"]} / {winner["variant"]}** (test MAE {fmt(metric(winner))} บาท)')
    readme.extend(['', f'**รายงานฉบับเต็ม:** [docs/index.html](docs/index.html) · [GitHub Pages]({PAGES_URL})', '',
                   f'ผลรอบเก่า {len(runs_old)} งาน (ข้อมูลรุ่นเก่า, กฎ 1-SE, เทียบกับรอบใหม่ไม่ได้): [docs/old.html](docs/old.html)', '',
                   '[ผลรันรอบใหม่และ MAE / RMSE / MAPE ของ 5-fold CV](docs/index.html#latest-results)', '',
                   'ค่าพารามิเตอร์ทั้งหมดและที่มารายไฟล์: [พารามิเตอร์ของแต่ละไฟล์](docs/index.html#parameters)', '',
                   'สร้างซ้ำจากโฟลเดอร์นี้: `python -X utf8 make_report.py` (อ่าน saved outputs/source และ defaults จาก sklearn/xgboost ที่ติดตั้ง ไม่รัน notebook)', '',
                   'เผยแพร่ GitHub Pages จาก branch `main` และโฟลเดอร์ `/docs`', ''])
    readme = '\n'.join(readme)
    for name, content in [('index.html', page), ('old.html', page_old), ('results.json', encoded), ('README.md', readme)]:
        if FORBIDDEN.search(content.replace(PAGES_URL, '')):
            raise ValueError('public output safety check failed: ' + name)
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    (OUT / 'old.html').write_text(page_old, encoding='utf-8')
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
    print('Model | Dataset | Variant | Generation | Test R2 | RMSE | MAE | MAPE | CV MAE +/- SD | CV RMSE +/- SD | CV MAPE +/- SD')
    for row in main_rows + rows_old:
        print(' | '.join(map(str, row)))


if __name__ == '__main__':
    main()
