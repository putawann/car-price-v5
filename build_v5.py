"""Deterministically derive twelve output-free v5 notebooks from reviewed templates.

Only explicitly permitted cells are patched. Sources of untouched cells are kept
as their original JSON string lists, including their line endings.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MODELS = ("DT", "RF", "XGB")
DATASETS = {
    "merge50minC": ("merge_50min_clean.csv", "191adb789285fe531bf82521e7ac0892020cbf0bf4325d22f6a657fabaab8aa1"),
    "new_cardb": ("new_car_dataset.csv", "04574b158eecae7a353528c575f19a14c542bdd379368fc756829446cf68d412"),
}
DATA_DIR = "D:/AI_Workspace/car-project/data/main/"
# One definition shared by every generated notebook; original template tuples.
AXES = {
    "MAE_AXIS": (20_000, 90_000), "R2_AXIS": (0.80, 1.00),
    "ROWS_AXIS": (0, 40_000), "FOLD_AXIS": (-0.5, 4.5),
    "MAE_BAR_AXIS": (0, 70_000), "MAE_CCP_AXIS": (50_000, 130_000),
    "MAE_OOB_AXIS": (52_000, 54_000), "MAE_ROUND_AXIS": (30_000, 70_000),
    "PERM_AXIS": (0, 150_000), "SHARE_AXIS": (0, 1.0),
}
RUN_ORDER = [f"{m}/{m}_{d}{suffix}.ipynb"
             for d in ("new_cardb", "merge50minC")
             for suffix in ("", "_adj") for m in MODELS]


def source(cell):
    return "".join(cell["source"])


def replace_once(text, old, new):
    assert text.count(old) == 1, f"Expected one occurrence of {old!r}, got {text.count(old)}"
    return text.replace(old, new, 1)


def new_source(number, title, body, purpose, caution):
    return (f"# [{number:02d}] {title}\n"
            "# ==================================================\n"
            f"# ทำอะไร: {purpose}\n"
            "# ทำไม: ใช้ข้อมูลและกติกาเดียวกันเพื่อเปรียบเทียบการทดลองได้อย่างเป็นธรรม\n"
            "# ผลที่ควรเห็น: ตารางผลที่คำนวณจริงและกราฟที่มีข้อความภาษาอังกฤษ\n"
            f"# ข้อควรระวัง: {caution}\n"
            "# ==================================================\n"
            f'print("=== [{number:02d}] {title} ===")\n\n' + body.strip() + "\n")


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def append_cell(notebook, title, body, purpose, caution):
    number = len(notebook["cells"]) + 1
    notebook["cells"].append({"cell_type": "code", "id": f"v5-cell-{number:02d}", "execution_count": None,
                              "metadata": {}, "outputs": [],
                              "source": new_source(number, title, body, purpose, caution).splitlines(True)})


READ_NEW = '''
# ตรวจอายุที่มาจากไฟล์ก่อนตัดคอลัมน์ซ้ำและปรับ schema ให้เหมือนข้อมูลอีกชุด
assert (raw_df["car_age"] == CURRENT_YEAR - raw_df["year"]).all(), "car_age ไม่ตรงกับปีรถ"
raw_df = raw_df.rename(columns={"car_ type": "car_type", "engine_capacity": "engine",
                                "fuel_type": "fuel", "gear_type": "gear"})
raw_df = raw_df.drop(columns=["new_engine", "car_age"])
assert set(raw_df.columns) == {"brand", "model", "year", "price", "mileage",
                               "engine", "gear", "fuel", "color", "car_type"}
'''

CLEAN_NEW = '''
before_age = len(df)
df = df.loc[(CURRENT_YEAR - df["year"]) <= MAX_CAR_AGE].copy()
print(f"ขั้นกรองอายุ: ลบ {before_age - len(df):,} แถว เหลือ {len(df):,} แถว")
before_normalize = len(df)
original_models = df["model"].copy()
model_key_clean = df["model"].str.upper().str.replace("[^A-Z0-9]", "", regex=True).replace(MODEL_ALIASES)
best_name = df.groupby(model_key_clean)["model"].agg(lambda names: names.value_counts().index[0])
df["model"] = model_key_clean.map(best_name)
print(f"ขั้นรวมชื่อรุ่น: ลบ {before_normalize - len(df):,} แถว เปลี่ยนชื่อ {(original_models != df['model']).sum():,} แถว")
print(f"หลังรวมชื่อรุ่น: {df.groupby(['brand', 'model']).ngroups:,} คู่ brand|model")
'''

SPLIT_NEW = '''
model_key = df["brand"] + "|" + df["model"]
forced = df.groupby(["brand", "model"]).sample(n=1, random_state=SEED)
rest = df.drop(forced.index)
n_train_left = round(TRAIN_SIZE * len(df)) - len(forced)
train_rest, temp = train_test_split(rest, train_size=n_train_left, random_state=SEED)
train = pd.concat([forced, train_rest])
val, test = train_test_split(temp, test_size=0.5, random_state=SEED)
'''

ABLATION = '''
# พารามิเตอร์ถูกปรับสำหรับ both และตรึงไว้ ไม่ปรับใหม่แยกแต่ละ variant
fixed_model_params = {key.removeprefix("model__regressor__"): value
                      for key, value in search.best_params_.items()}
ablation_rows = []
ablation_fold_mae = {}
for label, scale, log_target in [("none", False, False), ("scale only", True, False),
                                  ("log only", False, True), ("both", True, True)]:
    regressor = make_model().set_params(**fixed_model_params)
    if log_target:
        regressor = TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)
    estimator = Pipeline([("encode", FoldOneHot(NUM_COLS, CAT_COLS, scale_numeric=scale)),
                          ("model", regressor)])
    fold_mae = -cross_val_score(estimator, train[features], train["price"], cv=folds,
                                scoring="neg_mean_absolute_error", n_jobs=SEARCH_JOBS,
                                pre_dispatch=SEARCH_JOBS, error_score="raise")
    ablation_fold_mae[label] = fold_mae
    estimator.fit(train[features], train["price"])
    metrics = score(test["price"], estimator.predict(test[features]))
    ablation_rows.append({"variant": label, "CV MAE mean": fold_mae.mean(),
                          "CV MAE SD": fold_mae.std(ddof=0), "test MAE": metrics["MAE"],
                          "test RMSE": metrics["RMSE"], "test R2": metrics["R2"],
                          "test MAPE": metrics["MAPE"]})
ablation_table = pd.DataFrame(ablation_rows)
for metric in ["test MAE", "CV MAE mean"]:
    reference = ablation_table.loc[0, metric]
    ablation_table[f"change {metric} (baht)"] = ablation_table[metric] - reference
    ablation_table[f"change {metric} (%)"] = (ablation_table[metric] - reference) / reference * 100
print("พารามิเตอร์ปรับสำหรับ both เท่านั้น ไม่ได้ปรับใหม่ราย variant; ค่าเงินทั้งหมดเป็นบาท")
print(ablation_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
display(ablation_table)
none_se = ablation_table.loc[0, "CV MAE SD"] / np.sqrt(len(folds))
for _, row in ablation_table.iloc[1:].iterrows():
    delta = row["change CV MAE mean (baht)"]
    verdict = "ต่างจากไม่ปรับไม่เกินหนึ่ง CV SE" if abs(delta) <= none_se else ("ช่วยลดความผิดพลาด" if delta < 0 else "ทำให้ความผิดพลาดเพิ่มขึ้น")
    print(f"การปรับ {row['variant']} {verdict} เมื่อเทียบกับ none ตาม CV MAE")
'''

ABLATION_PLOT = '''
positions = np.arange(len(folds))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5))
for index, (label, values) in enumerate(ablation_fold_mae.items()):
    ax.bar(positions + (index - 1.5) * width, values, width, label=label)
ax.set_xticks(positions, [f"Fold {i + 1}" for i in positions])
ax.set(xlabel="CV fold", ylabel="MAE (baht)", title="Adjustment ablation (fixed parameters tuned for both)")
ax.set_ylim(*MAE_AXIS)
ax.legend()
plt.tight_layout()
plt.show()
plotted = np.concatenate(list(ablation_fold_mae.values()))
if plotted.min() < MAE_AXIS[0] or plotted.max() > MAE_AXIS[1]:
    print("WARN: ค่าบางจุดอยู่นอกแกนที่ล็อกไว้ (ablation) ให้ขยาย MAE_AXIS ในเซลล์ [03]")
'''

PERMUTATION = '''
importance_estimator = search.best_estimator_
X_test = test[features]
y_test = test["price"]
assert len(features) == 8 and list(X_test.columns) == features
permutation_result = permutation_importance(
    importance_estimator, X_test, y_test, scoring="neg_mean_absolute_error",
    n_repeats=10, random_state=SEED,
)
permutation_table = pd.DataFrame({"variable": features, "mean": permutation_result.importances_mean,
                                  "std": permutation_result.importances_std})
# ใช้ผลรวมแบบมีเครื่องหมาย ค่าลบหมายถึงสลับแล้วดีขึ้น; ถ้าผลรวมเป็นศูนย์ให้ share เป็นศูนย์
total_increase = permutation_table["mean"].sum()
permutation_table["share (%)"] = (100 * permutation_table["mean"] / total_increase
                                   if total_increase != 0 else 0.0)
permutation_table = permutation_table.sort_values("mean", ascending=False, kind="stable").reset_index(drop=True)
print("Permutation importance: increase in test MAE (baht)")
print(permutation_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
display(permutation_table)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(permutation_table["variable"], permutation_table["mean"], xerr=permutation_table["std"], capsize=3)
ax.invert_yaxis()
ax.set(xlabel="Increase in test MAE (baht)", ylabel="Variable", title="Test permutation importance (mean ± SD)")
ax.set_xlim(*PERM_AXIS)
plt.tight_layout()
plt.show()
if ((permutation_table["mean"] - permutation_table["std"]).min() < PERM_AXIS[0]
        or (permutation_table["mean"] + permutation_table["std"]).max() > PERM_AXIS[1]):
    print("WARN: ค่าบางจุดอยู่นอกแกนที่ล็อกไว้ (permutation importance) ให้ขยาย PERM_AXIS ในเซลล์ [03]")
'''

MODEL_IMPORTANCE = '''
fitted_model = importance_estimator.named_steps["model"]
if isinstance(fitted_model, TransformedTargetRegressor):
    fitted_model = fitted_model.regressor_
encoded_names = importance_estimator.named_steps["encode"].get_feature_names_out()
if MODEL_NAME == "XGB":
    # ขอ gain อย่างชัดเจนและจับคู่ชื่อตามลำดับจริงของ encoder (รองรับทั้งชื่อจริงและ f0, f1)
    booster = fitted_model.get_booster()
    gains = booster.get_score(importance_type="gain")
    booster_names = booster.feature_names
    if booster_names is None:
        booster_names = [f"f{i}" for i in range(len(encoded_names))]
    else:
        assert booster_names == list(encoded_names), "Booster feature order differs from encoder output"
    assert len(booster_names) == len(encoded_names)
    assert set(gains).issubset(booster_names), "Unknown booster feature name"
    weights = np.asarray([gains.get(name, 0.0) for name in booster_names], dtype=float)
else:
    weights = np.asarray(fitted_model.feature_importances_, dtype=float)
assert len(weights) == len(encoded_names)
grouped_importance = {name: 0.0 for name in features}
for encoded_name, weight in zip(encoded_names, weights):
    original_name = str(encoded_name).split("=", 1)[0]
    assert original_name in grouped_importance, f"Unknown encoded variable: {encoded_name}"
    grouped_importance[original_name] += float(weight)
total_weight = sum(grouped_importance.values())
assert total_weight > 0, "โมเดลไม่มีความสำคัญที่เป็นบวกให้ normalize"
grouped_importance = {name: weight / total_weight for name, weight in grouped_importance.items()}
assert np.isclose(sum(grouped_importance.values()), 1.0)
model_importance_table = pd.DataFrame({"variable": features,
    "model_based": [grouped_importance[name] for name in features]}).sort_values("model_based", ascending=False, kind="stable")
print("Model-based importance: gain (XGB) or mean decrease of impurity (DT/RF)")
print(model_importance_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
display(model_importance_table)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(model_importance_table["variable"], model_importance_table["model_based"])
ax.invert_yaxis()
ax.set(xlabel="Normalized importance (share)", ylabel="Variable", title="Grouped model-based variable importance")
ax.set_xlim(*SHARE_AXIS)
plt.tight_layout()
plt.show()
if (model_importance_table["model_based"].min() < SHARE_AXIS[0]
        or model_importance_table["model_based"].max() > SHARE_AXIS[1]):
    print("WARN: ค่าบางจุดอยู่นอกแกนที่ล็อกไว้ (model importance) ให้ขยาย SHARE_AXIS ในเซลล์ [03]")
'''

IMPORTANCE_COMPARISON = '''
# impurity/gain เอนเอียงไปหาตัวแปรที่มีหลายระดับ เช่น model ซึ่งมี dummy หลายร้อยตัว และคำนวณจากแถวฝึก
# permutation วัดว่า error เพิ่มบนแถวที่ไม่เคยเห็นเท่าใด เมื่อสองวิธีขัดกันควรเชื่อ permutation มากกว่า
importance_comparison = permutation_table.merge(model_importance_table, on="variable", validate="one_to_one")
importance_comparison["permutation rank"] = importance_comparison["mean"].rank(method="min", ascending=False).astype(int)
importance_comparison["model rank"] = importance_comparison["model_based"].rank(method="min", ascending=False).astype(int)
importance_comparison = importance_comparison[["variable", "mean", "std", "share (%)", "permutation rank", "model_based", "model rank"]]
print(importance_comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
display(importance_comparison)
print("IMPORTANCE_JSON=" + json.dumps({
    "run_id": RUN_ID,
    "permutation": {row["variable"]: {"mean": float(row["mean"]), "std": float(row["std"])}
                    for _, row in permutation_table.iterrows()},
    "model_based": grouped_importance,
}, allow_nan=False))
'''


def patch_new_dataset(cells):
    cells[1] = cells[1].replace("merge_50min_clean.csv", "new_car_dataset.csv")
    cells[3] = replace_once(cells[3], 'raw_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")',
                            'raw_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")\n' + READ_NEW)
    for i in range(4, 11):
        cells[i] = cells[i].replace("merge_50min_clean.csv", "new_car_dataset.csv")
    cells[10] = new_source(11, "ตรวจคุณภาพ 7: จำนวนรถต่อรุ่น", '''
rows_per_model = raw_df.groupby(["brand", "model"]).size()
print(f"จำนวนรุ่น (brand|model) : {len(rows_per_model):,}")
print(f"แถวต่อรุ่น ต่ำสุด       : {rows_per_model.min():,.0f}")
print(f"แถวต่อรุ่น มัธยฐาน      : {rows_per_model.median():,.0f}")
print(f"แถวต่อรุ่น สูงสุด       : {rows_per_model.max():,.0f}")
print(f"คู่ brand|model ที่มีแถวน้อยกว่า 10: {(rows_per_model < 10).sum():,}")
print(f"คู่ brand|model ที่มีแถวน้อยกว่า 50: {(rows_per_model < 50).sum():,}")
model_counts_only = raw_df.groupby("model").size()
print(f"ชื่อ model ที่มีแถวน้อยกว่า 10: {(model_counts_only < 10).sum():,}")
print(f"ชื่อ model ที่มีแถวน้อยกว่า 50: {(model_counts_only < 50).sum():,}")
smallest_models = rows_per_model.sort_values().head(10).rename("rows").reset_index()
display(smallest_models.style.hide(axis="index").format({"rows": "{:,.0f}"}))
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(rows_per_model, bins=50)
for threshold_rows in [10, 50]:
    ax.axvline(threshold_rows, linestyle="--", label=f"{threshold_rows} rows")
ax.set(title="Number of rows per brand|model", xlabel="Rows per brand|model", ylabel="Number of models")
ax.legend()
plt.tight_layout()
plt.show()
''', "รายงานจำนวนแถวต่อรุ่นและจำนวนรุ่นที่ต่ำกว่า 10 และ 50 แถว", "ข้อมูลชุดนี้ไม่มีข้อกำหนดจำนวนแถวขั้นต่ำต่อรุ่น ไม่ลบแถวในเซลล์ตรวจคุณภาพ")
    cells[11] = replace_once(cells[11], 'df = raw_df.copy()\n', 'df = raw_df.copy()\n' + CLEAN_NEW)
    cells[11] = cells[11].replace('#         (1) สร้างคอลัมน์', '#         ใช้กฎ v3 กรองอายุและรวมชื่อรุ่นก่อน โดยเลือกชื่อเดิมที่พบบ่อยที่สุดในแต่ละกลุ่ม\n#         (1) สร้างคอลัมน์')
    split_tail = cells[12][cells[12].index("split_table ="):]
    cells[12] = new_source(13, "แบ่ง train / validation / test", SPLIT_NEW + '\n' + split_tail,
                          "บังคับหนึ่งแถวต่อ brand|model เข้า train แล้วสุ่มเติมโควตา train และแบ่ง val/test ครึ่งต่อครึ่ง",
                          "มี 242 ชื่อรุ่นที่มีน้อยกว่า 10 แถว จึงแบ่งแบบ stratified ไม่ได้; ใช้กฎ v3 ด้วย SEED เดิม")


def patch_adjusted(cells, model):
    for i, cell in enumerate(cells):
        cell = cell.replace("model__", "model__regressor__")
        cell = cell.replace('FoldOneHot(NUM_COLS, CAT_COLS)', 'FoldOneHot(NUM_COLS, CAT_COLS, scale_numeric=True)')
        cell = cell.replace('("model", make_model())', '("model", TransformedTargetRegressor(regressor=make_model(), func=np.log, inverse_func=np.exp))')
        for attribute in ["get_n_leaves", "estimators_", "get_booster"]:
            cell = cell.replace(f'named_steps["model"].{attribute}', f'named_steps["model"].regressor_.{attribute}')
        cells[i] = cell
    cells[11] += '\nassert (df["price"] > 0).all(), "log(price) ต้องมีราคาเป็นบวกทุกแถว"\n'
    encoder = cells[18]
    encoder = replace_once(encoder, 'def __init__(self, numeric_columns, categorical_columns):',
                            'def __init__(self, numeric_columns, categorical_columns, scale_numeric=False):')
    encoder = replace_once(encoder, 'self.categorical_columns = categorical_columns',
                            'self.categorical_columns = categorical_columns\n        self.scale_numeric = scale_numeric')
    encoder = replace_once(encoder, '        self.levels_ = {}', '''        # เรียนรู้ mean/std จากแถว fit ของ fold นี้เท่านั้น ไม่แตะแถวตรวจ; dummy ไม่ถูก scale
        numeric = X[self.numeric_].astype(float)
        self.numeric_mean_ = numeric.mean()
        self.numeric_std_ = numeric.std(ddof=0).replace(0, 1.0)
        self.levels_ = {}''')
    encoder = replace_once(encoder, '        parts = [X[self.numeric_].astype(float)]', '''        numeric = X[self.numeric_].astype(float)
        if self.scale_numeric:
            numeric = (numeric - self.numeric_mean_) / self.numeric_std_
        parts = [numeric]''')
    encoder = encoder.replace('# ข้อควรระวัง: ผู้เรียกใช้', '# ข้อควรระวัง: scale_numeric=True เรียนรู้สเกลภายในแต่ละ fold เท่านั้น\n#              ผู้เรียกใช้')
    cells[18] = encoder
    if model == "DT":
        cells[21] = replace_once(cells[21], 'train_prices = train["price"].to_numpy()  # ราคาเป็นบาท ไม่แปลง log',
                                 'train_prices = train["price"].to_numpy()  # เก็บบาทสำหรับ CV และ pipeline\npruning_prices = np.log(train_prices)  # pruning path ใช้เป้าหมาย log เช่นเดียวกับต้นไม้ใน pipeline')
        cells[21] = replace_once(cells[21], '    train_prices,\n', '    pruning_prices,\n')
        cells[21] = cells[21].replace('# ทำไม: ผลราคารถเป็นหน่วยบาท ขนาดของ alpha ที่เหมาะจึงขึ้นกับข้อมูลและหน่วยของราคา',
                                     '# ทำไม: ต้นไม้ฝึก log(price) จึงหา alpha จาก log(price); MAE ที่ให้คะแนนยังเป็นบาท\n#         alpha อยู่บนสเกล log-target และแกน x แบบ log เลือกขอบจากข้อมูลเอง')
        cells[22] = replace_once(cells[22], 'tree.fit(encoded_train, train_prices)', 'tree.fit(encoded_train, pruning_prices)')
        cells[24] = replace_once(cells[24], '("model", DecisionTreeRegressor(', '("model", TransformedTargetRegressor(regressor=DecisionTreeRegressor(')
        cells[24] = replace_once(cells[24], '        )),', '        ), func=np.log, inverse_func=np.exp)),')
    elif model == "RF":
        cells[37] = replace_once(cells[37], 'forest = make_model().set_params(**best_params, warm_start=True, oob_score=True)',
                                 'forest = clone(search.best_estimator_.named_steps["model"].regressor_).set_params(warm_start=True, oob_score=True)')
        cells[37] = replace_once(cells[37], 'forest.fit(encoded_train, train_prices)', 'forest.fit(encoded_train, np.log(train_prices))')
        cells[37] = replace_once(cells[37], 'train_prices - forest.oob_prediction_', 'train_prices - np.exp(forest.oob_prediction_)')
        cells[37] = cells[37].replace('# ทำอะไร:', '# ทำอะไร: ฝึก log(price) และ exp คำทาย OOB ก่อนคำนวณ MAE เป็นบาท\n#         ' , 1)
    elif model == "XGB":
        cells[37] = replace_once(cells[37], 'round_model = clone(search.best_estimator_.named_steps["model"])',
                                 'round_model = clone(search.best_estimator_.named_steps["model"].regressor_)')
        cells[37] = replace_once(cells[37], 'curve_rounds =', '''# eval_metric รับค่า log จากโมเดล จึง exp ทั้งจริงและทายก่อนวัด MAE เป็นบาท
def baht_mae(actual_log, predicted_log):
    return mean_absolute_error(np.exp(actual_log), np.exp(predicted_log))

curve_rounds =''')
        cells[37] = cells[37].replace('round_model.set_params(n_estimators=curve_rounds)', 'round_model.set_params(n_estimators=curve_rounds, eval_metric=baht_mae)')
        cells[37] = cells[37].replace('        fit_part["price"],', '        np.log(fit_part["price"]),')
        cells[37] = cells[37].replace('(encoded_fit, fit_part["price"])', '(encoded_fit, np.log(fit_part["price"]))').replace('(encoded_check, check_part["price"])', '(encoded_check, np.log(check_part["price"]))')
        cells[37] = cells[37].replace('["mae"]', '["baht_mae"]')


def render(model, dataset, adjusted=False):
    template = json.loads((ROOT / f"base/v4_{model}_local.ipynb").read_text(encoding="utf-8"))
    notebook = copy.deepcopy(template)
    cells = [source(c) for c in template["cells"]]
    filename, sha = DATASETS[dataset]
    run_id = f"{model}_{dataset}" + ("_adj" if adjusted else "")
    # Imports only: leave all reviewed introductory prose byte-identical.
    cells[0] += '\nfrom sklearn.compose import TransformedTargetRegressor\nfrom sklearn.inspection import permutation_importance\nfrom sklearn.model_selection import cross_val_score\n'
    cells[2] = cells[2].replace('VERSION = "v4"', 'VERSION = "v5"')
    cells[2] = re.sub(r'DATA_SHA256 = "[a-f0-9]+"', f'DATA_SHA256 = "{sha}"', cells[2])
    cells[2] += f'\nRUN_ID = "{run_id}"\nDATASET = "{filename}"\nVARIANT = "{"adj" if adjusted else "base"}"\n'
    if dataset == "new_cardb":
        cells[2] += 'MAX_CAR_AGE = 25\nMODEL_ALIASES = {"ALTIS": "COROLLAALTIS", "NP300NAVARA": "NAVARA"}\n'
        patch_new_dataset(cells)
    for name, value in AXES.items():
        assignment = f"{name} = {value!r}"
        if re.search(rf"^{name} =", cells[2], re.M):
            cells[2] = re.sub(rf"^{name} = \([^\n]*?\)", assignment, cells[2], flags=re.M)
        else:
            cells[2] += assignment + "\n"
    result_index = next(i for i, text in enumerate(cells) if 'print("RESULT_JSON="' in text)
    cells[result_index] = replace_once(cells[result_index], '    "model": MODEL_NAME,',
                                      '    "run_id": RUN_ID,\n    "dataset": DATASET,\n    "variant": VARIANT,\n    "model": MODEL_NAME,')
    if adjusted:
        patch_adjusted(cells, model)
    for cell, text, original in zip(notebook["cells"], cells, template["cells"]):
        if text != source(original):
            set_source(cell, text)
        cell["outputs"] = []
        cell["execution_count"] = None
    if adjusted:
        append_cell(notebook, "Ablation: เปรียบเทียบการปรับข้อมูล", ABLATION,
                    "ตรึงพารามิเตอร์ที่ปรับสำหรับ both แล้ววัด none, scale only, log only, both ด้วย folds เดิมและ train ไป test",
                    "ไม่ได้ปรับพารามิเตอร์ใหม่ราย variant; test ใช้รายงานเท่านั้น; ตัดสินด้วยหนึ่ง SE ของ CV ของ none")
        append_cell(notebook, "Ablation: กราฟ CV MAE", ABLATION_PLOT,
                    "วาดกราฟแท่งแบบกลุ่มของ MAE ราย fold ทั้งสี่ variant", "แกน MAE ใช้ MAE_AXIS เดียวกับทุกการทดลอง")
    append_cell(notebook, "ความสำคัญแบบ permutation บน test", PERMUTATION,
                "สลับตัวแปรต้นฉบับทีละคอลัมน์ก่อน encoding สิบรอบแล้ววัด MAE ที่เพิ่มเป็นบาท",
                "ค่าติดลบเกิดได้; share ใช้ผลรวมมีเครื่องหมาย ไม่ตัดค่าลบทิ้ง; ถ้าผลรวมศูนย์ให้ share เป็นศูนย์")
    append_cell(notebook, "ความสำคัญจากโมเดลรวมกลับเป็นตัวแปร", MODEL_IMPORTANCE,
                "รวม impurity หรือ gain ของ dummy กลับเป็นแปดตัวแปรและ normalize ให้รวมเป็นหนึ่ง",
                "ใช้ผู้ชนะที่ fit บน train แล้ว; gain ขอผ่าน booster อย่างชัดเจน; ระดับมากอาจได้เปรียบ")
    append_cell(notebook, "เปรียบเทียบอันดับและพิมพ์ IMPORTANCE_JSON", IMPORTANCE_COMPARISON,
                "แสดงความสำคัญและอันดับทั้งสองวิธีเคียงกันแล้วพิมพ์ JSON หนึ่งบรรทัด",
                "เมื่อผลสองวิธีขัดกันควรเชื่อ permutation บนข้อมูลที่ไม่เคยเห็นมากกว่า")
    for cell in notebook["cells"]:
        ast.parse(source(cell))
    return notebook


def serialized(notebook):
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def main():
    for relative in RUN_ORDER:
        model = relative.split("/")[0]
        dataset = "new_cardb" if "new_cardb" in relative else "merge50minC"
        target = ROOT / relative
        target.parent.mkdir(exist_ok=True)
        target.write_text(serialized(render(model, dataset, relative.endswith("_adj.ipynb"))), encoding="utf-8", newline="\n")
    (ROOT / "run_order.txt").write_text("\n".join(RUN_ORDER) + "\n", encoding="utf-8", newline="\n")
    print("build_v5 ok: 12 notebooks + run_order.txt")


if __name__ == "__main__":
    main()
