"""Static contract and template-preservation checks; never executes training."""
from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import re
import sys
import tokenize

import nbformat

from build_v5 import AXES, DATASETS, DATA_DIR, MODELS, ROOT, RUN_ORDER, render, serialized, source


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assignments(text):
    values = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def without_added_imports(text):
    tree = ast.parse(text)
    return [ast.dump(n, include_attributes=False) for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]


def result_dict(text):
    dictionaries = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Dict)]
    require(len(dictionaries) == 1, "RESULT_JSON must contain exactly one dict")
    node = dictionaries[0]
    return {ast.literal_eval(k): ast.dump(v, include_attributes=False) for k, v in zip(node.keys, node.values)}


def check_all():
    require((ROOT / "run_order.txt").read_text(encoding="utf-8").splitlines() == RUN_ORDER, "run_order.txt differs from required order")
    shared_axes = None
    for relative in RUN_ORDER:
        path = ROOT / relative
        require(path.is_file(), f"Missing {relative}")
        model = relative.split("/")[0]
        dataset = "new_cardb" if "new_cardb" in relative else "merge50minC"
        adjusted = relative.endswith("_adj.ipynb")
        prefix = f"{relative}: "
        raw = path.read_text(encoding="utf-8")
        notebook = json.loads(raw)
        nbformat.validate(nbformat.reads(raw, as_version=4))
        template = json.loads((ROOT / f"base/v4_{model}_local.ipynb").read_text(encoding="utf-8"))
        original_count = len(template["cells"])
        require(len(notebook["cells"]) == original_count + (5 if adjusted else 3), prefix + "wrong cell count")
        require(notebook["metadata"] == template["metadata"], prefix + "notebook metadata changed")
        require(notebook["nbformat"] == template["nbformat"] and notebook["nbformat_minor"] == template["nbformat_minor"], prefix + "notebook format changed")
        for i, cell in enumerate(notebook["cells"], 1):
            label = prefix + f"cell {i}: "
            text = source(cell)
            require(cell["cell_type"] == "code", label + "not a code cell")
            require(cell.get("outputs") == [] and cell.get("execution_count") is None, label + "outputs/execution_count present")
            require(re.match(rf"^# \[{i:02d}\] .+", text), label + "non-contiguous first-line number")
            require(f"=== [{i:02d}] " in text, label + "missing numbered header")
            ast.parse(text, filename=relative + f":cell{i}")
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT:
                    for ref in re.findall(r"\[(\d{2})\]", token.string):
                        require(1 <= int(ref) <= len(notebook["cells"]), label + f"broken cross-reference [{ref}]")
            if i > original_count:
                require(all(word in text for word in ("ทำอะไร:", "ทำไม:", "ผลที่ควรเห็น:", "ข้อควรระวัง:")), label + "new cell missing Thai style block")
            if i <= original_count:
                require(cell["metadata"] == template["cells"][i-1]["metadata"], label + "cell metadata changed")
        constants = assignments(source(notebook["cells"][2]))
        require(constants.get("VERSION") == "v5", prefix + "VERSION is not v5")
        require(constants.get("RUN_ID") == path.stem, prefix + "RUN_ID mismatch")
        require(constants.get("VARIANT") == ("adj" if adjusted else "base"), prefix + "VARIANT mismatch")
        filename, sha = DATASETS[dataset]
        require(constants.get("DATASET") == filename and constants.get("DATA_SHA256") == sha, prefix + "dataset/sha mismatch")
        require(assignments(source(notebook["cells"][1])).get("DATA_PATH") == DATA_DIR + filename, prefix + "DATA_PATH mismatch")
        axes = {name: constants.get(name) for name in AXES}
        require(axes == AXES, prefix + "locked axis tuples differ from build definitions")
        if shared_axes is not None:
            require(axes == shared_axes, prefix + "axes differ between notebooks")
        shared_axes = axes
        if dataset == "new_cardb":
            require(constants.get("MAX_CAR_AGE") == 25, prefix + "MAX_CAR_AGE mismatch")
            require(constants.get("MODEL_ALIASES") == {"ALTIS": "COROLLAALTIS", "NP300NAVARA": "NAVARA"}, prefix + "MODEL_ALIASES mismatch")
            # Keep impossible-value checks byte-identical, including all four rules.
            require(source(notebook["cells"][6]) == source(template["cells"][6]), prefix + "impossible-value checks changed")
        result_index = next(i for i,c in enumerate(template["cells"]) if 'print("RESULT_JSON="' in source(c))
        original_result = result_dict(source(template["cells"][result_index]))
        generated_result = result_dict(source(notebook["cells"][result_index]))
        require(set(generated_result) == set(original_result) | {"run_id", "dataset", "variant"}, prefix + "RESULT_JSON keys changed")
        require(all(generated_result[k] == v for k,v in original_result.items()), prefix + "original RESULT_JSON values changed")
        if not adjusted:
            allowed = {0, 2, result_index}
            if dataset == "new_cardb":
                allowed |= set(range(1, 13))
            for i, original in enumerate(template["cells"]):
                if i not in allowed:
                    require(notebook["cells"][i]["source"] == original["source"], prefix + f"unallowed source edit in cell {i+1}")
            require(source(notebook["cells"][0]).startswith(source(template["cells"][0])), prefix + "introduction modified beyond appended imports")
            require(without_added_imports(source(notebook["cells"][0])) == without_added_imports(source(template["cells"][0])), prefix + "cell 01 changed beyond imports")
            if dataset == "new_cardb":
                for i in range(4,10):
                    require(ast.dump(ast.parse(source(notebook["cells"][i]))) == ast.dump(ast.parse(source(template["cells"][i]))), prefix + f"quality-check logic changed in cell {i+1}")
        else:
            all_text = "\n".join(source(c) for c in notebook["cells"])
            require('("model", TransformedTargetRegressor(regressor=make_model(), func=np.log, inverse_func=np.exp))' in all_text, prefix + "target wrapper missing")
            for match in re.finditer(r"model__(?!regressor__)", all_text):
                raise AssertionError(prefix + "unwrapped model parameter prefix remains")
            require("scale_numeric=True" in all_text and 'self.numeric_std_ = numeric.std(ddof=0).replace(0, 1.0)' in all_text, prefix + "fold-local scaling missing")
            require("fixed_model_params" in all_text and "ablation_fold_mae" in all_text, prefix + "ablation missing")
        # Exact deterministic build contract covers all other in-place edits and appended cells.
        require(raw == serialized(render(model, dataset, adjusted)), prefix + "content differs from deterministic generator; rebuild")
    return True


if __name__ == "__main__":
    try:
        check_all()
    except Exception as exc:
        print(f"check_v5 FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    print("check_v5 ok")
