"""Execute generated notebook cells sequentially with in-memory work reductions.

No notebook/data file is edited. Full cell stdout (including printed importance
and ablation tables) is retained in smoke_logs/<run_id>.log. Display and show are
no-ops. Expected statistical/grid/axis WARNs are recorded, not hidden; all other
WARN text and all exceptions fail the run. The tiny sample cannot validate full
data accuracy or the production axis ranges.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from build_v5 import ROOT, source


class Capture(io.StringIO):
    def __init__(self):
        super().__init__()
        self.cell = 0
        self.lines = []

    def write(self, text):
        return super().write(text)


def expected_warn(line, title):
    """Narrow, documented diagnostics expected from small samples/reduced grids."""
    if "ค่าบางจุดอยู่นอกแกนที่ล็อกไว้" in line:
        return True  # Original fixed production axes deliberately remain unchanged.
    if "ตรวจขอบ grid" in title:
        return any(s in line for s in ("ตกขอบ grid", "ขอบของ", "อ่านตาราง:", "สรุป:"))
    if "สรุปการตรวจ overfit" in title:
        return (line.startswith("WARN ") or line.startswith("WARN :") or line.startswith("สรุป:")
                or (line.startswith("INFO train MAE") and "ไม่นับเป็น WARN" in line))
    if "OOB: ตาราง" in title:
        return line.startswith("WARN: เพิ่มต้นไม้แล้วยังดีขึ้นเกิน 1 SE")
    return False


def run(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Notebook must be inside this worktree")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Configure backend before notebook imports; use the requested interpreter.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from threadpoolctl import threadpool_limits

    namespace = {"__name__": "v5_smoke_notebook"}
    output = Capture()
    expected = []
    started = time.monotonic()
    failure = None
    tables = {}
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), threadpool_limits(limits=1):
        try:
            for index, cell in enumerate(notebook["cells"], 1):
                text = source(cell)
                title = text.splitlines()[0]
                before = output.tell()
                exec(compile(text, f"{relative}:cell{index}", "exec"), namespace)
                if index == 1:
                    namespace["display"] = lambda *args, **kwargs: None
                    plt.show = lambda *args, **kwargs: None
                if index == 3:
                    namespace.update(LC_FRACTIONS=np.array([0.5, 1.0]), LC_REPEATS=1,
                                     SEARCH_JOBS=2, MODEL_THREADS=1)
                    if namespace["MODEL_NAME"] == "RF":
                        namespace.update(N_TREES=30, OOB_TREE_COUNTS=[30, 60])
                    if namespace["MODEL_NAME"] == "DT":
                        namespace["CCP_N_CANDIDATES"] = 3
                if index == 13:
                    for name, count in [("train", 700), ("val", 200), ("test", 200)]:
                        part = namespace[name]
                        namespace[name] = part.sample(n=min(count, len(part)), random_state=namespace["SEED"])
                    assert set(namespace["train"].index).isdisjoint(namespace["test"].index)
                if index == 21 or ('PARAM_GRID["ccp_alpha"] =' in text):
                    namespace["PARAM_GRID"] = {key: list(values)[:2] for key, values in namespace["PARAM_GRID"].items()}
                cell_output = output.getvalue()[before:]
                for line in cell_output.splitlines():
                    if "WARN" in line:
                        if expected_warn(line, title):
                            expected.append((index, line))
                        else:
                            raise AssertionError(f"Unexpected WARN in cell {index}: {line}")
                plt.close("all")
            # Meaningful consistency checks for the new behavior, in baht.
            assert np.isfinite(namespace["importance_comparison"].select_dtypes(include="number").to_numpy()).all()
            assert set(namespace["grouped_importance"]) == set(namespace["features"])
            assert np.isclose(sum(namespace["grouped_importance"].values()), 1)
            winner = namespace["search"].best_estimator_
            from sklearn.metrics import mean_absolute_error
            observed = mean_absolute_error(namespace["test"]["price"], winner.predict(namespace["test"][namespace["features"]]))
            assert np.isclose(observed, namespace["results"]["test"]["MAE"])
            if namespace["VARIANT"] == "adj":
                ablation = namespace["ablation_table"].set_index("variant")
                assert np.isfinite(ablation.to_numpy()).all()
                assert np.isclose(ablation.loc["both", "test MAE"], observed)
                assert np.isclose(ablation.loc["both", "CV MAE mean"], namespace["cv_mae_all"][namespace["chosen_index"]])
                encoder = winner.named_steps["encode"]
                transformed = encoder.transform(namespace["train"][namespace["features"]])
                assert np.allclose(transformed[namespace["NUM_COLS"]].mean(), 0, atol=1e-10)
                assert np.allclose(transformed[namespace["NUM_COLS"]].std(ddof=0), 1)
                tables["ablation"] = namespace["ablation_table"]
            if namespace["MODEL_NAME"] == "XGB":
                assert np.isclose(namespace["winner_row"]["CV MAE mean"], namespace["cv_mae_all"][namespace["chosen_index"]], rtol=1e-5)
            tables["importance"] = namespace["importance_comparison"]
        except Exception:
            traceback.print_exc()
            failure = sys.exc_info()[1]
    elapsed = time.monotonic() - started
    log_dir = ROOT / "smoke_logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{path.stem}.log").write_text(output.getvalue(), encoding="utf-8")
    if failure is not None:
        print(f"SMOKE FAIL {path.relative_to(ROOT).as_posix()} ({elapsed:.1f}s): {failure}")
        print(output.getvalue()[-6000:])
        return 1
    if elapsed >= 90:
        print(f"SMOKE FAIL {path.relative_to(ROOT).as_posix()} ({elapsed:.1f}s): exceeded 90-second smoke budget")
        return 1
    print(f"SMOKE PASS {path.relative_to(ROOT).as_posix()} ({elapsed:.1f}s; {len(expected)} expected diagnostic WARN lines)")
    # Print the new tables for every run; also retained in each complete cell log.
    for label, table in tables.items():
        print(f"SMOKE TABLE {path.stem} {label}")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    args = parser.parse_args()
    sys.exit(run(args.notebook))
