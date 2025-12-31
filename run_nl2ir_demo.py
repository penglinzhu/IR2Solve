# run_nl2ir_demo.py
# 单样例 demo：跑通 GM4OPT 新版 pipeline，并输出对 verifier 纠错最关键的中间结果：
# - ir_dict（抽取 JSON）
# - ModelIR 摘要（sets/params/vars/constraints/obj）
# - GraphIR meta/stats/checks
# - verifier_report（占位）
# - Gurobi 求解结果（status/obj/metrics）
# - estimation（占位）


from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict

from openai import OpenAI

from gm4opt_pipeline import PipelineConfig, run_gm4opt_pipeline

SAVE_ARTIFACTS = True
ARTIFACT_DIR = "demo_artifacts"


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _try_asdict(x: Any) -> Any:
    if is_dataclass(x):
        try:
            return asdict(x)
        except Exception:
            return str(x)
    return x


def main():
    QUESTION_TEXT = """
A factory needs to rent a warehouse to store materials for the next 4 months. The required warehouse area for each month is listed in Table 1-14.\nTable 1-14\n\\begin{tabular}{c|c|c|c|c}\n\\hline Month & 1 & 2 & 3 & 4 \\\\\n\\hline Required Warehouse Area $/ \\mathrm{m}^2$ & 1500 & 1000 & 2000 & 1200 \\\\\n\\hline\n\\end{tabular}\n\nThe longer the rental contract period, the greater the discount on warehouse rental fees. The specific data is listed in Table 1-15.\nTable 1-15\n\\begin{tabular}{c|c|c|c|c}\n\\hline Contract Rental Period $/$ months & 1 & 2 & 3 & 4 \\\\\n\\hline \\begin{tabular}{c} \nRental Fee for Warehouse \\\\\nArea within the Contract Period $/ \\mathrm{m}^2$\n\\end{tabular} & 28 & 45 & 60 & 73 \\\\\n\\hline\n\\end{tabular}\n\nThe warehouse rental contract can be processed at the beginning of each month, and each contract specifies the rental area and period. Therefore, the factory can rent a contract on any month, and each time, they can sign one contract or multiple contracts with different rental areas and rental periods. The overall goal is to minimize the rental fees paid. Try to establish a linear programming mathematical model based on the above requirements.
""".strip()

    client = OpenAI()

    cfg = PipelineConfig(
        model_name="gpt-4o",
        temperature=0.0,
        timelimit_sec=60.0,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ARTIFACT_DIR, f"demo_{run_id}")
    if SAVE_ARTIFACTS:
        ensure_dir(out_dir)

    _print_section("INPUT / CONFIG")
    print("Run ID:", run_id)
    print("PipelineConfig:", _pretty(_try_asdict(cfg)))
    print("\nQuestion (first 600 chars):")
    print((QUESTION_TEXT[:600] + ("..." if len(QUESTION_TEXT) > 600 else "")))

    if SAVE_ARTIFACTS:
        write_text(os.path.join(out_dir, "question.txt"), QUESTION_TEXT)
        write_json(os.path.join(out_dir, "pipeline_config.json"), _try_asdict(cfg))

    # ---- Run pipeline ----
    try:
        res = run_gm4opt_pipeline(
            question_text=QUESTION_TEXT,
            client=client,
            config=cfg,
        )
    except Exception as e:
        _print_section("PIPELINE ERROR")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # ---- Extracted JSON ----
    ir_dict = getattr(res, "ir_dict", None)
    _print_section("EXTRACTED JSON (ir_dict)")
    if isinstance(ir_dict, dict):
        print(_pretty(ir_dict))
        if SAVE_ARTIFACTS:
            write_json(os.path.join(out_dir, "ir_dict.json"), ir_dict)
    else:
        print("No ir_dict found on PipelineResult (unexpected).")

    # ---- ModelIR summary ----
    ir = getattr(res, "ir", None)
    _print_section("MODELIR SUMMARY")
    if ir is None:
        print("No ModelIR found on PipelineResult.")
    else:
        meta = getattr(ir, "meta", None)
        sets_ = getattr(ir, "sets", []) or []
        params_ = getattr(ir, "params", []) or []
        vars_ = getattr(ir, "vars", []) or []
        obj_ = getattr(ir, "objective", None)
        cons_ = getattr(ir, "constraints", []) or []

        print("meta:", _pretty(_try_asdict(meta)))
        print(f"\nCounts: sets={len(sets_)}, params={len(params_)}, vars={len(vars_)}, constraints={len(cons_)}")
        if obj_ is not None:
            print("\nobjective:", _pretty(_try_asdict(obj_)))

        def _head(items: list, n: int = 3) -> list:
            return items[:n] if len(items) > n else items

        print("\nsets(head):", _pretty([_try_asdict(x) for x in _head(sets_, 3)]))
        print("\nparams(head):", _pretty([_try_asdict(x) for x in _head(params_, 3)]))
        print("\nvars(head):", _pretty([_try_asdict(x) for x in _head(vars_, 3)]))
        print("\nconstraints(head):", _pretty([_try_asdict(x) for x in _head(cons_, 5)]))

        if SAVE_ARTIFACTS:
            modelir_dump = {
                "meta": _try_asdict(meta),
                "sets": [_try_asdict(x) for x in sets_],
                "params": [_try_asdict(x) for x in params_],
                "vars": [_try_asdict(x) for x in vars_],
                "objective": _try_asdict(obj_),
                "constraints": [_try_asdict(x) for x in cons_],
            }
            write_json(os.path.join(out_dir, "modelir_dump.json"), modelir_dump)

    # ---- Graph ----
    graph_built = getattr(res, "graph_built", None)
    _print_section("GRAPH")
    print("Graph built:", graph_built)

    graph = getattr(ir, "graph", None) if ir is not None else None
    if graph is None:
        print("Graph is None.")
        gmeta = None
    else:
        gmeta = getattr(graph, "meta", None) or {}
        print("\nGraph meta keys:", list(gmeta.keys()) if isinstance(gmeta, dict) else type(gmeta))
        if isinstance(gmeta, dict):
            stats = gmeta.get("stats", {})
            checks = gmeta.get("checks", {})
            print("\nGraph stats:", _pretty(stats))
            issues = (checks.get("issues", []) if isinstance(checks, dict) else [])
            print("\nGraph issues:")
            if issues:
                for it in issues:
                    print("  -", it)
            else:
                print("  (none)")

    if SAVE_ARTIFACTS:
        write_json(
            os.path.join(out_dir, "graph_meta.json"),
            gmeta if isinstance(gmeta, dict) else {"meta": str(gmeta)},
        )

    # ---- Verifier report (placeholder) ----
    _print_section("VERIFIER")
    verifier_report = getattr(res, "verifier_report", None)
    print(_pretty(verifier_report))
    if SAVE_ARTIFACTS:
        write_json(os.path.join(out_dir, "verifier_report.json"), verifier_report)

    # ---- Gurobi ----
    _print_section("GUROBI")
    print("Build OK:", getattr(res, "solver_build_ok", None))
    build_err = getattr(res, "solver_build_error", "") or ""
    if build_err:
        print("Build error:", build_err)

    print("Status code:", getattr(res, "gurobi_status", None))
    print("Status name:", getattr(res, "gurobi_status_name", None))
    print("Objective value:", getattr(res, "gurobi_obj_value", None))
    print("Metrics:", _pretty(getattr(res, "gurobi_metrics", {})))

    # ---- Estimation (placeholder) ----
    _print_section("ESTIMATION")
    estimation = getattr(res, "estimation", None)
    print(_pretty(estimation))
    if SAVE_ARTIFACTS:
        write_json(os.path.join(out_dir, "estimation.json"), estimation)

    # ---- Minimal summary ----
    _print_section("SUMMARY")
    print("graph_built:", graph_built)
    print("verifier_ok:", (verifier_report or {}).get("ok") if isinstance(verifier_report, dict) else None)
    print("gurobi_status:", getattr(res, "gurobi_status_name", None))
    print("gurobi_obj_value:", getattr(res, "gurobi_obj_value", None))

    if SAVE_ARTIFACTS:
        write_json(
            os.path.join(out_dir, "summary.json"),
            {
                "run_id": run_id,
                "graph_built": graph_built,
                "verifier_report": verifier_report,
                "gurobi_status": getattr(res, "gurobi_status", None),
                "gurobi_status_name": getattr(res, "gurobi_status_name", None),
                "gurobi_obj_value": getattr(res, "gurobi_obj_value", None),
                "gurobi_metrics": getattr(res, "gurobi_metrics", {}),
                "estimation": estimation,
            },
        )
        print(f"\n[Artifacts saved] {out_dir}")


if __name__ == "__main__":
    main()
