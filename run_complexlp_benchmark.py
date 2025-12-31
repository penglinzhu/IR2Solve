# run_complexlp_benchmark.py
# 批量跑 Mamo_complex_lp_clean.jsonl：
# NL -> system prompt -> IR(JSON) -> ModelIR -> GraphIR -> verifier -> Gurobi -> estimation
# 记录每个问题的求解结果，并保存 IR JSON

from __future__ import annotations

import os
import json
import csv
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from gm4opt_pipeline import run_gm4opt_pipeline, PipelineConfig

# ========== 配置区域 ==========

DATASET_PATH = "data/Mamo/Mamo_complex_lp_clean.jsonl"

RESULT_DIR = "result_Mamo_complex"
IR_OUTPUT_DIR = os.path.join(RESULT_DIR, "ir_outputs_Mamo_complex")

RESULT_CSV_PATH = os.path.join(RESULT_DIR, "Mamo_complex_results.csv")
SUMMARY_TXT_PATH = os.path.join(RESULT_DIR, "Mamo_complex_summary.txt")

LLM_MODEL_NAME = "gpt-4o"

# ========== 工具函数 ==========


def safe_float(x: Any) -> Optional[float]:
    """尽量把 Answer 转成 float，失败则返回 None。"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def is_close(a: float, b: float, atol: float = 1e-4, rtol: float = 1e-6) -> bool:
    """浮点数比较"""
    return abs(a - b) <= (atol + rtol * max(1.0, abs(b)))


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _try_asdict(x: Any) -> Any:
    if is_dataclass(x):
        try:
            return asdict(x)
        except Exception:
            return str(x)
    return x


def _build_pipeline_config(**kwargs) -> PipelineConfig:
    """
    根据 PipelineConfig 的真实字段自动过滤 kwargs 并构造 config。
    这样 pipeline config 字段调整时，benchmark 脚本不需要频繁联动。
    """
    if not is_dataclass(PipelineConfig):
        return PipelineConfig(**kwargs)
    field_names = set(PipelineConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in kwargs.items() if k in field_names}
    return PipelineConfig(**filtered)


def _stage_ok(stages: Any, name: str) -> int:
    """
    stages 是 pipeline 的 stages dict（或 None）。
    返回 0/1，用于 CSV 汇总。
    """
    try:
        if not isinstance(stages, dict):
            return 0
        st = stages.get(name, None)
        if st is None:
            return 0
        return int(bool(getattr(st, "ok", False)))
    except Exception:
        return 0


def solve_one_instance(
    idx: int,
    question_text: str,
    gt_answer_raw: Any,
    client: OpenAI,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    返回 (row_csv, ir_dict)
    """
    instance_id = f"mamo_complex_{idx}"
    gt_value = safe_float(gt_answer_raw)

    row_csv: Dict[str, Any] = {
        "index": idx,
        "problem_id": instance_id,
        "failure_stage": "",
        "json_extract_ok": 0,
        "ir_parse_ok": 0,
        "graph_build_ok": 0,
        "verifier_ok": 0,
        "solver_build_ok": 0,
        "solver_optimize_ok": 0,
        "estimation_ok": 0,
        "gurobi_status": "NONE",
        "obj_value": "",
        "ground_truth": gt_value if gt_value is not None else "",
        "correct": 0,
        "prompt_tokens": "",
        "completion_tokens": "",
        "total_tokens": "",
        "error": "",
    }
    ir_dict: Dict[str, Any] = {}

    try:
        cfg = _build_pipeline_config(
            model_name=LLM_MODEL_NAME,
            temperature=0.0,
            enable_graph=True,
            timelimit_sec=60.0,
        )

        res = run_gm4opt_pipeline(
            question_text=question_text,
            client=client,
            config=cfg,
            problem_id=instance_id,
            meta_override={"source": "Mamo_complex_lp"},
        )

        ir_dict = getattr(res, "ir_dict", None) or {}

        stages = getattr(res, "stages", None)
        row_csv["failure_stage"] = getattr(res, "failure_stage", "") or ""

        row_csv["json_extract_ok"] = _stage_ok(stages, "json_extract")
        row_csv["ir_parse_ok"] = _stage_ok(stages, "ir_parse")
        row_csv["graph_build_ok"] = _stage_ok(stages, "graph_build")
        row_csv["verifier_ok"] = _stage_ok(stages, "verifier")
        row_csv["solver_build_ok"] = _stage_ok(stages, "solver_build")
        row_csv["solver_optimize_ok"] = _stage_ok(stages, "solver_optimize")
        row_csv["estimation_ok"] = _stage_ok(stages, "estimation")

        # gurobi status / obj
        status_name = getattr(res, "gurobi_status_name", None)
        status_code = getattr(res, "gurobi_status", None)
        if status_name:
            row_csv["gurobi_status"] = status_name
        elif status_code is not None:
            row_csv["gurobi_status"] = str(status_code)

        obj_val = getattr(res, "gurobi_obj_value", None)
        if obj_val is not None:
            row_csv["obj_value"] = float(obj_val)

        # correctness
        correct = False
        if obj_val is not None and gt_value is not None:
            correct = is_close(float(obj_val), float(gt_value))
        row_csv["correct"] = int(correct)

        # tokens（如果 pipeline 仍保留 usage 字段就写；如果移除了就留空）
        u = getattr(res, "llm_usage", None) or {}
        if isinstance(u, dict):
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = u.get(k, None)
                row_csv[k] = v if v is not None else ""

    except Exception as e:
        row_csv["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc(limit=40)

    return row_csv, ir_dict


# ========== 主函数 ==========


def main():
    ensure_dir(RESULT_DIR)
    ensure_dir(IR_OUTPUT_DIR)

    client = OpenAI()

    total = 0
    solved = 0
    correct_cnt = 0

    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = [
            "index",
            "problem_id",
            "failure_stage",
            "json_extract_ok",
            "ir_parse_ok",
            "graph_build_ok",
            "verifier_ok",
            "solver_build_ok",
            "solver_optimize_ok",
            "estimation_ok",
            "gurobi_status",
            "obj_value",
            "ground_truth",
            "correct",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "error",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        with open(DATASET_PATH, "r", encoding="utf-8") as f_in:
            for idx, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue

                total += 1
                print(f"\n==== Solving instance #{idx} (line {idx+1}) ====")

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    row = {
                        "index": idx,
                        "problem_id": f"mamo_complex_{idx}",
                        "failure_stage": "input_json_decode",
                        "json_extract_ok": 0,
                        "ir_parse_ok": 0,
                        "graph_build_ok": 0,
                        "verifier_ok": 0,
                        "solver_build_ok": 0,
                        "solver_optimize_ok": 0,
                        "estimation_ok": 0,
                        "gurobi_status": "ERROR",
                        "obj_value": "",
                        "ground_truth": "",
                        "correct": 0,
                        "prompt_tokens": "",
                        "completion_tokens": "",
                        "total_tokens": "",
                        "error": f"JSONDecodeError: {e}",
                    }
                    writer.writerow(row)
                    continue

                # === 数据集字段：Question / Answer ===
                question_text = record.get("Question", "")
                gt_answer_raw = record.get("Answer", "")

                row, ir_dict = solve_one_instance(
                    idx=idx,
                    question_text=question_text,
                    gt_answer_raw=gt_answer_raw,
                    client=client,
                )

                writer.writerow(row)

                # 保存 IR JSON
                safe_problem_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in row["problem_id"])
                json_path = os.path.join(IR_OUTPUT_DIR, f"{idx:03d}_{safe_problem_id}.json")
                with open(json_path, "w", encoding="utf-8") as f_json:
                    json.dump(ir_dict, f_json, indent=2, ensure_ascii=False)

                if row["obj_value"] != "":
                    solved += 1
                if row["correct"]:
                    correct_cnt += 1

                print(
                    f"[RESULT] status={row['gurobi_status']}, "
                    f"obj={row['obj_value']}, gt={row['ground_truth']}, "
                    f"correct={row['correct']}, failure_stage={row['failure_stage']}, error={row['error']}"
                )

    # ====== SUMMARY（同时写入 txt）======
    summary_lines = []
    summary_lines.append("====== SUMMARY ======")
    summary_lines.append(f"Total instances: {total}")
    summary_lines.append(f"Solved (got ObjVal): {solved}")
    summary_lines.append(f"Correct (ObjVal ≈ Answer): {correct_cnt}")
    if total > 0:
        summary_lines.append(f"Accuracy over all instances: {correct_cnt}/{total} = {correct_cnt / total:.3f}")
        summary_lines.append(f"Solved ratio: {solved}/{total} = {solved / total:.3f}")

    print("\n" + "\n".join(summary_lines))

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f_sum:
        f_sum.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
