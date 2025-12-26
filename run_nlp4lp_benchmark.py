# run_nlp4lp_benchmark.py
# 批量跑 NL4LP 目录：每个子目录=一道题
# 读取方式：
# - description.txt -> question_text
# - solution.json["objective"] -> ground truth


from __future__ import annotations

import os
import json
import csv
import traceback
from typing import Any, Dict, Optional, List, Tuple

from openai import OpenAI

from gm4opt_pipeline import (
    run_gm4opt_pipeline,
    PipelineConfig,
)

# ========== 配置区域 ==========

# 数据集根目录：包含若干子文件夹（每个子文件夹是一道题）
NL4LP_ROOT_DIR = "data/NL4LP"

# 结果根目录
RESULT_DIR = "result_NL4LP"

# IR JSON 输出目录
IR_OUTPUT_DIR = os.path.join(RESULT_DIR, "ir_outputs_NL4LP")

# 结果日志文件（CSV）
RESULT_CSV_PATH = os.path.join(RESULT_DIR, "NL4LP_results.csv")

# Summary 输出
SUMMARY_TXT_PATH = os.path.join(RESULT_DIR, "NL4LP_summary.txt")

# 使用的 LLM 模型名称
LLM_MODEL_NAME = "gpt-4o"

# 子目录内的文件名约定
DESC_FILENAME = "description.txt"
SOL_FILENAME = "solution.json"

# ========== 工具函数 ==========

def safe_float(x: Any) -> Optional[float]:
    """尽量把 x 转成 float，失败则返回 None。"""
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


def _list_problem_dirs(root_dir: str) -> List[Tuple[str, str]]:
    """
    列出 NL4LP 的题目子目录。
    仅保留同时包含 description.txt 和 solution.json 的子目录。
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"NL4LP_ROOT_DIR not found or not a directory: {root_dir}")

    items: List[Tuple[str, str]] = []
    for name in os.listdir(root_dir):
        abs_path = os.path.join(root_dir, name)
        if not os.path.isdir(abs_path):
            continue
        desc_path = os.path.join(abs_path, DESC_FILENAME)
        sol_path = os.path.join(abs_path, SOL_FILENAME)
        if os.path.isfile(desc_path) and os.path.isfile(sol_path):
            items.append((name, abs_path))

    def sort_key(x: Tuple[str, str]):
        name = x[0]
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)

    items.sort(key=sort_key)
    return items


def _read_description(desc_path: str) -> str:
    with open(desc_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _read_ground_truth_objective(sol_path: str) -> Any:
    with open(sol_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("objective", None)


def solve_one_instance(
    idx: int,
    problem_dir_name: str,
    question_text: str,
    gt_objective_raw: Any,
    client: OpenAI,
) -> Dict[str, Any]:
    """
    对单条记录执行：
    NL -> (dynamic system prompt templating) -> IR(JSON) -> ModelIR -> (optional Graph/Difficulty) -> Gurobi
    返回一行日志用的 dict。
    """
    instance_id = f"NL4LP_{problem_dir_name}"
    status_str = "ERROR"
    obj_value: Optional[float] = None
    correct = False
    error_msg = ""

    gt_value = safe_float(gt_objective_raw)

    try:
        cfg = PipelineConfig(
            model_name=LLM_MODEL_NAME,
            temperature=0.0,
            enable_dynamic_prompt=True,
            enable_graph=True,
            enable_difficulty=True,
            solve_with_gurobi=True,
        )

        res = run_gm4opt_pipeline(
            question_text=question_text,
            client=client,
            config=cfg,
        )

        # 覆盖 problem_id，方便对齐输出文件
        if "meta" not in res.ir_dict or not isinstance(res.ir_dict["meta"], dict):
            res.ir_dict["meta"] = {}
        res.ir_dict["meta"]["problem_id"] = instance_id

        # 1) 保存 IR JSON 到本地（路径已改到 RESULT_DIR 下）
        ensure_dir(IR_OUTPUT_DIR)
        safe_problem_id = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in instance_id
        )
        json_path = os.path.join(IR_OUTPUT_DIR, f"{idx:03d}_{safe_problem_id}.json")
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(res.ir_dict, f_json, indent=2, ensure_ascii=False)

        # 2) 读取求解结果
        status_code = res.gurobi_status
        status_str = str(status_code) if status_code is not None else "NONE"
        obj_value = res.gurobi_obj_value

        # 3) 和 ground truth 比较
        if obj_value is not None and gt_value is not None:
            correct = is_close(obj_value, gt_value)
        else:
            correct = False

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return {
        "index": idx,
        "problem_id": instance_id,
        "problem_dir": problem_dir_name,
        "gurobi_status": status_str,
        "obj_value": obj_value if obj_value is not None else "",
        "ground_truth": gt_value if gt_value is not None else "",
        "correct": int(correct),
        "error": error_msg,
    }


# ========== 主函数 ==========

def main():
    # 确保结果目录存在
    ensure_dir(RESULT_DIR)
    ensure_dir(IR_OUTPUT_DIR)

    client = OpenAI()

    problem_dirs = _list_problem_dirs(NL4LP_ROOT_DIR)
    print(f"[INFO] Found {len(problem_dirs)} instances under {NL4LP_ROOT_DIR}")

    total = 0
    solved = 0
    correct_cnt = 0

    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = [
            "index",
            "problem_id",
            "problem_dir",
            "gurobi_status",
            "obj_value",
            "ground_truth",
            "correct",
            "error",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (dir_name, abs_dir) in enumerate(problem_dirs):
            total += 1
            print(f"\n==== Solving instance #{idx} (dir {dir_name}) ====")

            desc_path = os.path.join(abs_dir, DESC_FILENAME)
            sol_path = os.path.join(abs_dir, SOL_FILENAME)

            try:
                question_text = _read_description(desc_path)
            except Exception as e:
                print(f"[ERROR] Failed to read description: {desc_path} -> {e}")
                row = {
                    "index": idx,
                    "problem_id": f"NL4LP_{dir_name}",
                    "problem_dir": dir_name,
                    "gurobi_status": "ERROR",
                    "obj_value": "",
                    "ground_truth": "",
                    "correct": 0,
                    "error": f"ReadDescriptionError: {type(e).__name__}: {e}",
                }
                writer.writerow(row)
                continue

            try:
                gt_objective_raw = _read_ground_truth_objective(sol_path)
            except Exception as e:
                print(f"[ERROR] Failed to read solution: {sol_path} -> {e}")
                row = {
                    "index": idx,
                    "problem_id": f"NL4LP_{dir_name}",
                    "problem_dir": dir_name,
                    "gurobi_status": "ERROR",
                    "obj_value": "",
                    "ground_truth": "",
                    "correct": 0,
                    "error": f"ReadSolutionError: {type(e).__name__}: {e}",
                }
                writer.writerow(row)
                continue

            row = solve_one_instance(
                idx=idx,
                problem_dir_name=dir_name,
                question_text=question_text,
                gt_objective_raw=gt_objective_raw,
                client=client,
            )

            writer.writerow(row)

            if row["obj_value"] != "":
                solved += 1
            if row["correct"]:
                correct_cnt += 1

            print(
                f"[RESULT] status={row['gurobi_status']}, "
                f"obj={row['obj_value']}, gt={row['ground_truth']}, "
                f"correct={row['correct']}, error={row['error']}"
            )

    # ====== SUMMARY（同时写入 txt）======
    summary_lines = []
    summary_lines.append("====== SUMMARY ======")
    summary_lines.append(f"Total instances: {total}")
    summary_lines.append(f"Solved (got ObjVal): {solved}")
    summary_lines.append(f"Correct (ObjVal ≈ objective): {correct_cnt}")
    if total > 0:
        summary_lines.append(
            f"Accuracy over all instances: {correct_cnt}/{total} = {correct_cnt / total:.3f}"
        )
        summary_lines.append(f"Solved ratio: {solved}/{total} = {solved / total:.3f}")

    print("\n" + "\n".join(summary_lines))

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f_sum:
        f_sum.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
