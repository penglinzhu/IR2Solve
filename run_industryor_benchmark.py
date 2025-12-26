# run_industryor_benchmark.py
# 批量跑 IndustryOR.jsonl：NL -> (dynamic system prompt templating) -> IR -> Gurobi，记录每个问题的求解结果


from __future__ import annotations

import os
import json
import csv
import traceback
from typing import Any, Dict, Optional

from openai import OpenAI

from gm4opt_pipeline import (
    run_gm4opt_pipeline,
    PipelineConfig,
)

# ========== 配置区域 ==========

# IndustryOR 数据集路径
INDUSTRYOR_PATH = "data/IndustryOR/IndustryOR.jsonl"

# 结果根目录
RESULT_DIR = "result_industryOR"

# IR JSON 输出目录
IR_OUTPUT_DIR = os.path.join(RESULT_DIR, "ir_outputs_industryOR")

# 结果日志文件（CSV）
RESULT_CSV_PATH = os.path.join(RESULT_DIR, "industryOR_results.csv")

# Summary 输出
SUMMARY_TXT_PATH = os.path.join(RESULT_DIR, "industryOR_summary.txt")

# 使用的 LLM 模型名称
LLM_MODEL_NAME = "gpt-4o"

# ========== 工具函数 ==========

def safe_float(x: Any) -> Optional[float]:
    """尽量把 en_answer 转成 float，失败则返回 None。"""
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


def solve_one_instance(
    idx: int,
    question_text: str,
    gt_answer_raw: Any,
    client: OpenAI,
) -> Dict[str, Any]:
    """
    对单条 IndustryOR 记录执行：
    NL -> (dynamic system prompt templating) -> IR(JSON) -> ModelIR -> (optional Graph/Difficulty) -> Gurobi
    返回一行日志用的 dict。
    """
    instance_id = f"industryOR_{idx}"
    status_str = "ERROR"
    obj_value: Optional[float] = None
    correct = False
    error_msg = ""

    gt_value = safe_float(gt_answer_raw)

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

    client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY

    total = 0
    solved = 0
    correct_cnt = 0

    with open(RESULT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = [
            "index",
            "problem_id",
            "gurobi_status",
            "obj_value",
            "ground_truth",
            "correct",
            "error",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        # 逐行读取 IndustryOR.jsonl（每行一个 JSON）
        with open(INDUSTRYOR_PATH, "r", encoding="utf-8") as f_in:
            for idx, line in enumerate(f_in):
                line = line.strip()

                # jsonl 常见：空行（可跳过）
                if not line:
                    continue

                total += 1
                print(f"\n==== Solving instance #{idx} (line {idx+1}) ====")

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] JSON decode failed at line {idx+1}: {e}")
                    row = {
                        "index": idx,
                        "problem_id": f"industryOR_{idx}",
                        "gurobi_status": "ERROR",
                        "obj_value": "",
                        "ground_truth": "",
                        "correct": 0,
                        "error": f"JSONDecodeError: {e}",
                    }
                    writer.writerow(row)
                    continue

                question_text = record.get("en_question", "")
                gt_answer_raw = record.get("en_answer", "")

                row = solve_one_instance(
                    idx=idx,
                    question_text=question_text,
                    gt_answer_raw=gt_answer_raw,
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
    summary_lines.append(f"Correct (ObjVal ≈ en_answer): {correct_cnt}")
    if total > 0:
        summary_lines.append(
            f"Accuracy over all instances: {correct_cnt}/{total} = {correct_cnt / total:.3f}"
        )
        summary_lines.append(f"Solved ratio: {solved}/{total} = {solved / total:.3f}")

    # 控制台输出保持一致
    print("\n" + "\n".join(summary_lines))

    # 写入到 result_industryOR/industryOR_summary.txt
    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f_sum:
        f_sum.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
