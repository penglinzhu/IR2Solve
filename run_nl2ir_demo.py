# run_nl2ir_demo.py
# 用 GM4OPT 统一 pipeline 跑通一条题目（单样例），并打印：raw 输出、graph/difficulty、Gurobi 解


from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict

from openai import OpenAI

from gm4opt_pipeline import PipelineConfig, run_gm4opt_pipeline


def _build_pipeline_config(**kwargs) -> PipelineConfig:
    """
    根据 PipelineConfig 的真实字段，自动筛选 kwargs 并构造 config。
    """
    if not is_dataclass(PipelineConfig):
        return PipelineConfig(**kwargs) 

    field_names = set(PipelineConfig.__dataclass_fields__.keys()) 
    filtered: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in field_names}
    return PipelineConfig(**filtered)  


def main():
    # 可以换成任意题目
    QUESTION_TEXT = """
Imagine you are a dietitian and you have been tasked with creating a meal plan for a bodybuilder. You have six food items to choose from: Steak, Tofu, Chicken, Broccoli, Rice, and Spinach. Each food provides certain amounts of protein, carbohydrates, and calories, and each has its own cost.\n\nHere's the nutritional value and cost of each food:\n\n- Steak: It gives you 14 grams of protein, 23 grams of carbohydrates, and 63 calories for $4.\n- Tofu: It offers 2 grams of protein, 13 grams of carbohydrates, and 162 calories for $6.\n- Chicken: It packs a punch with 17 grams of protein, 13 grams of carbohydrates, and gives you 260 calories for $6.\n- Broccoli: It provides 3 grams of protein, a mere 1 gram of carbohydrates, and 55 calories for $8.\n- Rice: It gives a hearty 15 grams of protein, 23 grams of carbohydrates, and 231 calories for $8.\n- Spinach: It provides 2 grams of protein, 8 grams of carbohydrates, and a huge 297 calories for just $5.\n\nYour goal is to ensure that the bodybuilder gets at least 83 grams of protein, 192 grams of carbohydrates, and 2089 calories from whatever combination of these foods you choose. The challenge is to keep the cost as low as possible while meeting these nutritional targets. \n\nWhat is the minimum cost to meet these nutritional requirements with the available food options?
""".strip()

    client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY

    # 在这里改开关做消融实验
    cfg = _build_pipeline_config(
        model_name="gpt-4o",
        temperature=0.0,

        enable_dynamic_prompt=True,
        enable_graph=True,
        enable_difficulty=True,
        solve_with_gurobi=True,
    )

    res = run_gm4opt_pipeline(
        question_text=QUESTION_TEXT,
        client=client,
        config=cfg,
    )

    # ===== 输出：LLM 原始输出 =====
    raw_text = getattr(res, "raw_llm_text", None)
    if raw_text is not None:
        print("=== Raw LLM output ===")
        print(raw_text)
        print("=== End of raw LLM output ===\n")

    # ===== 输出：Graph（如果启用）=====
    print("Graph built:", getattr(res, "graph_built", None))

    # ===== 输出：Difficulty（如果启用）=====
    diff = getattr(res, "difficulty", None)
    if diff is not None:
        print(f"\nDifficulty: score={diff.score:.3f}, level={diff.level}")
        print("Difficulty features:")
        for k, v in diff.features.items():
            print(f"  {k}: {v}")

    # ===== 输出：Gurobi 求解结果 =====
    status = getattr(res, "gurobi_status", None)
    if status is not None:
        print("\n=== Gurobi ===")
        print("Status code:", status)
        obj_val = getattr(res, "gurobi_obj_value", None)
        if obj_val is not None:
            print("Objective value:", obj_val)
        else:
            print("No optimal objective value obtained.")

    # ===== 输出：Graph meta（如果构建了图）=====
    ir = getattr(res, "ir", None)
    if ir is not None and getattr(ir, "graph", None) is not None:
        gmeta = ir.graph.meta or {}
        print("\n=== Graph meta ===")
        print("Graph stats:", gmeta.get("stats", {}))
        print("Graph checks:")
        for issue in gmeta.get("checks", {}).get("issues", []):
            print("  -", issue)


if __name__ == "__main__":
    main()
