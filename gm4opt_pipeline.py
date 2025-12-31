# gm4opt_pipeline.py
# NL -> prompt -> JSON IR -> ModelIR -> GraphIR -> verifier -> Gurobi -> estimation
# 精简版：trace 不包含 prompts / LLM 原文 / json文本

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from gurobipy import GRB

from gm4opt_ir import ModelIR, ir_to_gurobi
from gm4opt_nl2ir import build_system_prompt, build_user_prompt, extract_json_from_text, json_to_model_ir
from gm4opt_graph import build_graph_from_ir


def _now() -> float:
    return time.perf_counter()


def _short_tb(limit: int = 20) -> str:
    return traceback.format_exc(limit=limit)


@dataclass
class PipelineConfig:
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    timelimit_sec: float = 60.0


@dataclass
class PipelineStageTrace:
    ok: bool
    t_sec: float
    error_type: str = ""
    error_msg: str = ""
    traceback: str = ""  # 仅失败时填充


@dataclass
class PipelineResult:
    # artifacts
    ir_dict: Dict[str, Any]
    ir: Optional[ModelIR]
    graph_built: bool

    # verifier / estimation placeholder outputs
    verifier_report: Dict[str, Any]
    estimation: Dict[str, Any]

    # solver
    solver_build_ok: bool
    solver_build_error: str
    gurobi_status: Optional[int]
    gurobi_status_name: str
    gurobi_obj_value: Optional[float]
    gurobi_metrics: Dict[str, Any]

    # trace
    stages: Dict[str, PipelineStageTrace]
    failure_stage: str
    timings: Dict[str, float]
    trace: Dict[str, Any]


# -----------------------------------------------------------------------------
# Placeholder: Verifier (future work)
# -----------------------------------------------------------------------------
def _run_verifier_placeholder(
    ir: ModelIR,
    graph_required: bool = True,
) -> Tuple[ModelIR, Dict[str, Any]]:
    """
    占位：未来 gm4opt_verifier.py 会在这里做 graph-based 自检与修复（含第二次LLM交互）。
    当前版本：不改动 ir，仅返回空报告结构。
    """
    report = {
        "ok": True,
        "issues": [],
        "repairs": [],
        "notes": "verifier placeholder (no-op)",
        "graph_required": graph_required,
    }
    return ir, report


# -----------------------------------------------------------------------------
# Placeholder: Estimation (future work; formerly Difficulty)
# -----------------------------------------------------------------------------
def _run_estimation_placeholder(ir: ModelIR) -> Dict[str, Any]:
    """
    占位：未来用 graph/solution 产出估计指标（由原 Difficulty 模块演化）。
    当前版本：只返回基本信息。
    """
    return {
        "ok": True,
        "notes": "estimation placeholder (no-op)",
    }


def run_gm4opt_pipeline(
    question_text: str,
    client: Optional[OpenAI] = None,
    config: Optional[PipelineConfig] = None,
    problem_id: Optional[str] = None,
    meta_override: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """
    Pipeline:
      NL -> system prompt -> user prompt -> LLM -> JSON -> ModelIR -> GraphIR
         -> verifier (placeholder) -> Gurobi -> estimation (placeholder)
    """
    if client is None:
        client = OpenAI()
    if config is None:
        config = PipelineConfig()

    stages: Dict[str, PipelineStageTrace] = {}
    failure_stage = ""

    # outputs (fail-safe defaults)
    data: Dict[str, Any] = {}
    ir: Optional[ModelIR] = None
    graph_built = False

    verifier_report: Dict[str, Any] = {"ok": False, "notes": "not run"}
    estimation: Dict[str, Any] = {"ok": False, "notes": "not run"}

    solver_build_ok = False
    solver_build_error = ""
    gurobi_status: Optional[int] = None
    gurobi_status_name = "NONE"
    gurobi_obj_value: Optional[float] = None
    gurobi_metrics: Dict[str, Any] = {}

    # ---- Stage 1: build prompts ----
    t0 = _now()
    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question_text)
        stages["build_prompts"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
    except Exception as e:
        stages["build_prompts"] = PipelineStageTrace(
            ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
        )
        failure_stage = "build_prompts"
        system_prompt = ""
        user_prompt = ""

    # ---- Stage 2: LLM call ----
    raw_llm_text = ""
    if not failure_stage:
        t0 = _now()
        try:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            completion = client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                temperature=config.temperature,
            )
            raw_llm_text = completion.choices[0].message.content or ""
            stages["llm_call"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
        except Exception as e:
            stages["llm_call"] = PipelineStageTrace(
                ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
            )
            failure_stage = "llm_call"

    # ---- Stage 3: JSON extract ----
    if not failure_stage:
        t0 = _now()
        try:
            data = extract_json_from_text(raw_llm_text)
            stages["json_extract"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
        except Exception as e:
            stages["json_extract"] = PipelineStageTrace(
                ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
            )
            failure_stage = "json_extract"
            data = {}

    # ---- Stage 4: meta fill (best-effort) ----
    t0 = _now()
    try:
        if "meta" not in data or not isinstance(data.get("meta"), dict):
            data["meta"] = {}
        if meta_override:
            for k, v in meta_override.items():
                data["meta"][k] = v
        if problem_id:
            data["meta"]["problem_id"] = problem_id
        if not data["meta"].get("problem_id"):
            data["meta"]["problem_id"] = "gm4opt_instance"
        stages["meta_fill"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
    except Exception as e:
        stages["meta_fill"] = PipelineStageTrace(
            ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
        )
        if not failure_stage:
            failure_stage = "meta_fill"

    # ---- Stage 5: JSON -> ModelIR ----
    if not failure_stage:
        t0 = _now()
        try:
            ir = json_to_model_ir(data)
            stages["ir_parse"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
        except Exception as e:
            stages["ir_parse"] = PipelineStageTrace(
                ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
            )
            failure_stage = "ir_parse"
            ir = None

    # ---- Stage 6: GraphIR build ----
    if not failure_stage and ir is not None:
        t0 = _now()
        try:
            ir.graph = build_graph_from_ir(ir)
            graph_built = True
            stages["graph_build"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
        except Exception as e:
            stages["graph_build"] = PipelineStageTrace(
                ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
            )
            failure_stage = "graph_build"
            graph_built = False

    # ---- Stage 7: verifier (placeholder) ----
    if not failure_stage and ir is not None:
        t0 = _now()
        try:
            ir, verifier_report = _run_verifier_placeholder(ir, graph_required=True)
            stages["verifier"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
        except Exception as e:
            stages["verifier"] = PipelineStageTrace(
                ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
            )
            failure_stage = "verifier"

    # ---- Stage 8: Gurobi build + optimize (ALWAYS) ----
    if not failure_stage and ir is not None:
        # build
        t0 = _now()
        model = None
        try:
            model = ir_to_gurobi(ir)
            solver_build_ok = True
            stages["solver_build"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
        except Exception as e:
            solver_build_ok = False
            solver_build_error = f"{type(e).__name__}: {e}"
            stages["solver_build"] = PipelineStageTrace(
                ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
            )
            failure_stage = "solver_build"

        # optimize
        if solver_build_ok and model is not None:
            t0 = _now()
            try:
                model.setParam("TimeLimit", float(config.timelimit_sec))
                model.optimize()

                gurobi_status = model.Status
                status_map = {
                    GRB.OPTIMAL: "OPTIMAL",
                    GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.UNBOUNDED: "UNBOUNDED",
                    GRB.INF_OR_UNBD: "INF_OR_UNBD",
                    GRB.TIME_LIMIT: "TIME_LIMIT",
                    GRB.INTERRUPTED: "INTERRUPTED",
                    GRB.SUBOPTIMAL: "SUBOPTIMAL",
                }
                gurobi_status_name = status_map.get(model.Status, str(model.Status))

                if model.Status == GRB.OPTIMAL:
                    gurobi_obj_value = float(model.ObjVal)

                gurobi_metrics = {
                    "runtime_sec": float(getattr(model, "Runtime", 0.0) or 0.0),
                    "node_count": int(getattr(model, "NodeCount", 0) or 0),
                }
                try:
                    gurobi_metrics["mip_gap"] = float(getattr(model, "MIPGap", 0.0) or 0.0)
                except Exception:
                    pass

                stages["solver_optimize"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
            except Exception as e:
                stages["solver_optimize"] = PipelineStageTrace(
                    ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
                )
                failure_stage = "solver_optimize"

    # ---- Stage 9: estimation (placeholder; after solve) ----
    if not failure_stage and ir is not None:
        t0 = _now()
        try:
            estimation = _run_estimation_placeholder(ir)
            stages["estimation"] = PipelineStageTrace(ok=True, t_sec=_now() - t0)
        except Exception as e:
            stages["estimation"] = PipelineStageTrace(
                ok=False, t_sec=_now() - t0, error_type=type(e).__name__, error_msg=str(e), traceback=_short_tb()
            )
            # estimation 失败不强制终止主流程（但这里保持 failure_stage 不变）
            estimation = {"ok": False, "error": str(e)}

    # timings
    timings = {k: v.t_sec for k, v in stages.items()}

    # trace：只保留 verifier 纠错需要的最小信息；不含 prompts / llm 原文 / json文本
    pid = (data.get("meta") or {}).get("problem_id", problem_id or "gm4opt_instance")
    trace: Dict[str, Any] = {
        "meta": {
            "problem_id": pid,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "timelimit_sec": config.timelimit_sec,
        },
        "failure_stage": failure_stage,
        "stages": {k: asdict(v) for k, v in stages.items()},
        "timings": timings,
        "features": {
            "graph_built": int(graph_built),
            "n_sets": len(data.get("sets") or []) if isinstance(data.get("sets"), list) else None,
            "n_params": len(data.get("params") or []) if isinstance(data.get("params"), list) else None,
            "n_vars": len(data.get("vars") or []) if isinstance(data.get("vars"), list) else None,
            "n_constraints": len(data.get("constraints") or []) if isinstance(data.get("constraints"), list) else None,
        },
        "verifier": verifier_report,
        "solver": {
            "build_ok": solver_build_ok,
            "build_error": solver_build_error,
            "status": gurobi_status,
            "status_name": gurobi_status_name,
            "obj_value": gurobi_obj_value,
            "metrics": gurobi_metrics,
        },
        "estimation": estimation,
        # verifier 离线复核/重放最需要：IR dict（不含LLM文本）
        "ir_dict": data,
    }

    return PipelineResult(
        ir_dict=data,
        ir=ir,
        graph_built=graph_built,
        verifier_report=verifier_report,
        estimation=estimation,
        solver_build_ok=solver_build_ok,
        solver_build_error=solver_build_error,
        gurobi_status=gurobi_status,
        gurobi_status_name=gurobi_status_name,
        gurobi_obj_value=gurobi_obj_value,
        gurobi_metrics=gurobi_metrics,
        stages=stages,
        failure_stage=failure_stage,
        timings=timings,
        trace=trace,
    )
