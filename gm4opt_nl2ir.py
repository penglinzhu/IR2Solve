# gm4opt_nl2ir.py
# NL -> JSON IR text -> ModelIR 解析模块

from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any, Dict, List

from gm4opt_ir import (
    MetaInfo,
    SetDef,
    ParamDef,
    VarDef,
    ObjectiveDef,
    ConstraintDef,
    ModelIR,
)

# =============================================================================
# 1) Prompt Constants (static; no type-hint; no dynamic prompt template)
# =============================================================================

BASE_SYSTEM_PROMPT = """
You are an expert in mathematical modeling and optimization.
Your task is to read a natural language description of a linear / integer
optimization problem and output a JSON object describing the model in a
strictly structured Intermediate Representation (IR) called ModelIR.

You MUST:
- Produce valid JSON (no comments, no trailing commas).
- Output JSON ONLY (no markdown, no explanation text).
- Follow the given schema exactly.
""".strip()


SCHEMA_AND_INSTRUCTIONS = r"""
You must generate a JSON object with the following top-level keys:

{
  "meta": { ... },
  "sets": [ ... ],
  "params": [ ... ],
  "vars": [ ... ],
  "objective": { ... },
  "constraints": [ ... ],
  "graph": null
}

1. meta
---------
meta = {
  "problem_id": str,
  "source": null or str,
  "description": str,
  "sense": "min" or "max",
  "version": 1
}

2. sets
---------
Each set is:

{
  "name": "CONTAINERS",
  "elements": [...],
  "description": "..."
}

Index consistency for sets/params/vars/expressions (IMPORTANT):
- All set elements that are used as indices in params/vars/expressions
  MUST be strings, even if they look like numbers.

3. params
---------
Each parameter is:

{
  "name": "time",
  "indices": ["WORKERS", "TASKS"], # 0D: [], 1D: ["SET"], 2D: ["SET1","SET2"]
  "values": { ... },
  "description": "..."
}

Dimensionality rules:
- 0D (scalar): indices=[], values is a number
- 1D: indices=["SET"], values is {key:value}
- 2D: indices=["SET1","SET2"], values is nested dict values[i][j]

Param usage in expressions (0D scalar) (IMPORTANT):
- If a parameter is 0D (scalar, "indices": []), refer to it DIRECTLY by name.

4. vars
---------
Each variable is:

{
  "name": "x",
  "indices": ["WORKERS", "TASKS"], # 0D, 1D, or 2D
  "vartype": "binary" | "integer" | "continuous",
  "lb": 0.0,
  "ub": null or number,
  "description": "..."
}

How to choose vartype (IMPORTANT):
- Use BINARY for yes/no selection/assignment/on-off decisions.
- Default to INTEGER for decision variables that represent counts of indivisible things
  (e.g., number of items/people/vehicles/orders/servings/trips).
- Use CONTINUOUS ONLY when you have POSITIVE EVIDENCE that fractional values are allowed.
  Evidence examples: "fractional", "can be split", "divisible", "continuous", "real-valued",
  "any amount", "arbitrary amount", "portion can be fractional", "half", "0.5".
- If the statement is ambiguous about divisibility, choose INTEGER (conservative).
- Natural continuous quantities typically include: flow/shipment amount, weight (kg/grams/tons),
  volume (liters), time (hours), energy, money, rates (GB/s).

5. objective
---------
{
  "name": "min_total_cost",
  "sense": "min" or "max",
  "expr": "Python expression using sets/params/vars",
  "description": "..."
}

6. constraints
---------
Each constraint is an object:

{
  "name": "constraint_name",
  "expr_lhs": "Python expression (scalar)",
  "sense": "<=" | ">=" | "==",
  "expr_rhs": "Python expression (scalar)",
  "description": "..."
}

CRITICAL RULES FOR CONSTRAINTS:
(1) Every constraint MUST be a fully specified SCALAR constraint.
(2) In constraint expressions, you MUST NOT use Python "for" loops, generator expressions, or comprehensions.
(3) If a conceptual constraint holds "for each/for all", you MUST UNROLL it into one constraint per element.
(4) Use string indices like x['I']['A'] (not x[I][A], not x[1]).

Constraint sense guidance (IMPORTANT):
- Map words to senses strictly:
  * "at least", "no less than", "needs", "must have minimum", "fulfill requirements" -> use ">=".
  * "at most", "no more than", "capacity", "limit" -> use "<=".
  * "exactly", "must equal", "balanced exactly", "equal to" -> use "==".
- DO NOT use "==" unless the statement explicitly requires exact equality.
"""

# =============================================================================
# 2) Prompt Assembly (static)
# =============================================================================

def build_system_prompt() -> str:
    """
    Static system prompt builder.
    - No problem-type hinting.
    - No dynamic prompt template.
    Parameters are kept for backward compatibility with callers.
    """
    return BASE_SYSTEM_PROMPT


def build_user_prompt(question_text: str) -> str:
    """
    Build the user message content: include schema/instructions + the NL question.
    """
    return (
        SCHEMA_AND_INSTRUCTIONS
        + "\n\nNow read the following optimization problem and output the JSON IR:\n\n"
        + (question_text or "")
    )

# =============================================================================
# 3) Output Extraction / Parsing
# =============================================================================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a model output text.

    Supports two common patterns:
    - A fenced code block ```json ... ```
    - A raw JSON object in the text
    """
    code_block_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    m = code_block_pattern.search(text or "")
    if m:
        json_str = m.group(1)
        return json.loads(json_str)

    raw = text or ""
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        json_str = raw[first : last + 1]
        return json.loads(json_str)

    return json.loads(raw)

# =============================================================================
# 4) JSON → ModelIR Conversion
# =============================================================================

def _filter_kwargs_for(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter raw dict to only include fields that exist in dataclass `cls`.
    Drop extra keys (e.g., LLM sometimes adds unknown keys).
    """
    field_names = {f.name for f in fields(cls)}
    extra_keys = [k for k in raw.keys() if k not in field_names]
    if extra_keys:
        print(f"[NL2IR WARN] Dropping unknown keys for {cls.__name__}: {extra_keys}")
    return {k: v for k, v in raw.items() if k in field_names}


def json_to_model_ir(data: Dict[str, Any]) -> ModelIR:
    """
    Convert JSON dict (following the schema) to a ModelIR instance.
    Unknown / extra keys in objects will be dropped.
    """
    meta = MetaInfo(**_filter_kwargs_for(MetaInfo, data["meta"]))

    sets = [
        SetDef(**_filter_kwargs_for(SetDef, s))
        for s in data.get("sets", [])
    ]

    params = [
        ParamDef(**_filter_kwargs_for(ParamDef, p))
        for p in data.get("params", [])
    ]

    vars_ = [
        VarDef(**_filter_kwargs_for(VarDef, v))
        for v in data.get("vars", [])
    ]

    obj = ObjectiveDef(**_filter_kwargs_for(ObjectiveDef, data["objective"]))

    constraints = [
        ConstraintDef(**_filter_kwargs_for(ConstraintDef, c))
        for c in data.get("constraints", [])
    ]

    ir = ModelIR(
        meta=meta,
        sets=sets,
        params=params,
        vars=vars_,
        objective=obj,
        constraints=constraints,
        graph=None,
    )
    return ir
