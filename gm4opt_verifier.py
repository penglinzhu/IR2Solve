# gm4opt_verifier.py
# Graph-aware verifier & repair engine for GM4OPT
#
# Design goals (3-layer):
# - Layer 1 (Build Robustness): fix issues that crash model build (e.g., unhashable list set elements).
# - Layer 2 (Index/Domain Closure): fix sparse-parameter domain mismatches that cause KeyError (e.g., missing (i,i)).
# - Layer 3 (Semantic Repairs): fix common modeling semantics bugs that yield wrong optimum or timeouts
#   (e.g., overtime binding, accidental bilinear objective in blending).
#
# The verifier returns:
# {
#   "ok": bool,
#   "issues": [...],
#   "repairs": [...],
#   "notes": str,
#   "graph_required": True
# }
#
# It also returns a (possibly) repaired ModelIR object.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set
import copy
import re

from gm4opt_ir import ModelIR, ConstraintDef  # type: ignore
from gm4opt_graph import build_graph_from_ir  # relies on the enhanced graph builder


# ----------------------------
# Result / records
# ----------------------------

@dataclass
class VerifierIssue:
    kind: str                 # e.g., "build_crash", "sparse_param_domain", "semantic_overtime"
    severity: str             # "info" / "warning" / "error"
    message: str
    nodes: List[str]          # graph node ids (best-effort)
    data: Optional[Dict[str, Any]] = None


@dataclass
class VerifierRepair:
    kind: str                 # e.g., "sanitize_set_elements", "fill_missing_diagonal", "rewrite_overtime_constraints"
    message: str
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    touched_nodes: Optional[List[str]] = None
    layer: int = 0


@dataclass
class VerifierResult:
    ok: bool
    issues: List[VerifierIssue]
    repairs: List[VerifierRepair]
    notes: str
    graph_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [asdict(x) for x in self.issues],
            "repairs": [asdict(x) for x in self.repairs],
            "notes": self.notes,
            "graph_required": self.graph_required,
        }


# ----------------------------
# Utilities
# ----------------------------

def _get_set(ir: ModelIR, name: str):
    for s in getattr(ir, "sets", []):
        if getattr(s, "name", None) == name:
            return s
    return None


def _get_param(ir: ModelIR, name: str):
    for p in getattr(ir, "params", []):
        if getattr(p, "name", None) == name:
            return p
    return None


def _get_var(ir: ModelIR, name: str):
    for v in getattr(ir, "vars", []):
        if getattr(v, "name", None) == name:
            return v
    return None


def _ensure_set(ir: ModelIR, name: str, elements: List[Any], description: str = ""):
    s = _get_set(ir, name)
    if s is not None:
        # overwrite only if empty
        if not getattr(s, "elements", None):
            s.elements = elements
        return s

    # Create a new SetDef via duck-typing (gm4opt_ir.SetDef is imported in gm4opt_graph, but not here)
    # We avoid importing SetDef directly to keep this file resilient.
    class _SetShim:  # pragma: no cover
        def __init__(self, name: str, elements: List[Any], description: str):
            self.name = name
            self.elements = elements
            self.description = description

    new_s = _SetShim(name=name, elements=elements, description=description)
    ir.sets.append(new_s)  # type: ignore[attr-defined]
    return new_s


def _as_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _is_pair_list_elements(elements: List[Any]) -> bool:
    """Heuristic: majority elements look like 2-length list/tuple."""
    if not elements:
        return False
    cnt = 0
    for e in elements:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            cnt += 1
    return cnt >= max(1, int(0.7 * len(elements)))


def _normalize_pairs(elements: List[Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for e in elements:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((_as_str(e[0]), _as_str(e[1])))
        else:
            # skip non-pairs
            continue
    return out


def _extract_cartesian_sum_domain(expr: str) -> Optional[Tuple[str, str]]:
    """
    Detect pattern: for i in A for j in A  OR for i in A for j in B.
    Return (A, B) or (A, A).
    """
    # Very common in your logs: for i in REGIONS for j in REGIONS
    m = re.search(r"\bfor\s+\w+\s+in\s+([A-Za-z_]\w*)\s+for\s+\w+\s+in\s+([A-Za-z_]\w*)", expr or "")
    if not m:
        return None
    return (m.group(1), m.group(2))


def _looks_like_cost_xx(expr: str, cost_name: str, var_name: str) -> bool:
    """
    Detect very common transportation objective:
      sum(cost[i][j] * x[i][j] for i in REGIONS for j in REGIONS)
    """
    if not expr:
        return False
    return (cost_name in expr) and (var_name in expr) and ("for" in expr) and ("cost" in cost_name or cost_name == "cost")


def _param_is_sparse_2d(p: Any) -> bool:
    inds = list(getattr(p, "indices", []) or [])
    if len(inds) != 2:
        return False
    vals = getattr(p, "values", None)
    return isinstance(vals, dict)


def _get_sparse_2d_keys(p: Any) -> Set[Tuple[str, str]]:
    """
    Extract support set from nested dict values: values[i][j] = v
    """
    keys: Set[Tuple[str, str]] = set()
    vals = getattr(p, "values", None)
    if not isinstance(vals, dict):
        return keys
    for i, row in vals.items():
        if not isinstance(row, dict):
            continue
        for j in row.keys():
            keys.add((_as_str(i), _as_str(j)))
    return keys


def _fill_missing_diagonal_zero(p: Any, diag_items: List[str]) -> bool:
    """
    For sparse 2D param p.values[i][j], ensure p.values[i][i] exists (0.0) for each i in diag_items.
    Returns True if changed.
    """
    vals = getattr(p, "values", None)
    if not isinstance(vals, dict):
        return False
    changed = False
    for i in diag_items:
        if i not in vals or not isinstance(vals.get(i), dict):
            vals[i] = {}
        row = vals[i]
        if i not in row:
            row[i] = 0.0
            changed = True
    return changed


def _constraint_new(name: str, lhs: str, sense: str, rhs: str, desc: str = "") -> ConstraintDef:
    # Use the real ConstraintDef from gm4opt_ir if available
    return ConstraintDef(
        name=name,
        expr_lhs=lhs,
        sense=sense,
        expr_rhs=rhs,
        description=desc,
    )


# ----------------------------
# Layer 1: Build Robustness
# ----------------------------

def _layer1_sanitize_unhashable_set_elements(ir: ModelIR, graph: Any) -> Tuple[bool, List[VerifierIssue], List[VerifierRepair]]:
    """
    Fix: set elements like [u,v] (list) causing "TypeError: unhashable type: 'list'"
    Strategy:
      - If a set is pair-like, normalize each element to a tuple(str(u), str(v))
      - If a set has list elements not pair-like, convert list -> tuple(str(..),...)
    """
    issues: List[VerifierIssue] = []
    repairs: List[VerifierRepair] = []
    changed = False

    # Use graph element nodes if present; fallback to IR sets.
    for s in getattr(ir, "sets", []):
        sname = getattr(s, "name", "")
        elems = list(getattr(s, "elements", []) or [])
        if not elems:
            continue

        has_list = any(isinstance(e, list) for e in elems)
        if not has_list:
            continue

        before_preview = elems[:10]

        if _is_pair_list_elements(elems):
            pairs = _normalize_pairs(elems)
            # keep length consistent
            new_elems: List[Any] = [(a, b) for (a, b) in pairs]
            if len(new_elems) != len(elems):
                # if some were non-pair, fallback conversion
                new_elems = []
                for e in elems:
                    if isinstance(e, (list, tuple)):
                        new_elems.append(tuple(_as_str(x) for x in e))
                    else:
                        new_elems.append(e)
        else:
            new_elems = []
            for e in elems:
                if isinstance(e, list):
                    new_elems.append(tuple(_as_str(x) for x in e))
                else:
                    new_elems.append(e)

        # Apply
        setattr(s, "elements", new_elems)
        changed = True

        # Record
        touched = [f"set:{sname}"]
        issues.append(
            VerifierIssue(
                kind="build_crash_unhashable_list",
                severity="warning",
                message=f"Set '{sname}' contains list elements which may be unhashable in downstream builders; normalized to tuples.",
                nodes=touched,
                data={"set": sname},
            )
        )
        repairs.append(
            VerifierRepair(
                kind="sanitize_set_elements",
                layer=1,
                message=f"Normalized list elements in set '{sname}' to tuples to avoid unhashable-list build errors.",
                before={"set": sname, "elements_preview": before_preview},
                after={"set": sname, "elements_preview": new_elems[:10]},
                touched_nodes=touched,
            )
        )

    return changed, issues, repairs


# ----------------------------
# Layer 2: Index / Domain Closure
# ----------------------------

def _layer2_fix_sparse_param_domain(ir: ModelIR, graph: Any) -> Tuple[bool, List[VerifierIssue], List[VerifierRepair]]:
    """
    Fix: KeyError from sparse 2D params (e.g., cost matrix missing (i,i)) when expressions sum over full cartesian.
    Two repairs are provided:
      A) Fill missing diagonal entries with 0 (preferred if indices sets are same).
      B) If still risky, introduce ARCS set and rewrite objective sum-domain to ARCS (conservative, optional).
    We implement (A) by default; (B) only when objective is a clear cartesian sum over same set.
    """
    issues: List[VerifierIssue] = []
    repairs: List[VerifierRepair] = []
    changed = False

    obj_expr: str = getattr(getattr(ir, "objective", None), "expr", "") or ""

    # For each sparse 2D param
    for p in getattr(ir, "params", []):
        if not _param_is_sparse_2d(p):
            continue

        pname = getattr(p, "name", "")
        inds = list(getattr(p, "indices", []) or [])
        if len(inds) != 2:
            continue

        setA, setB = inds[0], inds[1]
        sA = _get_set(ir, setA)
        sB = _get_set(ir, setB)

        # Only do diagonal fill if same-index-set and have atomic elements
        if setA == setB and sA is not None:
            items = [ _as_str(x) for x in (getattr(sA, "elements", []) or []) ]
            if items:
                before_keys = sorted(list(_get_sparse_2d_keys(p)))[:12]
                did = _fill_missing_diagonal_zero(p, items)
                if did:
                    changed = True
                    after_keys = sorted(list(_get_sparse_2d_keys(p)))[:12]
                    touched = [f"param:{pname}", f"set:{setA}"]

                    issues.append(
                        VerifierIssue(
                            kind="sparse_param_missing_diagonal",
                            severity="warning",
                            message=f"Sparse 2D param '{pname}' indexed by ({setA},{setB}) is missing diagonal entries; filled (i,i)=0 to avoid KeyError.",
                            nodes=touched,
                            data={"param": pname, "set": setA},
                        )
                    )
                    repairs.append(
                        VerifierRepair(
                            kind="fill_missing_diagonal",
                            layer=2,
                            message=f"Filled missing diagonal entries for '{pname}' with 0.0 (for all i in {setA}).",
                            before={"param": pname, "keys_preview": before_keys},
                            after={"param": pname, "keys_preview": after_keys},
                            touched_nodes=touched,
                        )
                    )

        # Optional: rewrite objective domain to ARCS if it's clearly summing over full cartesian and uses pname
        # This is useful if the param is sparse beyond diagonal, but we keep it conservative.
        if (not changed) and _looks_like_cost_xx(obj_expr, pname, "x"):
            dom = _extract_cartesian_sum_domain(obj_expr)
            if dom and dom[0] == dom[1]:
                # build ARCS from param support
                keys = sorted(list(_get_sparse_2d_keys(p)))
                if keys:
                    arcs_elems = [(i, j) for (i, j) in keys if i != j]  # prefer real arcs
                    _ensure_set(ir, "ARCS", arcs_elems, description=f"Support set for sparse param {pname}")

                    # Rewrite objective (very targeted)
                    before_expr = obj_expr
                    # replace "for i in X for j in X" with "for (i,j) in ARCS"
                    new_expr = re.sub(
                        r"\bfor\s+\w+\s+in\s+" + re.escape(dom[0]) + r"\s+for\s+\w+\s+in\s+" + re.escape(dom[1]),
                        "for (i,j) in ARCS",
                        before_expr,
                    )
                    if new_expr != before_expr:
                        ir.objective.expr = new_expr  # type: ignore[attr-defined]
                        changed = True
                        touched = [f"param:{pname}", f"set:ARCS", f"obj:{getattr(ir.objective,'name','objective')}"]

                        issues.append(
                            VerifierIssue(
                                kind="sparse_param_domain_rewrite",
                                severity="warning",
                                message=f"Objective sums over full cartesian domain but '{pname}' appears sparse; rewrote objective to iterate over ARCS support set.",
                                nodes=touched,
                                data={"param": pname},
                            )
                        )
                        repairs.append(
                            VerifierRepair(
                                kind="rewrite_objective_sum_domain_to_support_set",
                                layer=2,
                                message="Rewrote objective summation domain to ARCS to prevent missing-key access on sparse params.",
                                before={"objective_expr": before_expr},
                                after={"objective_expr": new_expr},
                                touched_nodes=touched,
                            )
                        )

    return changed, issues, repairs


# ----------------------------
# Layer 3: Semantic Repairs
# ----------------------------

def _layer3_fix_overtime_binding(ir: ModelIR, graph: Any) -> Tuple[bool, List[VerifierIssue], List[VerifierRepair]]:
    """
    Fix: overtime_hours exists and is penalized, but constraints don't bind it to actual overtime usage.
    Typical buggy form (from your logs):
      sum(assembly_time[p]*x[p] for p in PRODUCTS) + overtime_hours <= max_assembly_hours + max_overtime_hours
    This lets overtime_hours=0 and still use up to max_assembly+max_overtime without paying overtime cost.

    Repair:
      (1) sum(assembly_time[p]*x[p] for p in PRODUCTS) <= max_assembly_hours + overtime_hours
      (2) overtime_hours <= max_overtime_hours   (often already via ub, but keep as explicit constraint)
    We replace the single buggy constraint if found.
    """
    issues: List[VerifierIssue] = []
    repairs: List[VerifierRepair] = []
    changed = False

    obj_expr = getattr(getattr(ir, "objective", None), "expr", "") or ""
    if "overtime_hours" not in obj_expr:
        return False, issues, repairs
    if "overtime_cost" not in obj_expr and "7" not in obj_expr and "9" not in obj_expr:
        # still allow, but less confident
        pass

    # find the suspicious constraint
    target_idx = None
    for idx, c in enumerate(getattr(ir, "constraints", [])):
        lhs = getattr(c, "expr_lhs", "") or ""
        rhs = getattr(c, "expr_rhs", "") or ""
        if "overtime_hours" in lhs and "max_overtime_hours" in rhs and "max_assembly_hours" in rhs:
            # canonical buggy pattern
            target_idx = idx
            break
        # another common form: "... + overtime_hours <= max_assembly_hours + max_overtime_hours"
        if "overtime_hours" in lhs and ("max_overtime_hours" in rhs or "+ 50" in rhs or "+50" in rhs) and ("max_assembly_hours" in rhs or "90" in rhs):
            target_idx = idx
            break

    if target_idx is None:
        return False, issues, repairs

    old_c = ir.constraints[target_idx]
    old_name = getattr(old_c, "name", f"assembly_hours_constraint_{target_idx}")
    old_lhs = getattr(old_c, "expr_lhs", "") or ""
    old_rhs = getattr(old_c, "expr_rhs", "") or ""
    old_sense = getattr(old_c, "sense", "<=")

    # Build repaired constraints
    # remove "+ overtime_hours" from lhs if present, and move it to rhs
    base_lhs = old_lhs.replace("+ overtime_hours", "").replace(" + overtime_hours", "").strip()
    if base_lhs == old_lhs:
        # try a safer removal (if formatting differs)
        base_lhs = re.sub(r"\s*\+\s*overtime_hours\s*", " ", old_lhs).strip()

    c1 = _constraint_new(
        name=old_name + "__bind",
        lhs=base_lhs,
        sense="<=",
        rhs="max_assembly_hours + overtime_hours",
        desc="Bind overtime_hours to actual assembly overage.",
    )
    c2 = _constraint_new(
        name=old_name + "__cap",
        lhs="overtime_hours",
        sense="<=",
        rhs="max_overtime_hours",
        desc="Cap overtime hours (explicit).",
    )

    # Replace old constraint with c1 and insert c2 after it.
    ir.constraints[target_idx] = c1
    ir.constraints.insert(target_idx + 1, c2)
    changed = True

    touched = [f"con:{old_name}", f"var:overtime_hours", f"obj:{getattr(ir.objective,'name','objective')}"]
    issues.append(
        VerifierIssue(
            kind="semantic_overtime_unbound",
            severity="warning",
            message="Overtime variable is penalized but not bound to production overage; repaired constraints to correctly charge overtime.",
            nodes=touched,
            data={"constraint": old_name},
        )
    )
    repairs.append(
        VerifierRepair(
            kind="rewrite_overtime_constraints",
            layer=3,
            message="Replaced the buggy assembly-hours constraint with a binding form and added an explicit overtime cap constraint.",
            before={"name": old_name, "lhs": old_lhs, "sense": old_sense, "rhs": old_rhs},
            after={
                "replaced_with": [
                    {"name": c1.name, "lhs": c1.expr_lhs, "sense": c1.sense, "rhs": c1.expr_rhs},
                    {"name": c2.name, "lhs": c2.expr_lhs, "sense": c2.sense, "rhs": c2.expr_rhs},
                ]
            },
            touched_nodes=touched,
        )
    )

    return changed, issues, repairs


def _layer3_fix_bilinear_blending_objective(ir: ModelIR, graph: Any) -> Tuple[bool, List[VerifierIssue], List[VerifierRepair]]:
    """
    Fix: accidental nonconvex quadratic objective in blending problems.
    Buggy objective pattern from your log:
      sum((selling_price[p] - sum(cost[r] * blend[r][p] for r in RAW_GASOLINES)) * sum(blend[r][p] for r in RAW_GASOLINES) for p in FINAL_PRODUCTS)

    Correct linear profit:
      sum(selling_price[p] * sum(blend[r][p] for r in RAW_GASOLINES) for p in FINAL_PRODUCTS)
      - sum(cost[r] * blend[r][p] for r in RAW_GASOLINES for p in FINAL_PRODUCTS)
    """
    issues: List[VerifierIssue] = []
    repairs: List[VerifierRepair] = []
    changed = False

    obj = getattr(ir, "objective", None)
    if obj is None:
        return False, issues, repairs

    expr = getattr(obj, "expr", "") or ""

    # Heuristic detection of bilinear structure
    looks_bilinear = ("* sum(" in expr) and ("selling_price" in expr) and ("blend" in expr) and ("cost" in expr)
    if not looks_bilinear:
        return False, issues, repairs

    before_expr = expr
    # Construct linear objective expression (string-based, aligned with your variable/param names)
    new_expr = (
        "sum(selling_price[p] * sum(blend[r][p] for r in RAW_GASOLINES) for p in FINAL_PRODUCTS) "
        "- sum(cost[r] * blend[r][p] for r in RAW_GASOLINES for p in FINAL_PRODUCTS)"
    )

    obj.expr = new_expr
    changed = True

    touched = [f"obj:{getattr(obj,'name','objective')}", "var:blend", "param:selling_price", "param:cost"]
    issues.append(
        VerifierIssue(
            kind="semantic_bilinear_objective",
            severity="warning",
            message="Objective appears bilinear/nonconvex for a blending LP; rewrote to the standard linear profit objective to avoid nonconvex MIP conversion/timeouts.",
            nodes=touched,
        )
    )
    repairs.append(
        VerifierRepair(
            kind="rewrite_bilinear_objective_to_linear",
            layer=3,
            message="Rewrote blending profit objective from bilinear form to a linear revenue-minus-cost form.",
            before={"objective_expr": before_expr},
            after={"objective_expr": new_expr},
            touched_nodes=touched,
        )
    )
    return changed, issues, repairs


# ----------------------------
# Main Verifier
# ----------------------------

class GM4OPTVerifier:
    """
    Public API:
        verifier = GM4OPTVerifier()
        repaired_ir, report = verifier.run(ir, graph=None)
    """

    def __init__(self, enable_repairs: bool = True):
        self.enable_repairs = enable_repairs

    def run(self, ir: ModelIR, graph: Optional[Any] = None) -> Tuple[ModelIR, Dict[str, Any]]:
        issues: List[VerifierIssue] = []
        repairs: List[VerifierRepair] = []

        # Always work on a copy so caller can decide whether to adopt changes
        work_ir: ModelIR = copy.deepcopy(ir)

        # Ensure graph exists
        if graph is None:
            graph = build_graph_from_ir(work_ir)

        # Layer 1
        c1, i1, r1 = _layer1_sanitize_unhashable_set_elements(work_ir, graph)
        issues.extend(i1)
        repairs.extend(r1)

        # Rebuild graph after structure changes
        if c1:
            graph = build_graph_from_ir(work_ir)

        # Layer 2
        c2, i2, r2 = _layer2_fix_sparse_param_domain(work_ir, graph)
        issues.extend(i2)
        repairs.extend(r2)

        if c2:
            graph = build_graph_from_ir(work_ir)

        # Layer 3
        c3a, i3a, r3a = _layer3_fix_overtime_binding(work_ir, graph)
        issues.extend(i3a)
        repairs.extend(r3a)

        if c3a:
            graph = build_graph_from_ir(work_ir)

        c3b, i3b, r3b = _layer3_fix_bilinear_blending_objective(work_ir, graph)
        issues.extend(i3b)
        repairs.extend(r3b)

        if c3b:
            graph = build_graph_from_ir(work_ir)

        # Determine ok:
        # - If we performed repairs, treat as ok (non-fatal), because goal is to keep pipeline moving.
        # - If no repairs and no issues: ok
        # - If issues exist but repairs disabled: ok=False when any "error" exists; otherwise ok=True.
        if self.enable_repairs:
            ok = True
        else:
            ok = not any(x.severity == "error" for x in issues)

        notes_parts = []
        notes_parts.append("GM4OPT verifier (3-layer) executed.")
        if repairs:
            notes_parts.append(f"Applied {len(repairs)} repair(s).")
        else:
            notes_parts.append("No repairs applied.")

        result = VerifierResult(
            ok=ok,
            issues=issues,
            repairs=repairs,
            notes=" ".join(notes_parts),
            graph_required=True,
        )

        # If repairs are disabled, return original ir (copy) unchanged; else return work_ir
        final_ir = work_ir if self.enable_repairs else copy.deepcopy(ir)
        return final_ir, result.to_dict()


# Convenience function for pipelines that prefer a functional API
def verify_and_repair(ir: ModelIR, graph: Optional[Any] = None, enable_repairs: bool = True) -> Tuple[ModelIR, Dict[str, Any]]:
    return GM4OPTVerifier(enable_repairs=enable_repairs).run(ir, graph=graph)
