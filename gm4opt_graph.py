# gm4opt_graph.py
# 从 ModelIR 自动构建 GraphIR（增强版：支持 pattern matching / graph rewrite 的 verifier）
#
# 关键增强：
# 1) 增加 expr / element / param_entry 节点，显式化“表达式层”“集合元素层”“参数稀疏支持集”
# 2) 为 expr 节点提取 iter_sets / idents（用于检测 sum-domain / 索引闭包 / 变量绑定等）
# 3) 自检逻辑增强：unused_set 不再仅看“度数=0”，而看是否真正索引 param/var 或出现在迭代域
# 4) 提供更细粒度边类型：var-in-expr/param-in-expr/obj-has-expr/con-has-lhs-expr/con-has-rhs-expr 等
#
# 兼容性：保留旧的 var-in-objective / param-in-objective / var-in-constraint / param-in-constraint 边。

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Set, Tuple, Optional
import re

from gm4opt_ir import (
    ModelIR,
    GraphIR,
    GraphNode,
    GraphEdge,
)


# ----------------------------
# Graph check issue
# ----------------------------
@dataclass
class GraphCheckIssue:
    """
    图结构自检结果

    kind:     问题类型，例如 "orphan_var", "unused_param", "disconnected_component"
    severity: "info" / "warning" / "error"
    message:  简要说明
    nodes:    相关 node_id 列表，方便定位
    """
    kind: str
    severity: str
    message: str
    nodes: List[str]


# ----------------------------
# Small parsing helpers
# ----------------------------
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_ITER_SET_RE = re.compile(r"\bfor\s+\w+\s+in\s+([A-Za-z_]\w*)")

_PY_KEYWORDS = {
    "for", "in", "if", "else", "elif", "and", "or", "not", "True", "False", "None",
    "sum", "range", "len", "min", "max", "abs", "enumerate", "quicksum",
}
# 允许的“结构关键词”不一定都要过滤，这里只做轻量过滤，verifier 可再二次过滤


def _extract_iter_sets(expr: str) -> List[str]:
    """从类似 '... for i in REGIONS for j in REGIONS' 中提取迭代域集合名。"""
    if not expr:
        return []
    return _ITER_SET_RE.findall(expr)


def _extract_idents(expr: str) -> List[str]:
    """提取表达式中的标识符 token（用于快速依赖判断）。"""
    if not expr:
        return []
    toks = _IDENT_RE.findall(expr)
    # 轻量过滤：去重、保持顺序
    seen: Set[str] = set()
    out: List[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _infer_element_kind(x: Any) -> str:
    """
    给 set element 一个粗粒度类型标注：
    - atom: str/int/float/...
    - list: list/tuple (可能是 pair)
    - dict/other: 其他
    """
    if isinstance(x, (list, tuple)):
        return "list"
    if isinstance(x, (str, int, float, bool)) or x is None:
        return "atom"
    if isinstance(x, dict):
        return "dict"
    return type(x).__name__


def _stable_elem_id(set_name: str, elem: Any, idx: int) -> str:
    """
    为集合元素生成稳定 node_id：
    - 优先使用可读 repr
    - 对不可读/过长的情况，回退到序号
    """
    try:
        r = repr(elem)
    except Exception:
        r = ""
    r = r.replace(" ", "")
    if len(r) == 0 or len(r) > 60:
        r = f"#{idx}"
    return f"elem:{set_name}:{r}"


def _normalize_pair(elem: Any) -> Optional[Tuple[str, str]]:
    """
    若 elem 是 (u,v) 或 [u,v] 形式，返回 (str(u), str(v))
    """
    if isinstance(elem, (list, tuple)) and len(elem) == 2:
        try:
            return (str(elem[0]), str(elem[1]))
        except Exception:
            return None
    return None


# ----------------------------
# Basic adjacency (undirected) for stats/checks
# ----------------------------
def _build_adjacency(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> Dict[str, Set[str]]:
    """构建无向邻接表，用于统计度数和连通分量。"""
    adj: Dict[str, Set[str]] = {nid: set() for nid in nodes.keys()}
    for e in edges:
        if e.source in adj and e.target in adj:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)
    return adj


def _compute_graph_stats(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> Dict[str, Any]:
    """基于 nodes / edges 计算图统计量。"""
    n_nodes = len(nodes)
    n_edges = len(edges)

    by_type: Dict[str, int] = {}
    for n in nodes.values():
        by_type[n.type] = by_type.get(n.type, 0) + 1

    adj = _build_adjacency(nodes, edges)
    degrees = [len(neis) for neis in adj.values()] if adj else [0]
    max_degree = max(degrees) if degrees else 0
    avg_degree = (sum(degrees) / len(degrees)) if degrees else 0.0

    # 连通分量
    visited: Set[str] = set()
    component_sizes: List[int] = []
    for nid in nodes.keys():
        if nid in visited:
            continue
        stack = [nid]
        visited.add(nid)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        component_sizes.append(size)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "nodes_by_type": by_type,
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "n_connected_components": len(component_sizes),
        "component_sizes": component_sizes,
    }


# ----------------------------
# Self checks (enhanced)
# ----------------------------
def _run_graph_self_checks(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> List[GraphCheckIssue]:
    """
    轻量自检（增强版）：
    - orphan_var: 变量未出现在任何 objective/constraints 的 expr 节点中
    - unused_param: 参数未出现在任何 objective/constraints 的 expr 节点中
    - unused_set: 集合既未索引 param/var，也未出现在任何 expr.iter_sets 中
    - obj_without_var: 目标函数 expr 不包含任何变量
    - no_constraint: 没有约束节点
    - disconnected_component: 图存在多个连通分量
    """
    issues: List[GraphCheckIssue] = []

    # 节点分类
    var_nodes = [nid for nid, n in nodes.items() if n.type == "var"]
    param_nodes = [nid for nid, n in nodes.items() if n.type == "param"]
    set_nodes = [nid for nid, n in nodes.items() if n.type == "set"]
    con_nodes = [nid for nid, n in nodes.items() if n.type == "constraint"]
    obj_nodes = [nid for nid, n in nodes.items() if n.type == "objective"]
    expr_nodes = [nid for nid, n in nodes.items() if n.type == "expr"]

    # 快速映射
    var_used: Set[str] = set()
    param_used: Set[str] = set()
    obj_has_var: Dict[str, bool] = {nid: False for nid in obj_nodes}

    # set 是否“被用作迭代域”
    set_used_as_iter: Set[str] = set()
    for enid in expr_nodes:
        attrs = nodes[enid].attrs or {}
        for sname in (attrs.get("iter_sets") or []):
            set_used_as_iter.add(f"set:{sname}")

    # 遍历边：识别 var/param 是否出现在 expr 中；以及 obj 是否引用 var
    for e in edges:
        if e.etype in ("var-in-expr", "var-in-objective", "var-in-constraint"):
            if e.source in var_nodes:
                var_used.add(e.source)
            if e.etype == "var-in-objective" and e.target in obj_has_var:
                obj_has_var[e.target] = True

        if e.etype in ("param-in-expr", "param-in-objective", "param-in-constraint"):
            if e.source in param_nodes:
                param_used.add(e.source)

    # orphan_var
    orphan_vars = [nid for nid in var_nodes if nid not in var_used]
    if orphan_vars:
        issues.append(
            GraphCheckIssue(
                kind="orphan_var",
                severity="warning",
                message=f"{len(orphan_vars)} variable node(s) do not appear in any objective/constraint expression.",
                nodes=sorted(orphan_vars),
            )
        )

    # unused_param
    unused_params = [nid for nid in param_nodes if nid not in param_used]
    if unused_params:
        issues.append(
            GraphCheckIssue(
                kind="unused_param",
                severity="warning",
                message=f"{len(unused_params)} parameter node(s) do not appear in any objective/constraint expression.",
                nodes=sorted(unused_params),
            )
        )

    # obj_without_var
    for obj_id, has_var in obj_has_var.items():
        if not has_var:
            issues.append(
                GraphCheckIssue(
                    kind="obj_without_var",
                    severity="warning",
                    message=f"Objective node {obj_id} does not reference any variable node.",
                    nodes=[obj_id],
                )
            )

    # no_constraint
    if len(con_nodes) == 0:
        issues.append(
            GraphCheckIssue(
                kind="no_constraint",
                severity="warning",
                message="Graph contains no constraint nodes.",
                nodes=[],
            )
        )

    # unused_set（增强：必须真的“参与建模”才算使用）
    # 使用判定：a) set-index-of-param/var 边；b) 被用作 expr 的 iter_sets
    set_indexing: Set[str] = set()
    for e in edges:
        if e.etype in ("set-index-of-param", "set-index-of-var"):
            set_indexing.add(e.source)

    unused_sets = [
        nid for nid in set_nodes
        if (nid not in set_indexing) and (nid not in set_used_as_iter)
    ]
    if unused_sets:
        issues.append(
            GraphCheckIssue(
                kind="unused_set",
                severity="info",
                message=f"{len(unused_sets)} set node(s) do not index any param/var and are not used as an iteration domain.",
                nodes=sorted(unused_sets),
            )
        )

    # disconnected_component
    adj = _build_adjacency(nodes, edges)
    visited: Set[str] = set()
    components_nodes: List[List[str]] = []
    for nid in nodes.keys():
        if nid in visited:
            continue
        stack = [nid]
        visited.add(nid)
        comp: List[str] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components_nodes.append(comp)

    if len(components_nodes) > 1:
        flat_nodes: List[str] = []
        for comp in components_nodes:
            flat_nodes.extend(comp)
        issues.append(
            GraphCheckIssue(
                kind="disconnected_component",
                severity="info",
                message=f"Graph has {len(components_nodes)} connected components.",
                nodes=flat_nodes,
            )
        )

    return issues


# ----------------------------
# Main builder
# ----------------------------
def build_graph_from_ir(ir: ModelIR) -> GraphIR:
    """
    根据 ModelIR 构造 GraphIR（增强版）：
    - set/param/var/objective/constraint 节点
    - 新增：
        * element 节点：显式表示 set.elements
        * param_entry 节点：显式表示稀疏参数的支持集（例如 cost[i][j] 的已给条目）
        * expr 节点：objective / constraint(lhs/rhs) 的表达式节点（附带 iter_sets / idents）
    - 建立依赖边（保留旧边 + 新细粒度边），并进行自检与统计。
    """
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    def add_node(node_id: str, ntype: str, label: Optional[str] = None, attrs: Any = None):
        if node_id not in nodes:
            nodes[node_id] = GraphNode(
                id=node_id,
                type=ntype,
                label=label if label is not None else node_id,
                attrs=attrs,
            )

    def add_edge(src: str, dst: str, etype: str, attrs: Any = None):
        edges.append(GraphEdge(source=src, target=dst, etype=etype, attrs=attrs))

    # ---- 1) sets + elements ----
    for s in ir.sets:
        set_node_id = f"set:{s.name}"
        elems = list(s.elements)

        # set node
        # 元素类型摘要：atom/list/dict/...
        elem_kinds = [_infer_element_kind(e) for e in elems]
        kind_counts: Dict[str, int] = {}
        for k in elem_kinds:
            kind_counts[k] = kind_counts.get(k, 0) + 1

        # 若是“边集”形态（pair list），提取 pairs 便于 verifier
        pairs: List[Tuple[str, str]] = []
        for e in elems:
            p = _normalize_pair(e)
            if p is not None:
                pairs.append(p)

        add_node(
            node_id=set_node_id,
            ntype="set",
            label=s.name,
            attrs={
                "elements": elems,                   # 原始元素（保持）
                "element_kind_counts": kind_counts,  # {atom/list/...}
                "pair_like_count": len(pairs),
                "pairs_preview": pairs[:20],         # 预览（避免太大）
            },
        )

        # element nodes
        for idx, e in enumerate(elems):
            elem_node_id = _stable_elem_id(s.name, e, idx)
            add_node(
                node_id=elem_node_id,
                ntype="element",
                label=elem_node_id.split(":", 2)[-1],
                attrs={
                    "set": s.name,
                    "value": e,
                    "kind": _infer_element_kind(e),
                    "pair": _normalize_pair(e),  # (u,v) if applicable
                    "index": idx,
                },
            )
            add_edge(set_node_id, elem_node_id, etype="set-has-element")

    # ---- 2) params + index edges + sparse entries ----
    for p in ir.params:
        pnode = f"param:{p.name}"
        add_node(
            node_id=pnode,
            ntype="param",
            label=p.name,
            attrs={"indices": list(p.indices), "is_sparse": isinstance(p.values, dict)},
        )
        for set_name in p.indices:
            set_node_id = f"set:{set_name}"
            if set_node_id in nodes:
                add_edge(set_node_id, pnode, etype="set-index-of-param")

        # 尝试把 dict 参数 values 显式展开成 param_entry 节点
        # 支持两类常见形状：
        #  (A) 1D: values = {"i": v}
        #  (B) 2D nested: values = {"i": {"j": v}}
        if isinstance(p.values, dict) and len(p.indices) > 0:
            if len(p.indices) == 1:
                for k, v in p.values.items():
                    enode = f"entry:{p.name}:{k}"
                    add_node(
                        node_id=enode,
                        ntype="param_entry",
                        label=f"{p.name}[{k}]",
                        attrs={"param": p.name, "key": (str(k),), "value": v},
                    )
                    add_edge(pnode, enode, etype="param-has-entry")

                    # link to element node if exists
                    set0 = p.indices[0]
                    elem0 = f"elem:{set0}:{repr(k).replace(' ', '')}"
                    if elem0 in nodes:
                        add_edge(enode, elem0, etype="entry-index-0")
            elif len(p.indices) == 2:
                # nested dict preferred
                for i, row in p.values.items():
                    if not isinstance(row, dict):
                        continue
                    for j, v in row.items():
                        enode = f"entry:{p.name}:{i},{j}"
                        add_node(
                            node_id=enode,
                            ntype="param_entry",
                            label=f"{p.name}[{i},{j}]",
                            attrs={"param": p.name, "key": (str(i), str(j)), "value": v},
                        )
                        add_edge(pnode, enode, etype="param-has-entry")

                        set0, set1 = p.indices[0], p.indices[1]
                        elem0 = f"elem:{set0}:{repr(i).replace(' ', '')}"
                        elem1 = f"elem:{set1}:{repr(j).replace(' ', '')}"
                        if elem0 in nodes:
                            add_edge(enode, elem0, etype="entry-index-0")
                        if elem1 in nodes:
                            add_edge(enode, elem1, etype="entry-index-1")
            else:
                # 高维不展开（避免爆炸），但保留稀疏提示
                pass

    # ---- 3) vars + index edges ----
    for v in ir.vars:
        vnode = f"var:{v.name}"
        add_node(
            node_id=vnode,
            ntype="var",
            label=v.name,
            attrs={"indices": list(v.indices), "vartype": v.vartype, "lb": v.lb, "ub": v.ub},
        )
        for set_name in v.indices:
            set_node_id = f"set:{set_name}"
            if set_node_id in nodes:
                add_edge(set_node_id, vnode, etype="set-index-of-var")

    # ---- 4) objective + objective expr node ----
    obj = ir.objective
    obj_node_id = f"obj:{obj.name}"
    add_node(
        node_id=obj_node_id,
        ntype="objective",
        label=obj.name,
        attrs={"sense": obj.sense, "expr": obj.expr},
    )

    obj_expr_id = f"expr:obj:{obj.name}"
    add_node(
        node_id=obj_expr_id,
        ntype="expr",
        label=f"expr({obj.name})",
        attrs={
            "owner": obj_node_id,
            "role": "objective",
            "expr": obj.expr or "",
            "iter_sets": _extract_iter_sets(obj.expr or ""),
            "idents": _extract_idents(obj.expr or ""),
        },
    )
    add_edge(obj_node_id, obj_expr_id, etype="obj-has-expr")

    # ---- 5) constraints + lhs/rhs expr nodes ----
    for c in ir.constraints:
        con_id = f"con:{c.name}"
        add_node(
            node_id=con_id,
            ntype="constraint",
            label=c.name,
            attrs={"sense": c.sense, "expr_lhs": c.expr_lhs, "expr_rhs": c.expr_rhs},
        )

        lhs_id = f"expr:con:{c.name}:lhs"
        rhs_id = f"expr:con:{c.name}:rhs"
        add_node(
            node_id=lhs_id,
            ntype="expr",
            label=f"lhs({c.name})",
            attrs={
                "owner": con_id,
                "role": "constraint_lhs",
                "expr": c.expr_lhs or "",
                "iter_sets": _extract_iter_sets(c.expr_lhs or ""),
                "idents": _extract_idents(c.expr_lhs or ""),
            },
        )
        add_node(
            node_id=rhs_id,
            ntype="expr",
            label=f"rhs({c.name})",
            attrs={
                "owner": con_id,
                "role": "constraint_rhs",
                "expr": c.expr_rhs or "",
                "iter_sets": _extract_iter_sets(c.expr_rhs or ""),
                "idents": _extract_idents(c.expr_rhs or ""),
            },
        )
        add_edge(con_id, lhs_id, etype="con-has-lhs-expr")
        add_edge(con_id, rhs_id, etype="con-has-rhs-expr")

    # ---- 6) dependencies: var/param -> objective/constraint (兼容旧边) + var/param -> expr（新边） ----
    var_names = [v.name for v in ir.vars]
    param_names = [p.name for p in ir.params]

    # Objective
    obj_expr = obj.expr or ""
    for vname in var_names:
        if vname in obj_expr:
            add_edge(f"var:{vname}", obj_node_id, etype="var-in-objective")  # legacy
            add_edge(f"var:{vname}", obj_expr_id, etype="var-in-expr", attrs={"context": "objective"})
    for pname in param_names:
        if pname in obj_expr:
            add_edge(f"param:{pname}", obj_node_id, etype="param-in-objective")  # legacy
            add_edge(f"param:{pname}", obj_expr_id, etype="param-in-expr", attrs={"context": "objective"})

    # Constraints
    for c in ir.constraints:
        con_id = f"con:{c.name}"
        lhs_id = f"expr:con:{c.name}:lhs"
        rhs_id = f"expr:con:{c.name}:rhs"
        full_expr = f"{c.expr_lhs or ''} ; {c.expr_rhs or ''}"

        for vname in var_names:
            if vname in full_expr:
                add_edge(f"var:{vname}", con_id, etype="var-in-constraint")  # legacy
            if vname in (c.expr_lhs or ""):
                add_edge(f"var:{vname}", lhs_id, etype="var-in-expr", attrs={"context": "constraint", "side": "lhs"})
            if vname in (c.expr_rhs or ""):
                add_edge(f"var:{vname}", rhs_id, etype="var-in-expr", attrs={"context": "constraint", "side": "rhs"})

        for pname in param_names:
            if pname in full_expr:
                add_edge(f"param:{pname}", con_id, etype="param-in-constraint")  # legacy
            if pname in (c.expr_lhs or ""):
                add_edge(f"param:{pname}", lhs_id, etype="param-in-expr", attrs={"context": "constraint", "side": "lhs"})
            if pname in (c.expr_rhs or ""):
                add_edge(f"param:{pname}", rhs_id, etype="param-in-expr", attrs={"context": "constraint", "side": "rhs"})

    # ---- 7) stats + checks ----
    stats = _compute_graph_stats(nodes, edges)
    issues = _run_graph_self_checks(nodes, edges)

    graph_meta: Dict[str, Any] = {
        "builder": "build_graph_from_ir",
        "version": 2,  # bump
        "stats": stats,
        "checks": {"issues": [asdict(iss) for iss in issues]},
        "schema": {
            "node_types": sorted(list({n.type for n in nodes.values()})),
            "edge_types": sorted(list({e.etype for e in edges})),
        },
    }

    return GraphIR(nodes=nodes, edges=edges, meta=graph_meta)
