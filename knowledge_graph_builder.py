"""
Knowledge graph builder for PharmaDrishti regulatory-market intelligence.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import pandas as pd


def _drug_node_id(record: pd.Series) -> str:
    base = str(record.get("nppa_id", "UNK"))
    name = str(record.get("medicine_name", "")).strip().replace(" ", "_")[:50]
    return f"DRUG::{base}::{name}"


def _safe_text(value, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _add_base_nodes_and_edges(graph: nx.Graph, row: pd.Series, drug_id: str):
    matched_ingredient = row.get("matched_ingredient")
    extracted_ingredient = row.get("extracted_ingredient")
    ingredient = _safe_text(matched_ingredient, "")
    if not ingredient:
        ingredient = _safe_text(extracted_ingredient, "unknown")
    manufacturer = _safe_text(row.get("manufacturer"), "unknown")

    ingredient_id = f"ING::{ingredient}"
    manufacturer_id = f"MFR::{manufacturer}"

    graph.add_node(
        drug_id,
        node_type="Drug",
        medicine_name=row.get("medicine_name"),
        appl_no=row.get("appl_no"),
        nppa_year=row.get("nppa_year"),
        price_numeric=row.get("price_numeric"),
        approval_year=row.get("approval_year"),
        is_reference_drug=float(row.get("is_reference_drug", 0.0)),
        match_score=row.get("match_score"),
    )
    graph.add_node(ingredient_id, node_type="Ingredient", ingredient=ingredient)
    graph.add_node(manufacturer_id, node_type="Manufacturer", manufacturer=manufacturer)

    graph.add_edge(drug_id, ingredient_id, relation="contains")
    graph.add_edge(drug_id, manufacturer_id, relation="manufactured_by")


def _add_protection_nodes(
    graph: nx.Graph,
    drug_id: str,
    appl_no,
    patents_df: pd.DataFrame,
    exclusivity_df: pd.DataFrame,
):
    if pd.isna(appl_no):
        return

    matched_patents = patents_df[patents_df["appl_no"] == appl_no]
    for p_idx, patent in matched_patents.iterrows():
        patent_node = f"PAT::{appl_no}::{p_idx}"
        graph.add_node(
            patent_node,
            node_type="Patent",
            patent_no=patent.get("patent_no"),
            patent_expiry_year=patent.get("patent_expiry_year"),
        )
        graph.add_edge(drug_id, patent_node, relation="protected_by")

    matched_excl = exclusivity_df[exclusivity_df["appl_no"] == appl_no]
    for e_idx, excl in matched_excl.iterrows():
        excl_node = f"EXC::{appl_no}::{e_idx}"
        graph.add_node(
            excl_node,
            node_type="Exclusivity",
            exclusivity_code=excl.get("exclusivity_code"),
            exclusivity_expiry_year=excl.get("exclusivity_expiry_year"),
        )
        graph.add_edge(drug_id, excl_node, relation="protected_by")


def _attach_reference_flags(
    mapped_df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
) -> pd.DataFrame:
    prepared = mapped_df.copy()
    if reference_df is None or reference_df.empty:
        prepared["is_reference_drug"] = 0.0
        return prepared

    reference = reference_df.copy()
    reference["appl_no"] = pd.to_numeric(reference.get("appl_no"), errors="coerce")
    ref_appl_nos = set(reference["appl_no"].dropna().unique())
    prepared["is_reference_drug"] = prepared["appl_no"].isin(ref_appl_nos).astype(float)
    return prepared


def build_knowledge_graph(
    mapped_df: pd.DataFrame,
    patents_df: pd.DataFrame,
    exclusivity_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> nx.Graph:
    graph = nx.Graph()

    mapped_with_reference = _attach_reference_flags(mapped_df, reference_df)

    for _, row in mapped_with_reference.iterrows():
        drug_id = _drug_node_id(row)
        _add_base_nodes_and_edges(graph, row, drug_id)
        _add_protection_nodes(
            graph=graph,
            drug_id=drug_id,
            appl_no=row.get("appl_no"),
            patents_df=patents_df,
            exclusivity_df=exclusivity_df,
        )

    ingredient_to_drugs: dict[str, list[str]] = {}
    for node, data in graph.nodes(data=True):
        if data.get("node_type") != "Drug":
            continue
        for neighbor in graph.neighbors(node):
            n_data = graph.nodes[neighbor]
            if n_data.get("node_type") == "Ingredient":
                ingredient_to_drugs.setdefault(neighbor, []).append(node)

    for drug_nodes in ingredient_to_drugs.values():
        if len(drug_nodes) < 2:
            continue
        for left, right in combinations(drug_nodes, 2):
            graph.add_edge(left, right, relation="competes_with")

    return graph
