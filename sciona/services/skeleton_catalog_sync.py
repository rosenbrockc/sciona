"""Deterministic sync helpers for local skeleton-family artifact catalog rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from sciona.architect.skeleton_assets import (
    SkeletonFamilyAsset,
    load_local_skeleton_assets,
)
from sciona.architect.skeletons import infer_boundary_ports
from sciona.cdg_projection import (
    PublishedCDGProjection,
    build_published_cdg_projection,
)
from sciona.services.skeleton_artifacts import build_skeleton_asset_cdg

_ARTIFACT_NAMESPACE = uuid5(NAMESPACE_URL, "sciona/unified-artifacts")
_VERSION_NAMESPACE = uuid5(NAMESPACE_URL, "sciona/unified-artifact-versions")


@dataclass(frozen=True)
class SkeletonArtifactBundle:
    asset_id: str
    artifact: dict[str, Any]
    version: dict[str, Any]
    descriptions: list[dict[str, Any]] = field(default_factory=list)
    io_specs: list[dict[str, Any]] = field(default_factory=list)
    references_registry: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    audit_rollup: dict[str, Any] = field(default_factory=dict)
    cdg_nodes: list[dict[str, Any]] = field(default_factory=list)
    cdg_edges: list[dict[str, Any]] = field(default_factory=list)
    cdg_bindings: list[dict[str, Any]] = field(default_factory=list)
    projection: PublishedCDGProjection | None = None


def _artifact_fqdn(asset: SkeletonFamilyAsset) -> str:
    return f"cdg.skeleton.{asset.asset_id}"


def _artifact_id(asset: SkeletonFamilyAsset) -> str:
    return str(uuid5(_ARTIFACT_NAMESPACE, _artifact_fqdn(asset)))


def _content_hash(asset: SkeletonFamilyAsset) -> str:
    payload = asset.model_dump_json(by_alias=True, exclude_none=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _version_id(asset: SkeletonFamilyAsset) -> str:
    return str(uuid5(_VERSION_NAMESPACE, f"{_artifact_fqdn(asset)}@{_content_hash(asset)}"))


def _reference_type(reference: dict[str, Any]) -> str:
    url = str(reference.get("url", "") or "").strip().lower()
    if "doi.org" in url:
        return "paper"
    if url.startswith("http"):
        return "web"
    citation = str(reference.get("citation", "") or "").strip()
    if citation:
        return "book"
    return "web"


def _reference_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "reference"


def _reference_id(asset: SkeletonFamilyAsset, reference: dict[str, Any], ordinal: int) -> str:
    title = str(reference.get("title", "") or "").strip()
    return f"skeleton:{asset.asset_id}:{ordinal}:{_reference_slug(title)}"


def build_skeleton_artifact_bundle(asset: SkeletonFamilyAsset) -> SkeletonArtifactBundle:
    """Build the deterministic Supabase/Memgraph sync bundle for one asset."""
    fqdn = _artifact_fqdn(asset)
    artifact_id = _artifact_id(asset)
    version_id = _version_id(asset)
    content_hash = _content_hash(asset)
    cdg = build_skeleton_asset_cdg(asset)
    graph = asset.to_skeleton_graph()
    boundary_inputs, boundary_outputs = infer_boundary_ports(
        graph.template_nodes,
        graph.template_edges,
    )
    projection = build_published_cdg_projection(
        artifact={
            "artifact_id": artifact_id,
            "fqdn": fqdn,
            "artifact_kind": "cdg",
            "namespace_root": "sciona.architect.assets.skeletons",
            "namespace_path": asset.family,
        },
        version={
            "version_id": version_id,
            "semver": asset.asset_version,
            "content_hash": content_hash,
        },
        cdg=cdg,
    )
    review_status = str(asset.audit.review_status or "draft")
    is_publishable = review_status in {"approved", "transitional"}
    technical_description = str(asset.summary or "").strip()
    dejargonized_description = str(
        asset.dejargonized_summary or asset.audit.dejargonized_summary or asset.summary
    ).strip()

    references_registry: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for ordinal, reference in enumerate(asset.audit.references, start=1):
        reference_row = reference.model_dump(mode="json")
        ref_id = _reference_id(asset, reference_row, ordinal)
        ref_type = _reference_type(reference_row)
        title = str(reference_row.get("title", "") or "").strip()
        url = str(reference_row.get("url", "") or "").strip()
        note = str(reference_row.get("note", "") or "").strip()
        citation = str(reference_row.get("citation", "") or "").strip()
        registry_row = {
            "ref_id": ref_id,
            "ref_type": ref_type,
            "title": title,
            "authors": [],
            "year": None,
            "venue": "",
            "doi": None,
            "url": url,
            "bibtex_key": f"{asset.asset_id}_{ordinal}",
            "bibtex_raw": citation,
        }
        references_registry.append(registry_row)
        references.append(
            {
                "artifact_id": artifact_id,
                "ref_id": ref_id,
                "ref_key": registry_row["bibtex_key"],
                "doi": None,
                "title": title,
                "authors": [],
                "year": None,
                "url": url,
                "relevance_note": note or citation,
                "confidence": "medium" if review_status != "draft" else "low",
                "matched_nodes": [],
                "source": "manual",
                "verified": review_status != "draft",
            }
        )

    descriptions = [
        {
            "artifact_id": artifact_id,
            "kind": "technical",
            "content": technical_description,
            "language": "en",
            "generated_by": "local_skeleton_asset",
            "reviewed": review_status != "draft",
            "jargon_score": 0.85,
        },
        {
            "artifact_id": artifact_id,
            "kind": "dejargonized",
            "content": dejargonized_description,
            "language": "en",
            "generated_by": "local_skeleton_asset",
            "reviewed": True,
            "jargon_score": 0.2,
        },
        {
            "artifact_id": artifact_id,
            "kind": "conceptual_summary",
            "content": dejargonized_description,
            "language": "en",
            "generated_by": "local_skeleton_asset",
            "reviewed": True,
            "jargon_score": 0.2,
        },
    ]

    io_specs: list[dict[str, Any]] = []
    for ordinal, port in enumerate(boundary_inputs):
        io_specs.append(
            {
                "artifact_id": artifact_id,
                "version_id": version_id,
                "direction": "input",
                "name": port.name,
                "type_desc": port.type_desc,
                "constraints": port.constraints,
                "required": bool(port.required),
                "default_value_repr": port.default_value_repr,
                "ordinal": ordinal,
            }
        )
    for ordinal, port in enumerate(boundary_outputs):
        io_specs.append(
            {
                "artifact_id": artifact_id,
                "version_id": version_id,
                "direction": "output",
                "name": port.name,
                "type_desc": port.type_desc,
                "constraints": port.constraints,
                "required": bool(port.required),
                "default_value_repr": port.default_value_repr,
                "ordinal": ordinal,
            }
        )

    audit_rollup = {
        "artifact_id": artifact_id,
        "overall_verdict": "acceptable_with_limits"
        if review_status in {"approved", "transitional"}
        else "unknown",
        "structural_status": "documented",
        "runtime_status": "not_run",
        "semantic_status": "skeleton_asset",
        "developer_semantics_status": "reviewed"
        if review_status != "draft"
        else "draft",
        "risk_tier": "medium",
        "risk_score": 25 if review_status != "draft" else 45,
        "risk_dimensions": {},
        "risk_reasons": list(asset.audit.uncertainty_notes),
        "acceptability_score": 70 if review_status != "draft" else 40,
        "acceptability_band": (
            "acceptable_with_limits" if review_status != "draft" else "limited_acceptability"
        ),
        "parity_coverage_level": "not_applicable",
        "parity_test_status": "not_run",
        "parity_fixture_count": 0,
        "parity_case_count": 0,
        "review_status": review_status,
        "review_semantic_verdict": "documented",
        "review_developer_semantics_verdict": "documented",
        "review_limitations": list(asset.audit.provenance_notes),
        "review_required_actions": list(asset.audit.uncertainty_notes),
        "trust_readiness": "ready" if is_publishable else "not_ready",
        "trust_blockers": []
        if is_publishable
        else ["artifact_cdg_bindings pending", "leaf verification coverage pending"],
    }

    artifact = {
        "artifact_id": artifact_id,
        "artifact_kind": "cdg",
        "fqdn": fqdn,
        "owner_id": None,
        "source_repo_id": None,
        "namespace_root": "sciona.architect.assets.skeletons",
        "namespace_path": asset.family,
        "source_package": "sciona-matcher",
        "source_module_path": "",
        "source_symbol": asset.asset_id,
        "status": "approved" if is_publishable else "draft",
        "visibility_tier": "general",
        "description": technical_description,
        "source_kind": "hand_written",
        "stateful_kind": "none",
        "is_stochastic": False,
        "is_ffi": False,
        "is_publishable": is_publishable,
        "topo_hash": projection.topo_hash,
        "top_level_input_arity": len(boundary_inputs),
        "top_level_output_arity": len(boundary_outputs),
        "leaf_count": len(cdg.leaf_nodes()),
        "verified_leaf_coverage": 1.0 if is_publishable else 0.0,
    }
    version = {
        "version_id": version_id,
        "artifact_id": artifact_id,
        "content_hash": content_hash,
        "semver": asset.asset_version,
        "is_latest": True,
        "derives_from": None,
        "s3_key": f"skeleton-assets/{content_hash}.json",
        "fingerprint": content_hash,
    }
    cdg_nodes = [
        {
            "version_id": version_id,
            "node_id": str(node.get("node_id", "")),
            "parent_node_id": str(node.get("parent_id", "") or ""),
            "name": str(node.get("name", "") or ""),
            "description": str(node.get("description", "") or ""),
            "concept_type": str(node.get("concept_type", "") or ""),
            "status": str(node.get("status", "") or ""),
            "type_signature": str(node.get("type_signature", "") or ""),
            "matched_primitive": str(node.get("matched_primitive", "") or ""),
        }
        for node in projection.nodes
    ]
    cdg_edges = [
        {
            "version_id": version_id,
            "source_id": str(edge.get("source_id", "")),
            "target_id": str(edge.get("target_id", "")),
            "output_name": str(edge.get("output_name", "")),
            "input_name": str(edge.get("input_name", "")),
        }
        for edge in projection.edges
    ]

    return SkeletonArtifactBundle(
        asset_id=asset.asset_id,
        artifact=artifact,
        version=version,
        descriptions=descriptions,
        io_specs=io_specs,
        references_registry=references_registry,
        references=references,
        audit_rollup=audit_rollup,
        cdg_nodes=cdg_nodes,
        cdg_edges=cdg_edges,
        projection=projection,
    )


def load_skeleton_artifact_bundles() -> list[SkeletonArtifactBundle]:
    """Build deterministic sync bundles for all local skeleton-family assets."""
    bundles = [build_skeleton_artifact_bundle(asset) for asset in load_local_skeleton_assets()]
    bundles.sort(key=lambda bundle: bundle.artifact["fqdn"])
    return bundles


def sync_bundle_to_supabase(
    supabase: Any,
    bundle: SkeletonArtifactBundle,
) -> None:
    """Apply one skeleton artifact bundle to Supabase deterministically."""
    artifact_id = bundle.artifact["artifact_id"]
    version_id = bundle.version["version_id"]

    supabase.table("artifacts").upsert(bundle.artifact).execute()
    (
        supabase.table("artifact_versions")
        .update({"is_latest": False})
        .eq("artifact_id", artifact_id)
        .execute()
    )
    supabase.table("artifact_versions").upsert(bundle.version).execute()

    for table_name in (
        "artifact_descriptions",
        "artifact_io_specs",
        "artifact_references",
    ):
        supabase.table(table_name).delete().eq("artifact_id", artifact_id).execute()
    for table_name in ("artifact_cdg_nodes", "artifact_cdg_edges", "artifact_cdg_bindings"):
        supabase.table(table_name).delete().eq("version_id", version_id).execute()
    supabase.table("artifact_audit_rollups").delete().eq("artifact_id", artifact_id).execute()

    if bundle.references_registry:
        supabase.table("references_registry").upsert(bundle.references_registry).execute()
    if bundle.descriptions:
        supabase.table("artifact_descriptions").upsert(bundle.descriptions).execute()
    if bundle.io_specs:
        supabase.table("artifact_io_specs").upsert(bundle.io_specs).execute()
    if bundle.references:
        supabase.table("artifact_references").upsert(bundle.references).execute()
    if bundle.audit_rollup:
        supabase.table("artifact_audit_rollups").upsert(bundle.audit_rollup).execute()
    if bundle.cdg_nodes:
        supabase.table("artifact_cdg_nodes").upsert(bundle.cdg_nodes).execute()
    if bundle.cdg_edges:
        supabase.table("artifact_cdg_edges").upsert(bundle.cdg_edges).execute()
    if bundle.cdg_bindings:
        supabase.table("artifact_cdg_bindings").upsert(bundle.cdg_bindings).execute()


async def sync_bundles_to_graph_store(
    graph_store: Any,
    bundles: list[SkeletonArtifactBundle],
) -> list[tuple[str, dict[str, int]]]:
    """Apply the published CDG projection for each bundle to Memgraph."""
    await graph_store.ensure_constraints()
    results: list[tuple[str, dict[str, int]]] = []
    for bundle in bundles:
        if bundle.projection is None:
            continue
        counts = await graph_store.upsert_published_cdg(bundle.projection)
        results.append((bundle.asset_id, counts))
    return results
