#!/usr/bin/env python3
"""Validate sciona against a batch of Kaggle competitions.

For each competition:
1. Match the problem prompt against CDG templates via retrieval
2. Check grounding of the matched template
3. Compare proposed techniques against the winning solution's key_techniques
4. Rate as competitive/partial/divergent/inadequate

Usage:
    python scripts/validate_kaggle_batch.py \
        --corpus research/validation_corpus.json \
        --start 0 --end 50 \
        --output validation_results_0.json \
        --expansion-rounds 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sciona.sdk import load_catalog_from_repos, Sciona
from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec
from sciona.architect.dejargonizer import dejargonize_heuristic
from sciona.architect.solution_index import SolutionTemplateIndex
from sciona.principal.expansion_delta_planner import (
    DeltaAdaptationKind,
    DeltaPlan,
    DeltaPlanningQuery,
    plan_expansion_delta,
)


def match_prompt_to_templates(
    prompt: str,
    solution_index: SolutionTemplateIndex,
    k: int = 5,
) -> list[dict]:
    """Find CDG templates matching a competition prompt via dejargonized search."""
    dejargonized = dejargonize_heuristic(prompt)
    matches = solution_index.search_dejargonized(
        dejargonized_prompt=dejargonized,
        original_prompt=prompt,
        k=k,
    )
    return [
        {
            "template": tmpl.name,
            "overlap_score": score,
            "family": tmpl.family,
            "paradigm": tmpl.paradigm,
            "summary": (tmpl.dejargonized_summary or tmpl.summary)[:200],
        }
        for tmpl, score in matches
    ]


def evaluate_template_coverage(
    template: dict,
    bindings: dict | None,
    key_techniques: list[str],
    catalog,
) -> dict:
    """Evaluate how well a template covers the winning solution's techniques."""
    stages = template.get("stages", [])
    total_stages = len(stages)

    # Grounding from bindings
    bound_active = 0
    bound_approximate = 0
    orchestration = 0
    trivial = 0
    external = 0
    gap = 0

    if bindings:
        binding_map = {b["stage_id"]: b for b in bindings.get("bindings", [])}
        for stage in stages:
            b = binding_map.get(stage["stage_id"], {})
            status = b.get("status", "unassessed")
            action = b.get("action_class", "")
            if action == "orchestration":
                orchestration += 1
            elif action == "trivial_inline":
                trivial += 1
            elif action in ("external_knowledge", "external_tool"):
                external += 1
            elif status == "active":
                bound_active += 1
            elif status == "approximate":
                bound_approximate += 1
            else:
                gap += 1

    resolved = bound_active + bound_approximate + orchestration + trivial + external
    grounding_rate = resolved / total_stages if total_stages else 0.0

    # Technique coverage: check how many key_techniques appear in stage descriptions
    # Also include atom names and aliases from bindings
    stage_text_parts = []
    for s in stages:
        stage_text_parts.append(s.get("description", ""))
        stage_text_parts.append(s.get("name", ""))
        stage_text_parts.append(s.get("stage_id", ""))
    if bindings:
        for b in bindings.get("bindings", []):
            fqdn = b.get("bound_artifact_fqdn", "") or ""
            stage_text_parts.append(fqdn.replace(".", " ").replace("_", " "))
            rationale = b.get("evidence_summary", {}).get("binding_rationale", "")
            stage_text_parts.append(rationale)
    stage_text = " ".join(stage_text_parts).lower()

    # Also add atom descriptions from catalog for bound atoms
    if bindings:
        for b in bindings.get("bindings", []):
            fqdn = b.get("bound_artifact_fqdn") or ""
            atom_name = fqdn.split(".")[-1] if fqdn else ""
            prim = catalog.get(atom_name) if atom_name else None
            if prim:
                stage_text += " " + prim.description.lower()
                stage_text += " " + prim.name.replace("_", " ")
                for alias in prim.aliases:
                    stage_text += " " + alias.replace("_", " ")

    covered = []
    missing = []
    for tech in key_techniques:
        # Tokenize technique — split on spaces, parens, commas, hyphens
        import re
        tech_words = set(re.split(r"[\s,\(\)\-/]+", tech.lower()))
        # Remove very common words and short tokens
        tech_words -= {"the", "a", "an", "of", "in", "for", "and", "or", "to",
                       "is", "with", "using", "based", "on", "via", "e.g.",
                       "from", "each", "per", "at", "by", ""}
        tech_words = {w for w in tech_words if len(w) >= 3}
        if not tech_words:
            covered.append(tech)
            continue
        # Count how many technique words appear in the stage text
        matches = sum(1 for w in tech_words if w in stage_text)
        # More lenient: 40% word overlap OR any 2+ word match
        if matches >= max(len(tech_words) * 0.4, 1) and matches >= 2:
            covered.append(tech)
        elif matches >= 1 and len(tech_words) <= 2:
            covered.append(tech)
        else:
            missing.append(tech)

    technique_coverage = len(covered) / len(key_techniques) if key_techniques else 0.0

    return {
        "total_stages": total_stages,
        "bound_active": bound_active,
        "bound_approximate": bound_approximate,
        "orchestration": orchestration,
        "trivial_inline": trivial,
        "external": external,
        "gap": gap,
        "grounding_rate": round(grounding_rate, 3),
        "technique_coverage": round(technique_coverage, 3),
        "covered_techniques": covered,
        "missing_techniques": missing,
    }


def assess_result(
    template_match: dict | None,
    evaluation: dict | None,
) -> str:
    """Rate the overall result."""
    if template_match is None:
        return "no_template"

    if evaluation is None:
        return "no_evaluation"

    tc = evaluation.get("technique_coverage", 0)
    gr = evaluation.get("grounding_rate", 0)

    if tc >= 0.8 and gr >= 0.7:
        return "competitive"
    elif tc >= 0.5:
        return "partial"
    elif gr >= 0.7:
        return "divergent"
    else:
        return "inadequate"


_ASSESSMENT_RANK = {
    "no_template": 0,
    "no_evaluation": 0,
    "inadequate": 1,
    "divergent": 2,
    "partial": 3,
    "competitive": 4,
}


def _stage_names(template: dict) -> list[str]:
    return [
        str(value)
        for stage in template.get("stages", [])
        for value in (
            stage.get("stage_id", ""),
            stage.get("name", ""),
            stage.get("description", ""),
            stage.get("dejargonized_description", ""),
            stage.get("concept_type", ""),
            stage.get("matched_primitive", ""),
        )
        if value
    ]


def _port_names(template: dict, key: str) -> list[str]:
    names: list[str] = []
    for port in template.get(key, []):
        if port.get("name"):
            names.append(str(port["name"]))
    for stage in template.get("stages", []):
        for port in stage.get(key, []):
            if port.get("name"):
                names.append(str(port["name"]))
    return names


def _delta_plan_summary(plan: DeltaPlan) -> dict:
    selected = plan.selected
    operations = selected.operation_rule_names
    sequence = selected.operation_sequence
    return {
        "decision": plan.decision.value,
        "should_compose_novel": plan.should_compose_novel,
        "base_coverage": plan.base_coverage,
        "projected_coverage": selected.projected_coverage,
        "intrusion_cost": selected.intrusion_cost,
        "utility_score": selected.utility_score,
        "covered_terms": list(selected.covered_terms),
        "missing_terms_after_plan": list(selected.missing_terms_after_plan),
        "operation_rule_names": list(operations),
        "operation_path": list(selected.path),
        "asset_family": sequence.asset_family if sequence else "",
        "asset_id": sequence.asset_id if sequence else "",
        "candidate_count": plan.candidate_count,
        "rationale": selected.rationale,
    }


def evaluate_counterfactual_expansion(
    *,
    template_match: dict,
    template: dict,
    base_evaluation: dict,
    key_techniques: list[str],
    max_rounds: int,
) -> tuple[dict, dict]:
    """Project adapted coverage after a few expansion/refinement rounds."""
    if max_rounds <= 0 or not key_techniques:
        return dict(base_evaluation), {}

    covered = tuple(base_evaluation.get("covered_techniques", []))
    missing = tuple(base_evaluation.get("missing_techniques", []))
    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=tuple(
                value
                for value in (
                    template_match.get("family", ""),
                    template.get("family", ""),
                    template.get("paradigm", ""),
                )
                if value
            ),
            matched_techniques=covered,
            missing_techniques=missing,
            stage_names=tuple(_stage_names(template)),
            input_names=tuple(_port_names(template, "inputs")),
            output_names=tuple(_port_names(template, "outputs")),
            runtime_keys=tuple(_port_names(template, "inputs")),
            intermediate_keys=tuple(_stage_names(template)),
            base_coverage=float(base_evaluation.get("technique_coverage", 0.0)),
            min_adapted_coverage=0.50,
            max_sequences=5,
            max_operations_per_sequence=max(1, max_rounds),
        )
    )

    adapted = dict(base_evaluation)
    if plan.decision not in (DeltaAdaptationKind.DIRECT_USE, DeltaAdaptationKind.TRUE_NOVEL):
        adapted_covered = list(dict.fromkeys([*covered, *plan.selected.covered_terms]))
        adapted["technique_coverage"] = plan.selected.projected_coverage
        adapted["covered_techniques"] = adapted_covered
        adapted["missing_techniques"] = list(plan.selected.missing_terms_after_plan)
        adapted["coverage_source"] = (
            str(base_evaluation.get("coverage_source", "unknown"))
            + "_plus_delta_counterfactual"
        )

    adapted["counterfactual_expansion"] = {
        "rounds_allowed": max_rounds,
        **_delta_plan_summary(plan),
    }
    return adapted, _delta_plan_summary(plan)


def is_rescued_by_expansion(
    base_assessment: str,
    adapted_assessment: str,
    delta_plan: dict,
) -> bool:
    """Return true when expansion/refinement improves validation status."""
    if not delta_plan.get("operation_rule_names"):
        return False
    return _ASSESSMENT_RANK.get(adapted_assessment, 0) > _ASSESSMENT_RANK.get(base_assessment, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cdg-dir", default=str(
        Path.home() / "personal/sciona-atoms/data/solution_cdgs"
    ))
    parser.add_argument("--rerank", action="store_true",
                        help="Use LLM to rerank top-N template candidates")
    parser.add_argument("--llm-provider", default=None,
                        help="LLM provider for reranking (e.g., claude_shim, anthropic)")
    parser.add_argument(
        "--expansion-rounds",
        type=int,
        default=0,
        help=(
            "Counterfactually grade base-plus-delta coverage after this many "
            "expansion/refinement operations. Default keeps base-only scoring."
        ),
    )
    args = parser.parse_args()

    # Load corpus
    corpus = json.loads(Path(args.corpus).read_text())
    competitions = corpus["competitions"][args.start:args.end]
    print(f"Processing {len(competitions)} competitions ({args.start}-{args.end})")

    # Load catalog
    repos = [
        Path.home() / "personal" / r
        for r in [
            "sciona-atoms", "sciona-atoms-ml", "sciona-atoms-dl",
            "sciona-atoms-bio", "sciona-atoms-physics", "sciona-atoms-signal",
            "sciona-atoms-cs", "sciona-atoms-geo", "sciona-atoms-fintech",
            "sciona-atoms-robotics",
        ]
        if (Path.home() / "personal" / r).exists()
    ]
    catalog = load_catalog_from_repos(repos)
    print(f"Catalog: {catalog.size} atoms")

    # Load solution template index
    cdg_dir = Path(args.cdg_dir)
    solution_index = SolutionTemplateIndex.from_directory(cdg_dir)
    print(f"Templates: {solution_index.size}")

    # Optional: set up LLM reranker
    llm = None
    if args.rerank:
        try:
            from sciona.config import AgeomConfig
            from sciona.commands.llm_helpers import _create_llm

            config = AgeomConfig()
            if args.llm_provider:
                config.llm_provider = args.llm_provider
            llm_args = argparse.Namespace(
                mode=None,
                llm_provider=config.llm_provider,
                llm_model=config.llm_model,
                llm_max_tokens=config.llm_max_tokens,
            )
            llm = _create_llm(llm_args, config, "architect")
            print(f"LLM reranking enabled: {config.llm_provider}/{config.llm_model}")
        except Exception as e:
            print(f"WARNING: LLM reranking requested but failed to create client: {e}")
            print("Falling back to keyword-only matching")

    # Process each competition
    import asyncio
    results = []

    async def _process_competition(comp):
        cid = comp["competition_id"]
        prompt = comp["prompt"]
        key_techniques = comp.get("key_techniques", [])
        solution_summary = comp.get("solution_summary", "")

        # Phase 1: Match prompt to templates (dejargonized keyword search)
        matches = match_prompt_to_templates(prompt, solution_index, k=10)

        # Phase 2: Optional LLM reranking of top candidates
        rerank_output = None
        if llm and matches:
            try:
                from sciona.architect.template_reranker import rerank_templates
                top_candidates = [
                    solution_index.get(m["template"])
                    for m in matches[:5]
                    if solution_index.get(m["template"]) is not None
                ]
                if top_candidates:
                    rerank_output = await rerank_templates(
                        prompt, top_candidates, llm,
                        max_candidates=5,
                        key_techniques=key_techniques or None,
                    )
                    # Reorder matches based on LLM ranking
                    if rerank_output.best_match and rerank_output.best_match != "none":
                        # Move best match to front
                        reranked = []
                        for r in rerank_output.rankings:
                            m = next(
                                (m for m in matches if m["template"] == r.template_name),
                                None,
                            )
                            if m:
                                m = dict(m)
                                m["llm_score"] = r.score
                                m["llm_reasoning"] = r.reasoning
                                reranked.append(m)
                        # Keep any that weren't in the reranked set
                        reranked_names = {m["template"] for m in reranked}
                        for m in matches:
                            if m["template"] not in reranked_names:
                                reranked.append(m)
                        matches = reranked
            except Exception as e:
                logger.debug("Reranking failed for %s: %s", cid, e)

        # Evaluate top match
        evaluation = None
        if matches:
            top_name = matches[0]["template"]
            tmpl = solution_index.get(top_name)
            if tmpl:
                bindings_path = cdg_dir / f"{top_name}_bindings.json"
                bindings = None
                if bindings_path.exists():
                    bindings = json.loads(bindings_path.read_text())

                # Use LLM technique coverage if available, else keyword
                if rerank_output and rerank_output.technique_coverage:
                    covered = [
                        tc.technique for tc in rerank_output.technique_coverage
                        if tc.covered
                    ]
                    missing = [
                        tc.technique for tc in rerank_output.technique_coverage
                        if not tc.covered
                    ]
                    tc_rate = len(covered) / len(key_techniques) if key_techniques else 0.0
                    evaluation = evaluate_template_coverage(
                        tmpl.raw_cdg, bindings, key_techniques, catalog
                    )
                    # Override technique coverage with LLM-evaluated values
                    evaluation["technique_coverage"] = round(tc_rate, 3)
                    evaluation["covered_techniques"] = covered
                    evaluation["missing_techniques"] = missing
                    evaluation["coverage_source"] = "llm_semantic"
                else:
                    evaluation = evaluate_template_coverage(
                        tmpl.raw_cdg, bindings, key_techniques, catalog
                    )
                    evaluation["coverage_source"] = "keyword_heuristic"

        base_assessment = assess_result(
            matches[0] if matches else None,
            evaluation,
        )
        assessment = base_assessment
        base_evaluation = dict(evaluation) if evaluation else None
        adapted_evaluation = None
        delta_plan = {}
        rescued = False
        if args.expansion_rounds > 0 and matches and evaluation:
            top_name = matches[0]["template"]
            tmpl = solution_index.get(top_name)
            if tmpl:
                adapted_evaluation, delta_plan = evaluate_counterfactual_expansion(
                    template_match=matches[0],
                    template=tmpl.raw_cdg,
                    base_evaluation=evaluation,
                    key_techniques=key_techniques,
                    max_rounds=args.expansion_rounds,
                )
                assessment = assess_result(matches[0], adapted_evaluation)
                rescued = is_rescued_by_expansion(
                    base_assessment,
                    assessment,
                    delta_plan,
                )

        result = {
            "competition_id": cid,
            "title": comp.get("title", ""),
            "domain": comp.get("domain", ""),
            "problem_type": comp.get("problem_type", ""),
            "assessment": assessment,
            "base_assessment": base_assessment,
            "template_matches": matches[:3],
            "evaluation": adapted_evaluation or evaluation,
            "base_evaluation": base_evaluation,
            "adapted_evaluation": adapted_evaluation,
            "delta_plan": delta_plan or None,
            "rescued_by_expansion": rescued,
            "rescued_by_operations": delta_plan.get("operation_rule_names", []) if rescued else [],
            "key_techniques_count": len(key_techniques),
            "solution_summary_preview": solution_summary[:200],
        }
        if rerank_output:
            result["rerank"] = {
                "best_match": rerank_output.best_match,
                "should_compose_novel": rerank_output.should_compose_novel,
                "novel_reasoning": rerank_output.novel_reasoning,
            }
        results.append(result)

        status = "✓" if assessment == "competitive" else \
                 "~" if assessment == "partial" else \
                 "!" if assessment == "divergent" else "✗"
        final_evaluation = adapted_evaluation or evaluation
        base_tc = base_evaluation["technique_coverage"] if base_evaluation else 0
        final_tc = final_evaluation["technique_coverage"] if final_evaluation else 0
        final_gr = final_evaluation["grounding_rate"] if final_evaluation else 0
        delta_suffix = ""
        if delta_plan:
            op_count = len(delta_plan.get("operation_rule_names", []))
            delta_suffix = (
                f", base={base_assessment}/{base_tc:.0%},"
                f" delta={delta_plan.get('decision', 'none')}:{op_count}"
            )
        print(f"  {status} {cid}: {assessment}"
              f" (tc={final_tc:.0%}, gr={final_gr:.0%}{delta_suffix})")

    async def _run_all():
        for comp in competitions:
            await _process_competition(comp)

    asyncio.run(_run_all())

    # Write results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    # Summary
    from collections import Counter
    assessments = Counter(r["assessment"] for r in results)
    print(f"\n=== SUMMARY ({len(results)} competitions) ===")
    for a, c in assessments.most_common():
        print(f"  {a}: {c} ({100*c/len(results):.0f}%)")
    rescued = [r for r in results if r.get("rescued_by_expansion")]
    if args.expansion_rounds > 0:
        print(f"  rescued_by_expansion: {len(rescued)} ({100*len(rescued)/len(results):.0f}%)")
        rescue_ops = Counter(
            op for result in rescued for op in result.get("rescued_by_operations", [])
        )
        for op, count in rescue_ops.most_common(10):
            print(f"    {op}: {count}")


if __name__ == "__main__":
    main()
