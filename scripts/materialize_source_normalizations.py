#!/usr/bin/env python3
"""Create non-latest, source-AST normalization alternatives; preserve every original parse."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
from sciona.physics_ingest.source_normalization import normalize_pinned_source_ast
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--symbol-file',required=True,type=Path);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();symbol_bytes=args.symbol_file.read_bytes();counts=Counter();root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('pdg-source-normalization.v1'))")
        rows=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,v.content_hash AS previous_content_hash FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id JOIN artifacts a ON a.artifact_id=e.artifact_id WHERE v.is_latest AND a.status='draft' AND NOT a.is_publishable AND e.review_status NOT IN ('human_reviewed','blocked') AND e.evidence_json->'pdg_source_comparison'->'source_comparison'->>'correspondence'='different_ast' AND e.evidence_json->'pdg_source_comparison'->'source_comparison'->>'dimension_status'='checker_passed' ORDER BY e.expression_id FOR SHARE OF e,q,s,v,a").fetchall()
        for row in rows:
            fresh=prepare_pdg_evidence(row,symbol_bytes)
            if fresh!=row['evidence_json']['pdg_source_comparison']:raise ValueError('source evidence drift')
            definitions=load_pinned_pdg_scalars(symbol_bytes,fresh['symbol_file_sha256'])
            normalized,source_symbols=normalize_pinned_source_ast(fresh['upstream_symbolic']['sympy_srepr'],definitions)
            if normalized.srepr_str==row['sympy_srepr']:continue
            digest=_digest({'kind':'pdg-source-normalization.v1','previous_content_hash':row['previous_content_hash'],'source_ast':fresh['upstream_symbolic']['sympy_srepr'],'normalized_srepr':normalized.srepr_str,'dimensional_hash':normalized.dimensional_hash,'candidate_payload_sha256':fresh['candidate_payload_sha256']})
            version_id=uuid5(UUID(str(row['artifact_id'])),'pdg-source-normalization.v1:'+digest);expression_id=uuid5(version_id,'primary')
            created=ensure_row(db,'artifact_versions',{'version_id':version_id},{'version_id':version_id,'artifact_id':row['artifact_id'],'content_hash':digest,'semver':'0.0.0+source-ast.'+digest[:12],'derives_from':row['version_id'],'is_latest':False,'trust_tier':3})
            proposal={**row,'version_id':version_id,'expression_id':expression_id,'sympy_srepr':normalized.srepr_str}
            comparison=prepare_pdg_evidence(proposal,symbol_bytes)
            if comparison['source_comparison']['correspondence']!='exact_ast_match' or comparison['source_comparison']['dimension_status']!='checker_passed':raise ValueError('new source correspondence failed')
            provenance={'runner_version':'pdg-source-normalization.v1','selected_representation':'pinned upstream SymPy AST with source-defined scalar labels','scope':'alternative source representation, not a claim that the original LaTeX parse was corrected or that physical behavior is validated','original_latex_agreement':'not_established','previous_expression_id':str(row['expression_id']),'previous_version_id':str(row['version_id']),'previous_content_hash':row['previous_content_hash'],'previous_evidence_sha256':_digest(row['evidence_json']),'source_symbol_mapping':source_symbols,'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['sciona/physics_ingest/source_normalization.py','sciona/ghost/symbolic_normalization.py']}}
            dimensional={'runner_version':'pdg-source-normalization.v1','status':'passed','dimensional_hash':normalized.dimensional_hash,'source_comparison_sha256':_digest(comparison),'scope':'source-backed SI dimensional consistency, not physical validity'}
            expression={'expression_id':expression_id,'artifact_id':row['artifact_id'],'version_id':version_id,'candidate_id':row['candidate_id'],'expression_kind':row['expression_kind'],'expression_role':row['expression_role'],'sympy_srepr':normalized.srepr_str,'canonical_expr_hash':normalized.expression_hash,'topology_hash':normalized.topology_hash,'dimensional_hash':normalized.dimensional_hash,'raw_formula':normalized.srepr_str,'raw_formula_format':'sympy','source_expression_id':row['source_expression_id'],'parse_status':'normalized','parse_confidence':0.95,'review_status':'unreviewed','validation_status':'unknown','evidence_json':Jsonb({'source_normalization':provenance,'parse_roundtrip':{'status':'passed'},'pdg_source_comparison':comparison,'dimensional_analysis':dimensional})}
            ensure_row(db,'artifact_symbolic_expressions',{'expression_id':expression_id},expression)
            for ordinal,(name,variable) in enumerate(sorted(normalized.variables.items())):
                variable_id=uuid5(expression_id,name)
                counts['variable_rows_created']+=ensure_row(db,'artifact_symbolic_variables',{'variable_id':variable_id},{'variable_id':variable_id,'expression_id':expression_id,'symbol_name':name,'source_symbol':source_symbols[name],'variable_role':variable.role,'dim_signature':variable.dim_signature.to_compact(),'dimension_source':'source','ordinal':ordinal,'evidence_json':Jsonb(dimensional)})
            audit_id=uuid5(version_id,'source-ast-intake')
            ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':row['artifact_id'],'version_id':version_id,'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'pdg-source-normalization.v1','details':Jsonb({'scope':'source AST, scalar mapping, roundtrip and dimensions only; existing bindings and latest version unchanged','content_hash':digest,'provenance':provenance})})
            counts['versions_created' if created else 'unchanged_versions']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
