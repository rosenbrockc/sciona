#!/usr/bin/env python3
"""Atomically intake and approve the reviewed corrected escape_speed realization at Tier 3."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID,uuid5,NAMESPACE_URL
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms,AtomSeedRow
from sciona.physics_ingest.escape_speed_execution import build_escape_speed_execution,PRIMITIVE,SOURCE_VERSION,SOURCE_HASH
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.cdg_projection import build_published_cdg_projection
from scripts.import_residual_execution_drafts import ensure_row
from scripts.validate_escape_speed_execution import validate

REFERENCE='openstax-escape-zero-energy'
URL='https://openstax.org/books/university-physics-volume-1/pages/13-3-gravitational-potential-energy-and-total-energy'
LIMITATIONS=[
    'Automated Tier3; no human-reviewed Tier1 designation.',
    'Positive G, test mass, central mass and radius; fixed exterior spherical Newtonian field, negligible test-mass backreaction.',
    'Escape threshold magnitude reaches asymptotic rest at infinity; no drag, rotation, relativity, finite-time arrival or trajectory-clearance guarantee.',
    'Positive required launch energy equals external work against gravity; outward gravitational work has opposite sign.',
    'Negative inward_counterpart_velocity is time-reversed infall, NOT outward escape velocity.',
    'Reuses unchanged approved infall provider with positional output aliases; provider audits and ports remain unchanged.',
    'Corrected antiderivative, kinetic-energy aliases, infinity, mass cancellation and signed-root semantics; no literal source AST parity.',
]


def provider_snapshot(db, atom_id):
    result={}
    for table,key in [('artifacts','artifact_id'),('artifact_versions','artifact_id'),('artifact_io_specs','artifact_id'),
                      ('artifact_audit_rollups','artifact_id'),('artifact_audit_evidence','artifact_id'),('artifact_validity_bounds','artifact_id'),
                      ('atoms','atom_id'),('atom_versions','atom_id'),('atom_io_specs','atom_id'),('atom_audit_rollups','atom_id')]:
        rows=db.execute('SELECT to_jsonb(t) AS row FROM '+table+' t WHERE '+key+'=%s',(atom_id,)).fetchall()
        result[table]=sorted([r['row'] for r in rows],key=lambda v:json.dumps(v,sort_keys=True))
    return _digest(result)



async def promote(root,symbol_file,rule_file,expression_file,apply=False):
    report=await validate(root,symbol_file,rule_file,expression_file)
    retained=json.loads((root/'docs/reviews/escape_speed_execution.json').read_text())
    if report!=retained:raise ValueError('Retained execution evidence differs from fresh validation')
    tests=json.loads((root/'docs/reviews/escape_speed_test_review.json').read_text())
    if tests['test_results']!=dict(tests=48,failures=0,errors=0,skipped=0):raise ValueError('Complete tests required')
    for path,sha in tests['test_source_sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=sha:raise ValueError('Test source changed')
    directory=root.parent/'sciona-atoms-physics'
    specs=[s for s in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-physics',directory),artifact_root=directory/'src/sciona/atoms') if s.import_module+'.'+s.source_symbol==PRIMITIVE]
    if len(specs)!=1:raise ValueError('Unique provider identity required')
    spec=specs[0]
    if hashlib.sha256(spec.file_path.read_bytes()).hexdigest()!=report['provider_sha256'] or tests['provider_sha256']!=report['provider_sha256']:raise ValueError('Provider source differs')
    graph=build_escape_speed_execution();digest,nodes,edges=encode_execution_graph(graph)
    if report['graph_digest']!=digest or report['full_runner_cases']!=6 or report['synthetic_states']!=29 or report['maximum_ulp_error']!=0:raise ValueError('Execution gates differ')
    evidence_hashes={p:hashlib.sha256((root/'docs/reviews'/p).read_bytes()).hexdigest() for p in ['escape_speed_execution.json','escape_speed_test_review.json','escape_speed_correction_review.md']}
    atom_id=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)
    rollup=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass',developer_semantics_status='pass',review_status='approved',review_semantic_verdict='pass',review_developer_semantics_verdict='pass',trust_readiness='ready',review_limitations=LIMITATIONS,review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits',parity_coverage_level='positive_and_negative',parity_test_status='pass')
    created=0
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('escape_speed-corrected-community.v1'))")
        source=db.execute('SELECT a.artifact_id,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s FOR SHARE OF a,v',(SOURCE_VERSION,)).fetchone()
        if not source or source['content_hash']!=SOURCE_HASH:raise ValueError('Exact source version missing')
        graph_id=uuid5(source['artifact_id'],'escape_speed-corrected-execution.v1');version_id=uuid5(graph_id,digest);fqdn=source['fqdn']+'.escape_speed_corrected_execution'
        if str(spec.version_id)!='23b6922d-d058-5dbc-aee2-2276694c7cdd' or spec.content_hash!='485f1797fab10da7d52f2ef79af046281f01db9abadff2070b520363cb5bcd04':
            raise ValueError('Exact reused provider pin differs')
        provider=db.execute('SELECT a.status,a.is_publishable,v.trust_tier,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=%s AND v.version_id=%s FOR SHARE OF a,v',(atom_id,spec.version_id)).fetchone()
        if provider!=dict(status='approved',is_publishable=True,trust_tier=3,is_latest=True):raise ValueError('Reused provider not approved')
        before=provider_snapshot(db,atom_id)
        description='Compute ideal escape speed and required launch energy using the approved infall provider, retaining an explicitly labeled inward counterpart. Automated Tier3.'
        existing_graph=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(graph_id,)).fetchone()
        if existing_graph and existing_graph!=dict(status='approved',is_publishable=True):raise ValueError('Unexpected existing graph state')
        created+=ensure_row(db,'artifacts',{'artifact_id':graph_id},dict(artifact_id=graph_id,artifact_kind='cdg',fqdn=fqdn,description=description,**(existing_graph or dict(status='draft',is_publishable=False))))
        created+=ensure_row(db,'artifact_versions',{'version_id':version_id},dict(version_id=version_id,artifact_id=graph_id,content_hash=digest,semver='0.0.0+execution.'+digest[:12],is_latest=bool(existing_graph),trust_tier=3))
        for node in nodes:
            created+=ensure_row(db,'artifact_cdg_nodes',{'version_id':version_id,'node_id':node['node_id']},{'version_id':version_id,**{k:node[k] for k in ['node_id','parent_node_id','name','description','concept_type','status','matched_primitive','type_signature']}})
        for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
            for ordinal,port in enumerate(ports):
                common=dict(direction=direction,name=port.name,ordinal=ordinal,type_desc=port.type_desc,constraints=port.constraints,required=port.required,default_value_repr=port.default_value_repr)
                for table,key,identity,selected in [('artifact_io_specs','artifact_id',graph_id,version_id)]:
                    row={key:identity,'version_id':selected,**common}
                    if table=='artifact_io_specs':row['dim_signature']=port.dim_signature
                    created+=ensure_row(db,table,{k:row[k] for k in [key,'version_id','direction','name']},row)
        created+=ensure_row(db,'artifact_cdg_bindings',{'version_id':version_id,'node_id':'escape'},dict(version_id=version_id,node_id='escape',bound_artifact_fqdn=spec.fqdn,bound_version_content_hash=spec.content_hash,binding_confidence=1.,binding_source='escape_speed-corrected-community.v1',status='active',alternatives=Jsonb([]),evidence_summary=Jsonb(dict(runtime_fqdn=PRIMITIVE,provider_version_id=str(spec.version_id),output_aliases_by_ordinal=['escape_speed','inward_counterpart_velocity','required_launch_energy','potential_energy']))))
        dependency=dict(dependent_version_id=version_id,dependency_artifact_fqdn=source['fqdn'],dependency_content_hash=SOURCE_HASH,port_name='')
        created+=ensure_row(db,'artifact_dependencies',dependency,{**dependency,'dependency_role':'cdg','optional':False,'binding_metadata':Jsonb(dict(scope='Mandatory original-source provenance; external-work convention, antiderivative, kinetic-energy aliases, mass cancellation and positive speed reconstructed. Original draft unchanged.'))})
        title='OpenStax University Physics Volume1: Gravitational Potential Energy and Total Energy'
        created+=ensure_row(db,'references_registry',{'ref_id':REFERENCE},dict(ref_id=REFERENCE,ref_type='web',title=title,url=URL))
        for identity,selected,kind in [(graph_id,version_id,'execution')]:
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(selected,)).fetchone():raise ValueError('Unresolved failed audit of publication target')
            tables=[('artifact_references','artifact_id'),('atom_references','atom_id')] if kind=='provider' else [('artifact_references','artifact_id')]
            for table,key in tables:
                created+=ensure_row(db,table,{key:identity,'ref_key':REFERENCE},{key:identity,'ref_id':REFERENCE,'ref_key':REFERENCE,'title':title,'url':URL,'source':'llm_extracted','verified':True,'confidence':'high','relevance_note':'Zero-at-infinity gravitational potential and ideal zero-energy escape/infall speed magnitude.'})
            for table,key in ([('artifact_audit_rollups','artifact_id'),('atom_audit_rollups','atom_id')] if kind=='provider' else [('artifact_audit_rollups','artifact_id')]):
                created+=ensure_row(db,table,{key:identity},{key:identity,**rollup})
            evidence_id=uuid5(selected,'escape_speed-corrected-community.v1')
            created+=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},dict(evidence_id=evidence_id,artifact_id=identity,version_id=selected,audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version='escape_speed-corrected-community.v1',details=Jsonb(dict(kind=kind,publication_tier=3,review_source='automated',evidence_sha256=evidence_hashes,provider_version_id=str(spec.version_id),provider_content_hash=spec.content_hash,execution_graph_sha256=digest,source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,corrected_proof=report['source_proof'],source_parity_claim=False,reused_provider_snapshot_sha256=before,limitations=LIMITATIONS))))
            bound_id=uuid5(selected,'escape-speed-fixed-central-domain.v1')
            created+=ensure_row(db,'artifact_validity_bounds',{'bound_id':bound_id},dict(bound_id=bound_id,artifact_id=identity,version_id=selected,scope='version',bound_kind='regime',validity_statement='Positive G, test mass, fixed central mass and radius; negligible test-mass backreaction in exterior spherical field. Positive escape threshold and required launch energy; negative velocity output is inward counterpart only. Asymptotic rest at infinity. No finite two-body, drag or relativistic claim.',evidence_ref_key=REFERENCE,review_status='automated_pass',metadata=Jsonb(dict(scope='Automated reviewed domain; caller establishes physical applicability.'))))
        for selected in [spec.version_id,version_id]:
            if db.execute('SELECT count(*) AS n FROM artifact_io_specs WHERE version_id=%s',(selected,)).fetchone()['n']!=8:
                raise ValueError('Unexpected canonical ports')
        if db.execute('SELECT count(*) AS n FROM atom_io_specs WHERE version_id=%s',(spec.version_id,)).fetchone()['n']!=8:
            raise ValueError('Unexpected legacy ports')
        if db.execute('SELECT count(*) AS n FROM artifact_cdg_bindings WHERE version_id=%s',(version_id,)).fetchone()['n']!=1:
            raise ValueError('Unexpected graph bindings')
        if db.execute('SELECT count(*) AS n FROM artifact_dependencies WHERE dependent_version_id=%s',(version_id,)).fetchone()['n']!=1:
            raise ValueError('Unexpected graph dependencies')
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(spec.version_id,)).fetchone():
            raise ValueError('Unexpected provider dependencies')
        doc=db.execute('SELECT get_artifact_document(%s) AS d',(fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(doc,version_id=str(version_id),content_hash=digest,require_execution_envelope=True)!=graph:raise ValueError('Catalog graph differs')
        topo=build_published_cdg_projection(artifact={'artifact_id':str(graph_id),'fqdn':fqdn},version={'version_id':str(version_id),'content_hash':digest},cdg=graph).topo_hash
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=1,top_level_input_arity=4,top_level_output_arity=4,topo_hash=%s WHERE artifact_id=%s",(topo,graph_id))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s',(version_id,graph_id))
        if not db.execute('SELECT 1 FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.atom_id=%s AND v.version_id=%s AND v.is_latest',(atom_id,spec.version_id)).fetchone():raise ValueError('Provider not served')
        if not db.execute('SELECT 1 FROM catalog_artifacts_served a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=%s AND v.version_id=%s AND v.is_latest',(graph_id,version_id)).fetchone():raise ValueError('Graph not served')
        if provider_snapshot(db,atom_id)!=before:raise ValueError('Reused provider records changed')
        if not apply:db.rollback()
    return dict(applied=apply,rows_created=created,trust_tier=3,served_provider_versions=1,served_execution_cdgs=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(promote(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file,args.apply))))
