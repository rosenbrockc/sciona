"""Synthetic checks that source reconciliation preserves evidence distinctions."""
from scripts.audit_physics_legacy_reconciliation import source_inventory


def signature(ids, projected=False, rule='rule'):
    value=dict(inference_rule_id=rule,assumptions=[],dimensions={},
               variable_bindings={'derivation_id':'synthetic-derivation','step_id':'step'},
               relationship_kind='derives_from',source_pdg_inference_id=ids[0])
    if projected:
        value.update(source_pdg_step_id='step',source_pdg_inference_ids=ids)
    return value


def binding(sha='hash'):
    return dict(bound_artifact_fqdn='synthetic.equation',bound_version_content_hash=sha,status='active')


def test_projection_grouping_preserves_source_relations():
    old=source_inventory([signature(['edge-a']),signature(['edge-b'])],[binding(),binding()])
    new=source_inventory([signature(['edge-a','edge-b'],True)],[binding()])
    assert old['relations']==new['relations']
    assert old['binding_versions']==new['binding_versions']
    assert not old['projected'] and new['projected']


def test_added_edge_and_changed_rule_are_not_silent_equivalence():
    old=source_inventory([signature(['edge-a'])],[binding()])
    extra=source_inventory([signature(['edge-a','edge-b'],True)],[binding()])
    changed=source_inventory([signature(['edge-a'],True,rule='corrected-rule')],[binding()])
    assert old['relations']!=extra['relations']
    assert old['relations']!=changed['relations']


def test_version_and_active_state_are_preserved():
    old=source_inventory([signature(['edge-a'])],[binding()])
    new=source_inventory([signature(['edge-a'],True)],[binding('different-hash')])
    inactive=binding();inactive['status']='inactive'
    assert old['binding_versions']!=new['binding_versions']
    assert not source_inventory([signature(['edge-a'])],[inactive])['all_bindings_active']
