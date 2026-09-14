"""Draft full-workflow TGS graph; callable binding and promotion still pending."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus


def build_graph():
    ports = [
        ('populations', 'dict', 'Aligned labeled/query populations in Keras BGR0..255 and PyTorch grayscale0..1 representations; binary labeled masks; explicit branch folds, nonconstant flags and pseudo-only labeled validation indices. Caller establishes identity and disjointness.'),
        ('reference_artifacts', 'dict', 'Caller-entitled ResNet34 and ResNeXt50 pretrained file paths and exact qualified hashes; no model artifacts bundled.'),
        ('runtime_directory', 'str', 'Private writable directory outside source repositories for verified checkpoint and workflow-state persistence.'),
        ('seed', 'int', 'Explicit uint32 initialization/sampling seed; immutable across resumed workflow context.'),
    ]
    node = AlgorithmicNode(node_id='execute', name='tgs_execute',
        description='Execute all63 source-budget fits, both pseudo-label rounds, selected checkpoint/TTA ensembles and final mosaic propagation',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.tgs_execution.tgs_execute',
        inputs=[IOSpec(name=name, type_desc=kind, constraints=constraints) for name, kind, constraints in ports],
        outputs=[IOSpec(name='result', type_desc='dict', constraints='Private ordered query binary masks plus aggregate verified execution counts; runtime arrays must not be published as catalog evidence.')])
    return CDGExport(nodes=[node], edges=[], metadata={
        'artifact_source':'competition_source_corrected_execution', 'publication_status':'draft', 'target_tier':3,
        'source_version_ids':['1796103a-eea9-5b66-9dff-ecdea34a459d'],
        'source_content_hash':'0a144d569512b1cae9884bbbb48ee831ede61475c68fbf1f1732752351d91a3e',
        'scope':'Complete corrected TGS training and transductive prediction workflow; original conceptual intake remains draft.',
        'training_plan_sha256':'377c93c42ea1b68ab82d0a1cca429cb92c0322e09115a545717e9779157d5c5b',
        'choices':['63fits and6530epoch ceiling with source early stopping; no reduced execution budget.',
                   'Corrected checkpoint copies, independent fold initialization, eval dropout and mosaic origin indexing.',
                   'Modern CPU reconstruction preserves reviewed Keras2.2.0 variance correction and source photometric/metric behavior.',
                   'Positional inputs replace source identifier sorting and filesystem enumeration; explicit seeds govern reconstruction.'],
        'limitations':['Draft callable target is not yet registered or qualified for selection.',
                       'Complete training execution, provider source binding and publication rejection/rollback/served checks remain pending.',
                       'No historical GPU equivalence, competition performance or human Tier1 certification claim.',
                       'External reference rights and aligned/disjoint population identity are caller responsibilities.'],
        'num_nodes':1, 'num_edges':0})
