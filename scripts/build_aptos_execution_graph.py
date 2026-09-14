"""Build the unpublished, corrected APTOS two-stage execution graph."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_graph():
    constraints = {
        'base': 'Pair of unique private keys and supplied 0..4 labels; caller-prepared combined first-stage population',
        'average': 'Pair of unique private keys and supplied 0..4 labels; enters only stage two with label/teacher averaging',
        'grouped': 'Pair of unique private keys and supplied 0..3 group labels; enters only stage two with group-mean target bounding',
        'pseudo_keys': 'Nonempty unique unlabeled private keys; all four population roles disjoint',
        'plan': 'Validated Aptos Population with source-separated roles; revalidated at execution',
        'query_keys': 'Nonempty unique private keys; disjoint from supplied-label roles; may overlap pseudo keys',
        'images': 'Exactly population and query coverage; values are nonempty RGB uint8 HWC arrays; caller establishes physical identity and provenance',
        'references': 'Exactly four model families mapped to local safetensors paths and reviewed full SHA256; no downloads',
        'first_stage_epochs': 'Exactly eight (family, integer replica) keys; positive integer multiples of five; historical winning budgets unproven',
        'seeds': 'Exactly eight (family, integer replica) keys with distinct nonnegative 32-bit integer seeds',
        'batch_size': 'Explicit positive integer; retains final partial batch',
        'learning_rate': 'Explicit finite positive Adam learning rate; constant schedule and fresh stage-two optimizer are reference choices',
        'lower_deviation': 'Finite nonnegative lower bound distance from group mean; source demonstrates 0.5',
        'upper_deviation': 'Finite nonnegative upper bound distance; symmetric 0.5 is a reference choice',
        'tie_policy': 'Explicit upper or lower handling of equality at fixed cut points 0.7,1.5,2.5,3.5',
        'work_root': 'Existing caller-managed private directory; temporary trained checkpoints removed after execution',
        'result': 'Private ordered scalar scores and ordinals, 16 fit histories, teacher/final model outputs and soft targets; checkpoint replay evidence',
    }
    def node(name, description, inputs, output, output_type):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.ml.aptos_execution.aptos_' + name,
            inputs=[IOSpec(name=n, type_desc=t, constraints=constraints[n]) for n, t in inputs],
            outputs=[IOSpec(name=output, type_desc=output_type, constraints=constraints[output])])
    nodes = [node('population', 'Validate first-stage base and three disjoint second-stage added populations',
        [('base', 'list'), ('average', 'list'), ('grouped', 'list'), ('pseudo_keys', 'list')], 'plan', 'object'),
        node('train_ensemble', 'Fit eight models, refine targets using ensemble teacher, continue ten epochs and blend scores',
        [('plan', 'object'), ('query_keys', 'list'), ('images', 'dict'), ('references', 'dict'),
         ('first_stage_epochs', 'dict'), ('seeds', 'dict'), ('batch_size', 'int'), ('learning_rate', 'float'),
         ('lower_deviation', 'float'), ('upper_deviation', 'float'), ('tie_policy', 'str'), ('work_root', 'str')],
         'result', 'dict')]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='population', target_id='train_ensemble',
        output_name='plan', input_name='plan', source_type='object', target_type='object')], metadata={
        'artifact_source': 'competition_source_corrected_execution', 'publication_status': 'draft', 'target_tier': 3,
        'source_version_ids': ['94df8d52-0272-5d0b-b84f-1f150d8e11f0'],
        'scope': 'Corrected pretrained eight-model two-stage CPU regression and thresholded inference reference.',
        'choices': ['Two separately seeded models in each of four Inception/SE-ResNeXt families; native 512/384 resolutions',
            'Trainable GeM and scalar linear head; Smooth-L1 loss; arithmetic model averaging',
            'Only base labels enter stage one; averaging, four-level group bounding and soft pseudo-label roles enter stage two',
            'Ensemble teacher; each own checkpoint continues ten epochs with a fresh constant-LR Adam optimizer',
            'Reference augmentation before plain bilinear resize and legacy publisher normalization; single-view inference'],
        'limitations': ['Original winner runtime, model artifact identity and optimization settings unproven',
            'First-stage budgets, augmentation semantics, teacher aggregation, full group-bound interval and threshold equality are explicit reconstruction choices',
            'Caller establishes input-role provenance and physical-image identity; queries may overlap pseudo-label training population',
            'External weights remain caller-managed with separate use rights; no model redistribution grant inferred',
            'Synthetic execution does not establish competition accuracy or human-reviewed Tier 1 status'],
        'num_nodes': 2, 'num_edges': 1})
