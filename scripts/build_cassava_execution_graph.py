"""Build the unpublished, corrected Cassava CPU execution graph."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_graph():
    constraints = {
        'sample_keys': 'Nonempty unique runtime strings; physical-image aliases resolved by caller',
        'labels': 'Aligned integer class ordinals 0..4; bool rejected; all classes in each training complement',
        'assignments': 'Aligned integer held-out fold ordinals 0..4; bool rejected; all five folds nonempty',
        'plan': 'Validated Cassava FoldPlan; revalidated at execution',
        'source_rows': 'Keyed encoded 600x800 RGB images; exact plan coverage; no identical decoded images',
        'efficientnet_rows': 'Keyed encoded 512x512 RGB images; exact plan coverage; caller proves preparation relationship',
        'query_rows': 'Nonempty unique keyed encoded 600x800 RGB images; no identical decoded images',
        'references': 'Exactly vit, resnext, efficientnet and cropnet; local paths plus reviewed SHA-256 references; no implicit download',
        'torch_batch_size': 'Explicit positive integer; training and validation populations must supply retained batches',
        'torch_workers': 'Explicit nonnegative integer; qualified synthetic execution uses zero',
        'efficientnet_batch_size': 'Explicit positive integer; training population must supply at least one full batch',
        'output_directory': 'Caller-managed private checkpoint directory that must not already exist',
        'result': 'Private predictions (N x 5 scores summing to 3 and argmax labels), family outputs, histories, evaluations and checkpoint paths',
    }
    def node(name, description, inputs, output, output_type):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.ml.cassava_execution.cassava_' + name,
            inputs=[IOSpec(name=n, type_desc=t, constraints=constraints[n]) for n, t in inputs],
            outputs=[IOSpec(name=output, type_desc=output_type, constraints=constraints[output])])
    nodes = [
        node('fold_plan', 'Validate aligned identities, classes and five disjoint held-out folds',
             [('sample_keys', 'list'), ('labels', 'list'), ('assignments', 'list')], 'plan', 'object'),
        node('train_ensemble', 'Train three fivefold families, refit B4 and combine with frozen CropNet',
             [('plan', 'object'), ('source_rows', 'list'), ('efficientnet_rows', 'list'),
              ('query_rows', 'list'), ('references', 'dict'), ('torch_batch_size', 'int'),
              ('torch_workers', 'int'), ('efficientnet_batch_size', 'int'),
              ('output_directory', 'str')], 'result', 'dict')]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='fold_plan',
        target_id='train_ensemble', output_name='plan', input_name='plan',
        source_type='object', target_type='object')], metadata={
        'artifact_source': 'competition_source_corrected_execution',
        'publication_status': 'draft', 'target_tier': 3,
        'source_version_ids': ['735c71d1-6f8a-565b-a5c3-f1bcf3a89285'],
        'scope': 'Corrected pretrained four-family CPU training and inference reconstruction.',
        'choices': ['ResNeXt 15 epochs per fold; ViT 10 epochs per fold',
                    'EfficientNet fivefold CV up to 20 epochs with early stopping; separate 14-epoch full refit',
                    'Frozen official CropNet reference; unknown-class mass distributed equally',
                    'Half ViT/ResNeXt sum plus B4 plus CropNet; scores sum to three',
                    'Disjoint ViT validation correction; explicit full-population B4 normalization adaptation'],
        'limitations': ['Caller establishes physical-image identity and prepared-view provenance',
                        'Explicit CPU batch/worker settings; historical TPU and leaderboard equivalence unproven',
                        'Historical-reference timm weights and official CropNet; exact winner runtime/cache identity unproven',
                        'Synthetic execution evidence does not establish competitive accuracy',
                        'Local reference artifacts and private checkpoints remain caller managed'],
        'num_nodes': 2, 'num_edges': 1})
