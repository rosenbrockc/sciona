"""Complete reconstructed teacher/student round with explicit member recipes.

A frozen input checkpoint population generates the teacher before any student
updates. Recipes supply fold-specific supervised/validation batches and target
eligibility/draws; the caller must establish their provenance separately.
"""
import gc
from pathlib import Path

import numpy as np

from sciona.openvaccine_checkpoints import load_checkpoint, save_checkpoint
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_population import reverse_training_batch
from sciona.openvaccine_sections import run_pseudo_label_section
from sciona.openvaccine_round_contract import validate_recipes
from sciona.openvaccine_targets import uncertainty_targets, validation_rollback_required
from sciona.openvaccine_teacher import MemberCheckpoint, predict_teacher


def refine_round(source_dir, checkpoints, pseudo_nodes, pseudo_adjacency, *, recipes,
                 output_directory, std_ddof, rollback_policy, absolute_tolerance):
    records=tuple(checkpoints)
    validate_recipes(recipes,pseudo_nodes,pseudo_adjacency,std_ddof=std_ddof)
    if rollback_policy not in ('weights_only','model_and_optimizer'):
        raise ValueError('Explicit rollback policy required')
    validation_rollback_required(0.,0.,absolute_tolerance=absolute_tolerance)
    output_directory=Path(output_directory)
    if output_directory.exists():
        raise FileExistsError('Refinement output must be a new directory')
    teacher=predict_teacher(source_dir,records,pseudo_nodes,pseudo_adjacency,std_ddof=std_ddof)
    prepared={}
    for key,recipe in recipes.items():
        target=uncertainty_targets(teacher['teacher_mean'],teacher['teacher_std'],recipe['eligible'],
            maximum_uncertainty=recipe['maximum_uncertainty'],perturbations=recipe['perturbations'])
        if not (np.isfinite(target) & (np.asarray(recipe['sample_weights'])>0)[:,None,None]).any():
            raise ValueError('No positive-weight pseudo-labels remain after uncertainty masking')
        prepared[key]=reverse_training_batch(pseudo_nodes,pseudo_adjacency,target,
            recipe['sample_weights'],recipe['reverse_flags'])
    output_directory.mkdir(parents=True,exist_ok=False)
    outputs=[];history=[]
    for record in records:
        key=(record.family,record.slot);recipe=recipes[key]
        runtime=load_checkpoint(source_dir,record.directory,
            expected_manifest_sha256=record.manifest_sha256,expected_family=record.family)
        for batch in recipe['supervised_batches']:
            runtime.train_step(*batch)
        def score(model):
            n,a,t,w=recipe['validation_batch']
            value=float(model.model.loss(model.tf.constant(t),model.tf.constant(model.predict(n,a)),model.tf.constant(w)).numpy())
            if not np.isfinite(value):raise ValueError('Nonfinite round validation')
            return value
        section=run_pseudo_label_section(runtime,[prepared[key]],score,
            absolute_tolerance=absolute_tolerance,rollback_policy=rollback_policy)
        directory=output_directory/(record.family+'_'+str(record.slot))
        digest=save_checkpoint(runtime,directory)
        outputs.append(MemberCheckpoint(record.family,record.slot,directory,digest))
        history.append(dict(family=record.family,slot=record.slot,
                            supervised_steps=len(recipe['supervised_batches']),pseudo_label_section=section,
                            parent_manifest_sha256=record.manifest_sha256))
        del runtime;gc.collect()
    return tuple(outputs),history
