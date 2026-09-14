"""Raw-population reconstructed workflow, with explicit synthetic/provisioned inputs.

A shared unsupervised base uses the labeled sequence population. Each model
then uses its explicit supervised split; pseudo-label teachers combine all
members. This is transductive and does not establish independent CV accuracy.
"""
import gc
from pathlib import Path
import tempfile

import numpy as np
from sciona.openvaccine_checkpoints import save_checkpoint
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_folding import fold_features
from sciona.openvaccine_models import create_model
from sciona.openvaccine_lifecycle_contract import validate_lifecycle
from sciona.openvaccine_population import population_weights,reverse_training_batch
from sciona.openvaccine_round import refine_round
from sciona.openvaccine_splits import validate_member_splits
from sciona.openvaccine_teacher import MemberCheckpoint,predict_teacher


def execute_population(source_dir,*,binary,parameters,sequences,targets,cluster_ids,proximity_factors,
                       splits,pseudo_sequences,prediction_sequences,pseudo_rounds,
                       pretraining_steps,initial_supervised_steps,rollback_policy,absolute_tolerance,std_ddof):
    for steps in (pretraining_steps,initial_supervised_steps):
        if type(steps) is not int or steps<1:raise ValueError('Positive explicit training step counts required')
    if not isinstance(pseudo_rounds,(list,tuple)) or not pseudo_rounds:
        raise ValueError('At least one explicit pseudo-label round required')
    targets=np.array(targets,dtype=np.float32,copy=True)
    weights=population_weights(cluster_ids,proximity_factors)
    membership=validate_member_splits(targets,weights,splits)
    validate_lifecycle(sequences,targets,pseudo_sequences,prediction_sequences,pseudo_rounds,membership,
        rollback_policy=rollback_policy,absolute_tolerance=absolute_tolerance,std_ddof=std_ddof)
    n,a=fold_features(source_dir,sequences,binary=binary,parameters=parameters)
    if targets.shape!=(len(n),n.shape[1]-2,5):raise ValueError('Labeled target length mismatch')
    pn,pa=fold_features(source_dir,pseudo_sequences,binary=binary,parameters=parameters)
    qn,qa=fold_features(source_dir,prediction_sequences,binary=binary,parameters=parameters)
    batches={};validations={}
    for key,split in membership.items():
        train=np.array(split.train);validation=np.array(split.validation)
        # Fixed reversal decisions are explicit recipe inputs in later rounds.
        batches[key]=(n[train],a[train],targets[train],weights[train])
        validations[key]=(n[validation],a[validation],targets[validation],weights[validation])
    ae=create_model(source_dir,'autoencoder')
    for _ in range(pretraining_steps):ae.train_step(n,a)
    base=ae.base.get_weights();del ae;gc.collect()
    counts=[]
    with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_population_') as temp:
        root=Path(temp);records=[]
        for family,slot in sorted(EXPECTED_MEMBERS):
            runtime=create_model(source_dir,family,base_weights=base)
            for _ in range(initial_supervised_steps):runtime.train_step(*batches[(family,slot)])
            directory=root/'initial'/(family+str(slot));digest=save_checkpoint(runtime,directory)
            records.append(MemberCheckpoint(family,slot,directory,digest))
            del runtime;gc.collect()
        del base
        for index,plan in enumerate(pseudo_rounds):
            recipes={}
            for key in EXPECTED_MEMBERS:
                supplied=plan[key]
                prepared=reverse_training_batch(*batches[key],supplied['supervised_reverse_flags'])
                recipes[key]=dict(supervised_batches=[prepared],validation_batch=validations[key],
                    eligible=supplied['eligible'],maximum_uncertainty=supplied['maximum_uncertainty'],
                    perturbations=supplied['perturbations'],sample_weights=supplied['sample_weights'],
                    reverse_flags=supplied['reverse_flags'])
            records,history=refine_round(source_dir,records,pn,pa,recipes=recipes,
                output_directory=root/('round'+str(index)),std_ddof=std_ddof,
                rollback_policy=rollback_policy,absolute_tolerance=absolute_tolerance)
            counts.append(dict(members=len(records),supervised_steps=sum(h['supervised_steps'] for h in history),
                pseudo_label_steps=sum(h['pseudo_label_section']['steps'] for h in history),
                rollbacks=sum(h['pseudo_label_section']['rolled_back'] for h in history)))
        outputs=predict_teacher(source_dir,records,qn,qa,std_ddof=std_ddof)
    return dict(predictions=outputs['prediction'],teacher_mean=outputs['teacher_mean'],
                teacher_std=outputs['teacher_std'],members=20,rounds=counts,
                pretraining_steps=pretraining_steps,initial_supervised_steps=20*initial_supervised_steps)
