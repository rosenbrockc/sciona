"""Pin and inspect actual training construction; does not execute TensorFlow."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def review(root, source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    trees={}
    for name in ['model.py','trainer.py','hparams.py','Readme.md','cocob.py']:
        content=(source/name).read_bytes()
        assert hashlib.sha256(content).hexdigest()==pins['files'][name]
        if name.endswith('.py'):trees[name]=ast.parse(content)
    model=trees['model.py']
    function=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='make_train_op')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call)]
    adam=next(n for n in calls if ast.unparse(n.func)=='tf.train.AdamOptimizer')
    assert not adam.args and not adam.keywords
    assert not any('COCOB' in ast.unparse(n) for n in model.body)
    clip=next(n for n in calls if ast.unparse(n.func)=='tf.clip_by_global_norm')
    assert ast.unparse(clip.args[1])=='GRAD_CLIP_THRESHOLD'
    threshold=next(n.value.value for n in model.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='GRAD_CLIP_THRESHOLD' for t in n.targets))
    assert threshold==10
    apply=next(n for n in calls if ast.unparse(n.func)=='optimizer.apply_gradients')
    assert not apply.keywords
    ema=next(n for n in calls if ast.unparse(n.func)=='tf.train.ExponentialMovingAverage')
    assert {k.arg:ast.unparse(k.value) for k in ema.keywords}=={'decay':'ema_decay','num_updates':'glob_step'}
    ema_call=next(n for n in calls if ast.unparse(n.func)=='ema.apply')
    control=next(n for n in ast.walk(function) if isinstance(n,ast.With))
    assert ema_call.lineno < control.lineno
    assert ast.unparse(control.items[0].context_expr)=='tf.control_dependencies([sgd_op])'
    assert 'tf.group(update_ema)' in ast.unparse(control)
    trainer=trees['trainer.py']
    train=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    saves=[n for n in ast.walk(train) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tf.train.Saver']
    assert any(any(k.arg=='max_to_keep' and isinstance(k.value,ast.Constant) and k.value.value==10 for k in n.keywords) for n in saves)
    increments=[n for n in ast.walk(train) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tf.assign_add']
    assert any(ast.unparse(n)=='tf.assign_add(global_step, 1)' for n in increments)
    readme=(source/'Readme.md').read_text()
    for flag in ['--asgd_decay=0.99','--max_steps=11500','--save_from_step=10500','--n_models=3','--no_eval','--no_forward_split']:
        assert flag in readme
    paths=['scripts/review_webtraffic_training.py','docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,inspection_only=True,
        optimizer={'class':'tf.train.AdamOptimizer','arguments':'TensorFlow defaults; version-specific constants must be verified when runtime is selected',
                   'global_norm_clip':threshold,'apply_gradients_increments_global_step':False},
        moving_average={'decay':.99,'num_updates':'shared global step',
                        'variable_selection':'prefix-filtered trainable variables',
                        'ordering_caveat':'ema.apply is constructed before control_dependencies([sgd_op]); grouping both does not itself prove EMA assignments execute after Adam. Inspect actual graph/runtime before claiming ordered post-update EMA.'},
        schedule={'models':3,'max_steps':11500,'save_from_step':10500,'max_checkpoints_retained':10,
                  'checkpoint_cadence':'inside eval_every_step branch, even when evaluation stages are disabled',
                  'global_step':'incremented separately by trainer'},
        unused_source={'cocob.py':'Not imported or selected by model.py; file has separate Apache-2.0 header, not repository MIT header.'},
        corrections=['Earlier work-queue COCOB requirement was inferred from file presence and is superseded: implement source Adam path.',
                     'EMA post-Adam ordering is not yet established; do not silently repair or claim it.'],
        remaining=['Actual legacy TF graph construction and optimizer defaults verification','GRU and state conversion','Loss and regularization semantics','Training/EMA execution ordering and checkpoint lifecycle'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=review(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_training_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'approved':False,'source_inspection':'passed','optimizer':report['optimizer'],
                      'corrections':report['corrections']}))
