"""Pinned per-level training schedule and connected batch execution."""
import numpy as np
from sciona.m5u_sampling import sample
from sciona.m5u_targets import assemble
from sciona.m5u_bagging import train

QUANTILES=(.005,.025,.165,.25,.5,.75,.835,.975,.995)
QUANTILE_WEIGHTS=(.1,.2,.6,.8,1.,.9,.7,.2,.1)
PARAMETERS={1:(.3,.7),2:(.1,.7),3:(.1,.5),4:(.3,.5),5:(.15,1.),
            6:(.2,.5),7:(.1,1.),8:(.2,.5),9:(.1,.5),10:(.05,.5),
            11:(.04,1.),13:(.12,2.),14:(.065,2.),15:(.03,.5)}


def schedule(level,factor,*,speed=False,super_speed=False):
    if (type(level) is not int or level not in PARAMETERS
            or isinstance(factor,bool) or not isinstance(factor,(int,float))
            or not np.isfinite(factor) or factor<=0
            or type(speed) is not bool or type(super_speed) is not bool):
        raise ValueError('Supported source level, positive multiplier and boolean modes required')
    fraction,scale_range=PARAMETERS[level]
    sample_multiplier=.8/(5 if super_speed else 2 if speed else 1)
    iterations=int((2.2 if level<=9 else 1.66)*(16-min(level,12))
                   *(.25 if super_speed else .5 if speed else 1))
    fraction=fraction*sample_multiplier*(1/factor)**.6
    if not np.isfinite(fraction):raise ValueError('Sampling fraction overflow')
    return dict(fraction=fraction,scale_range=scale_range,iterations=iterations,
                fast=speed or super_speed,bags=1,single_fold=True)


def train_batch(prepared,population,calendar,controller_level,*,speed=False,super_speed=False,seed=0):
    """One shared sample per level, all nine quantiles, latest outer holdout.

    The controller level is deliberately distinct from the fitted level: the
    documented combined batch uses -1 and thus the low-level mask exponent.
    """
    if type(controller_level) is not int or type(seed) is not int or not 0<=seed<2**31:
        raise ValueError('Integer controller level and valid seed required')
    levels=np.unique(prepared['levels'])
    if controller_level in (13,14,15):
        if not np.array_equal(levels,[controller_level]):raise ValueError('Base controller requires its own partition')
    elif controller_level==-1:
        if (levels>11).any():raise ValueError('Combined aggregate controller requires levels through 11')
    else:raise ValueError('Use the documented aggregate or separate base controllers')
    rng=np.random.RandomState(seed);plans=[]
    for level in levels:
        factors=np.unique(prepared['factors'][prepared['levels']==level])
        if factors.size!=1:raise ValueError('One multiplier per fitted level required')
        config=schedule(int(level),float(factors[0]),speed=speed,super_speed=super_speed)
        plans.append((int(level),config,[int(rng.randint(0,2**31-1)) for _ in range(3)]))
    fitted={};reports={}
    for level,config,seeds in plans:
        rows,horizons=sample(population['weights'],population['levels'],level,config['fraction'],seed=seeds[0])
        bag=assemble(population,rows,horizons,prepared['history'],calendar,-1,
                     scale_range=config['scale_range'],seed=seeds[1])
        models=train([bag],QUANTILES,QUANTILE_WEIGHTS,controller_level,single_fold=True,
                     fast=config['fast'],iterations=config['iterations'],seed=seeds[2])
        fitted[level]=models
        reports[level]=dict(config,models=len(models),training_rows=len(bag['targets']),
                            features=len(bag['features'].columns),controller_level=controller_level,
                            mask_exponent=.35 if controller_level>=11 else .25,
                            candidate_fits=sum(m.report['candidate_fits'] for m in models),
                            refits=len(models),outer_group_isolation=all(m.report['outer_group_excluded'] for m in models))
    return fitted,reports
