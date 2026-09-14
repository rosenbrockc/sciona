"""Final no-evaluation Web Traffic trainer schedule over runtime population size.

MIT source adaptation; no neural execution or TensorFlow scheduling is implied.
"""


def training_schedule(n_pages, batch_size=256, max_epoch=100, max_steps=11500, save_from_step=10500):
    """Yield (step, epoch, save_checkpoint); preserve source strict stop comparison."""
    if any(type(v) is not int or v<=0 for v in [n_pages,batch_size,max_epoch]):
        raise ValueError('Positive integer population, batch size and epoch count required')
    for value in [max_steps,save_from_step]:
        if value is not None and (type(value) is not int or value<0):
            raise ValueError('Optional nonnegative integer step thresholds required')
    per_epoch=n_pages//batch_size
    every=int(round(per_epoch*.1))
    if every==0:
        raise ValueError('Population too small for source evaluation cadence')
    step=0
    for epoch in range(max_epoch):
        for _ in range(per_epoch):
            step+=1
            save=bool(step%every==0 and save_from_step and step>=save_from_step)
            yield step,epoch,save
            if max_steps and step>max_steps:
                break
        if max_steps and step>max_steps:
            break


def retained_checkpoint_steps(schedule):
    """Source Saver keeps the ten most recent saved checkpoints."""
    return [step for step,_,save in schedule if save][-10:]
