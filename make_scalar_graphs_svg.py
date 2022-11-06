import sys
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    print(ea.Tags()["scalars"])
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


if __name__=='__main__':
    events_path=sys.argv[1]
    scalars=['D_loss','G_loss','global_step/sec']
    data_frames=parse_tensorboard(events_path,scalars)

    smoothed=list()
    for scalar in scalars:
        smoothed.append(data_frames[scalar].ewm(alpha=1-0.95).mean())
    for p in range(3):
        plt.plot(data_frames[scalars[p]]["value"],alpha=0.4)
        plt.plot(smoothed[p]["value"])
        plt.title(scalars[p])
        plt.grid(alpha=0.3)
        plt.savefig((scalars[p]+'.svg').replace('/','_'))
