import numpy as np
from descartes import crops_list

crops_enc = {id_: k + 1 for k, id_ in enumerate(sorted(crops_list))}
crops_dec = {crops_enc[id_]: id_ for id_ in crops_enc}


def get_monthly_arrays(data: np.ndarray, info: list, time_steps=12) -> np.ndarray:
    """Average imagery arrays by month to return a 4D tensor with 12 time steps.
       Default Descatres temporal imagery is of shape: (time_steps, bands, height, width)
       Need to also permute axes to the Tensorflow standard: (time_steps, height, width, bands)"""
    years = sorted(list(set([x['group'][0] for x in info])))
    months = sorted(list(set([x['group'][1] for x in info])))
    dates_ = list(zip(np.arange(data.shape[0], dtype=int), [x['group'] for x in info]))
    date_ranges = {(x, y): [] for x in years for y in range(1, 13)}
    for d in dates_:
        date_ranges[d[1][:2]].append(d[0])
    avg_array = np.zeros((time_steps * len(years), data.shape[1], data.shape[2], data.shape[3]))
    for k, dr in enumerate(sorted([(x, y) for x in years for y in months])):
        if date_ranges[dr]:
            avg_array[k] = data[date_ranges[dr][0]:date_ranges[dr][-1] + 1].mean(axis=0)
    return avg_array.transpose((0, 2, 3, 1))


def mask_crop_layer(cdl: np.ndarray, nclasses: int) -> np.ndarray:
    cdl = np.array(cdl[0, 0])
    y = np.zeros((cdl.shape[0], cdl.shape[1], nclasses), dtype='int32')
    y[:, :, 0] = np.asanyarray(~np.isin(cdl, list(crops_enc)), dtype='int32')
    for k in crops_dec:
        y[:, :, k] = np.asanyarray(cdl == crops_dec[k], dtype='int32')
    return y
