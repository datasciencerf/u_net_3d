import numpy as np


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
    return avg_array.transpose((0, 3, 1, 2))


def mask_crop_layer(cdl_img: np.ndarray, cdl_mask: np.ndarray) -> np.ndarray:
    cdl_img, cdl_mask = cdl_img[0, 0], np.array(cdl_mask[0, 0], dtype='int')
    return cdl_img * cdl_mask
