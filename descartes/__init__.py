import numpy as np
import descarteslabs.workflows as wf


def get_masked_daily_product(product_id: str, start_datetime: str, end_datetime: str) -> wf.ImageCollection:
    "Get a product by ID, masked by the DL cloud mask and mosaicked by day"
    if 'airbus' in product_id.lower():
        cloud = 'derived:visual_cloud_mask'
    else:
        cloud = "valid-cloudfree"
    ic = wf.ImageCollection.from_id(product_id, start_datetime, end_datetime)
    cloudmask = (
            wf.ImageCollection.from_id(
                product_id, start_datetime, end_datetime
            ).pick_bands(cloud)
            == 0
    )

    # Make an ImageCollectionGroupby object, for quicker lookups from `ic` by date (you can use it like a dict)
    ic_date_groupby = ic.groupby(dates=("year", "month", "day"))
    # For each cloudmask date, pick the corresponding image from `ic` by date, mosiac both, and mask them.
    # (Not all scenes have cloudmasks processed, so this ensures we only return scenes that do.)
    return cloudmask.groupby(dates=("year", "month", "day")).map(
        lambda ymd, mask_imgs: ic_date_groupby[ymd].mosaic().mask(mask_imgs.mosaic())
    )


def get_cdl(start_date, end_date):
    return wf.ImageCollection.from_id(
        "usda:cdl:v1", start_datetime=start_date, end_datetime=end_date
    ).pick_bands("class")


def ndvi(ic: wf.ImageCollection) -> wf.ImageCollection:
    nir, red = ic.unpack_bands("nir red")
    ndvi = (nir - red) / (nir + red)
    return ndvi.rename_bands("ndvi")


def isin(ic: wf.ImageCollection, values: list) -> wf.ImageCollection:
    "Like np.isin, for Workflows"
    assert len(values) > 0
    result = False
    for value in values:
        result = result | (ic == value)
    return result


grains_oils_grass_beans = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                           30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 51,
                           52, 53, 225, 226, 228, 230, 232, 234, 235, 236, 237, 238, 239, 240, 241, 254]

deli_crops = [14, 48, 49, 50, 54, 55, 57, 206, 207, 208, 209, 213, 214, 216,
              219, 221, 222, 224, 227, 229, 231, 242, 243, 244, 245, 246, 247,
              248, 249, 250]

tree_crops = [66, 67, 68, 69, 72, 74, 75, 76, 77, 204, 210, 211, 212, 215, 217,
              218, 220, 223]
crops_list = deli_crops + tree_crops
