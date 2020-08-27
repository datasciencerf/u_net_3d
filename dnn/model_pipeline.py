import numpy as np
from descartes import get_masked_daily_product, ndvi, get_cdl, isin, crops_list
from descartes.process_images import get_monthly_arrays, mask_crop_layer
from descartes.workflow import wf
from tqdm import tqdm
from dnn import u_net_3d


class UNetPipeline(object):

    def __init__(self,
                 model_params,
                 tiles,
                 img_prod_id,
                 train_ix,
                 test_ix,
                 random_seed=2020):
        self.model = u_net_3d(**model_params)
        self.img_prod_id = img_prod_id
        self.tiles = tiles
        self.train_ix = train_ix
        self.test_ix = test_ix
        self.model_params = model_params
        self.random_state = np.random.RandomState(random_seed)
        self.date_ranges = {
            '2017': ['2017-01-01', '2018-01-01'],
            '2018': ['2018-01-01', '2019-01-01'],
            '2019': ['2019-01-01', '2020-01-01']
        }

    def data_loader(self, batch_ix, year):
        batch_size = len(batch_ix)
        start_date, end_date = self.date_ranges[year]
        image_ = get_masked_daily_product(self.img_prod_id, start_date, end_date).pick_bands("red green blue nir swir1")
        image_ = image_.concat_bands(ndvi(image_))
        cdl_ = get_cdl(start_date, end_date)
        cdl_iscrop = isin(cdl_, crops_list)
        time_steps = self.model_params['time_steps']
        img_height = self.model_params['img_height']
        img_width = self.model_params['img_width']
        bands = self.model_params['bands']
        batch_x = np.zeros((batch_size, time_steps, img_height, img_width, bands))
        batch_y = np.zeros((batch_size, img_height, img_width, self.model_params['nclasses']))
        tiles_to_run = self.tiles[batch_ix]
        jobs = [wf.compute([image_.ndarray, image_.properties, cdl_.ndarray, cdl_iscrop.ndarray],
                               tile, block=False) for tile in tiles_to_run]
        for k, job in enumerate(jobs):
            job_not_done = True
            while job_not_done:
                try:
                    img_data, img_info, cdl_data, cdl_mask = job.result(progress_bar=False)
                    job_not_done = False
                except Exception as e:
                    print(e)
            batch_x[k] = get_monthly_arrays(img_data, img_info)
            batch_y[k] = mask_crop_layer(cdl_data, self.model_params['nclasses'])
        return batch_x, batch_y

    def train_model(self, batch_size, epochs):
        for year in self.date_ranges:
            test_x, test_y = self.data_loader(self.test_ix, year)
            ix_base = np.array(self.train_ix)
            for e in range(epochs):
                self.random_state.shuffle(ix_base)
                for k in range(0, len(ix_base), batch_size):
                    train_x, train_y = self.data_loader(ix_base[k:k + batch_size], year)
                    print(f"Year {year}")
                    print(f"Epoch {e + 1} / {epochs}: ")
                    print(f"Batch {(k // batch_size) + 1} out of {int(np.ceil(len(ix_base) / batch_size))}")
                    print("#" * 32)
                    self.model.fit(x=train_x, y=train_y, batch_size=batch_size,
                                   epochs=1, verbose=1,
                                   validation_data=[test_x, test_y])

    def save_model(self, save_fp):
        self.model.save(save_fp)
