import numpy as np
import pathlib


class UNetPipeline(object):

    def __init__(self, model, model_params,
                 image_dir, image_base_fp,
                 label_dir, label_base_fp,
                 train_ix, test_ix, random_seed=2020):
        self.model = model
        self.image_dir = pathlib.Path(image_dir)
        self.label_dir = pathlib.Path(label_dir)
        self.image_base_fp = image_base_fp
        self.label_base_fp = label_base_fp
        self.train_ix = train_ix
        self.test_ix = test_ix
        self.model_params = model_params
        self.random_state = np.random.RandomState(random_seed)

    def data_loader(self, batch_ix):
        batch_size = len(batch_ix)
        time_steps = self.model_params['time_steps']
        img_height = self.model_params['img_height']
        img_width = self.model_params['img_width']
        bands = self.model_params['bands']
        batch_x = np.zeros((batch_size, time_steps, img_height, img_width, bands))
        batch_y = np.zeros((batch_size, img_height, img_width))
        for k, ix in enumerate(batch_ix):
            batch_x[k] = np.load(str(self.image_dir / f"{self.image_base_fp}{ix}"))
            batch_y[k] = np.load(str(self.label_dir / f"{self.label_base_fp}{ix}"))
        return batch_x, batch_y

    def train_model(self, batch_size, epochs):
        test_x, test_y = self.data_loader(self.test_ix)
        ix_base = np.array(self.train_ix)
        for e in range(epochs):
            self.random_state.shuffle(ix_base)
            for k in range(0, len(ix_base), batch_size):
                train_x, train_y = self.data_loader(ix_base[k:k + batch_size])
                print(f"Epoch {e + 1} / {epochs}: ")
                print(f"Batch {(k // batch_size) + 1} out of {int(np.ceil(len(ix_base) / batch_size))}")
                print("#"*32)
                self.model.fit(x=train_x, y=train_y, batch_size=batch_size,
                               epochs=1, verbose=1,
                               validation_data=[test_x, test_y])

    def save_model(self, save_fp):
        self.model.save(save_fp)
