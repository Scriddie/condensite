import torch
from torch.utils.data import DataLoader
import condensite as cde
import lightgbm as lgb
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Converting data to scipy sparse matrix.")


class LGBMPredictor(cde.CondensitePredictor):
    def __init__(self, params):
        self.tree = None
        self.params = params

    def device(self):
        return "cpu"

    def to(self, device):
        return self

    def forward(self, x):
        return self.tree.predict(x)

    def predict(self, x):
        return torch.tensor(self.forward(x), dtype=torch.float32)

    def fit(self, dataset):
        # create dataset (c.f. RepeatDataset and concat_collate)
        train_DL = torch.utils.data.DataLoader(dataset=dataset,                 
                                               batch_size=dataset.n*dataset.M, 
                                               shuffle=True, 
                                               collate_fn=cde.concat_collate)
        xtorch, ytorch = next(iter(train_DL))  # single batch of all data
        x, y = xtorch.numpy(), ytorch.numpy()
        train_data = lgb.Dataset(x, label=y)
        self.tree = lgb.train(self.params, train_data)
        return np.mean(np.square(self.tree.predict(x)-y)).item()  # mse