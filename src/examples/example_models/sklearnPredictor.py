import torch
import numpy as np
import condensite as cde
from sklearn.ensemble import HistGradientBoostingRegressor


class TreePredictor(cde.CondensitePredictor):
    def __init__(self, params):
        self.tree = HistGradientBoostingRegressor(**params)

    def device(self):
        return "cpu"

    def to(self, device):
        return self

    def forward(self, x):
        return self.tree.predict(x)

    def predict(self, x):
        return torch.tensor(self.forward(x), dtype=torch.float32)

    def fit(self, dataset: cde.RepeatDataset):
        # create data loader (c.f. cde.RepeatDataset and cde.concat_collate)
        train_DL = torch.utils.data.DataLoader(dataset=dataset,                 
                                               batch_size=dataset.n*dataset.M, 
                                               shuffle=True, 
                                               collate_fn=cde.concat_collate)
        xtorch, ytorch = next(iter(train_DL))  # single batch of all data
        x, y = xtorch.numpy(), ytorch.numpy()
        self.tree.fit(x, y)
        return np.mean(np.square(self.tree.predict(x)-y))  # mse