""" Use TabICL as predictor in Condensité. """
import torch
import numpy as np
from condensite import CondensitePredictor
from tabicl import TabICLRegressor


class TabICLPredictor(CondensitePredictor):

    def __init__(self, n_estimators, batch_size=5000):
        self.batch_size = batch_size  # for keeping cuda memory in check
        # TabICLRegressor uses the GPU if available; the kv cache avoids
        # re-encoding the training data for every prediction batch
        self.model = TabICLRegressor(n_estimators=n_estimators, kv_cache=True)

    def device(self):
        # return "cpu" since TabICLRegressor takes care of moving things to gpu
        return "cpu"

    def to(self, device):
        return self

    def forward(self, features):
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        return self.model.predict(features)

    def predict(self, features):
        preds = [self.forward(batch)
                 for batch in torch.split(features, self.batch_size)]
        return torch.tensor(np.concatenate(preds), dtype=torch.float32)

    def fit(self, dataset):
        # assemble the full (n*M) feature/target arrays directly, no batching
        X_expanded = dataset.x[dataset.j_map]
        features = torch.cat([X_expanded, dataset.y_aux.reshape(-1, 1)], dim=1).numpy()
        targets = dataset.targets.numpy()
        self.model.fit(features, targets)
        # don't return a loss since there is no actual fitting
        loss = None
        return loss
