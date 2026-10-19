""" Use TabPFN as predictor in Condensité. """
import torch
import numpy as np
from condensite import CondensitePredictor
from tabpfn import TabPFNRegressor


class TabPFNPredictor(CondensitePredictor):

    def __init__(self, n_estimators, batch_size=5000):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size  # for keeping cuda memory in check
        self.model = TabPFNRegressor(
            fit_mode="fit_with_cache",
            n_estimators=n_estimators,
            keep_cache_on_device=False,
            device=self._device,
        )

    def device(self):
        # return "cpu" since TabPFNRegressor takes care of moving things to gpu
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
        return torch.tensor(np.concatenate(preds), 
                            dtype=torch.float32, device=self._device)

    def fit(self, dataset):
        # assemble the full (n*M) feature/target arrays directly, no batching
        X_expanded = dataset.x[dataset.j_map]
        features = torch.cat([X_expanded, dataset.y_aux.reshape(-1, 1)], dim=1).numpy()
        targets = dataset.targets.numpy()
        self.model.fit(features, targets)
        # don't return a loss since there is no actual fitting
        loss = None
        return loss
