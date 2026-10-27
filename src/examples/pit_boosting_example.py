"""
Correction of a condensité model in PIT space (PITBoosting), with LightGBM regressors for both the base model and the
correction.
"""
import warnings
from copy import deepcopy
import numpy as np
import torch
import lightgbm as lgb
import condensite as cde
warnings.filterwarnings("ignore", message="Converting data to scipy sparse matrix.")


# create a condensité predictor based on LightGBM
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
        # single batch of all n*M rows (c.f. cde.RepeatDataset and cde.concat_collate)
        train_DL = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=dataset.n*dataset.M,
                                               shuffle=True,
                                               collate_fn=cde.concat_collate)
        xtorch, ytorch = next(iter(train_DL))
        x, y = xtorch.numpy(), ytorch.numpy()
        self.tree = lgb.train(self.params, lgb.Dataset(x, label=y))
        return np.mean(np.square(self.tree.predict(x)-y)).item()  # mse


# create some example data (as in compare_predictors_example.py)
def sample_data(n_obs=1000, n_features=10):
    x = torch.empty((n_obs, n_features)).normal_(0,1)
    x1 = x[:, 0]
    std = torch.sqrt(0.25 + x1**2)
    noise = torch.distributions.Normal(torch.zeros_like(x1), std).sample()
    y = x1 + noise
    return x, y

x_train, y_train = sample_data()

# ------------------------------------------------------------------------------

lgbm_params = {
    'objective': 'regression',
    'metric': 'mse',
    'min_child_samples': 50,
    'max_bin': 40,
    'num_leaves': 40,
    'verbosity': -1,
}
# base model of the conditional density of y
base = cde.Condensite(
    LGBMPredictor(lgbm_params),
    h=[0.05], c=1, name='base'  # h deliberately chosen wide so boosting applies
)
# correction: conditional density of the base model's PITs on [0, 1]
correction = cde.Condensite(
    LGBMPredictor(lgbm_params),
    h=[0.1], c=1, name='correction'
)
booster = cde.PITBoosting(base, correction, K=5, name='boosted')
booster.fit(x_train, y_train)
# for comparison: the unboosted base model, fitted on all training data
unboosted = deepcopy(base)
unboosted.fit(x_train, y_train)

# cross-fitted PITs of the base model; uniform on [0, 1] if the base model is calibrated
freq = torch.histc(booster.PIT, bins=10, min=0, max=1) / len(booster.PIT)
print('PIT frequency per decile:', [round(f, 3) for f in freq.tolist()])
print(f'weight of the correction: lambda = {booster.lambda_:.3f}')

# out-of-sample ISE comparison of unboosted and boosted model
print('RESULTS:')
x_test, y_test = sample_data()
for model in [unboosted, booster]:
    test_ISE = cde.utils.ISE(model, x_test, y_test, n_grid=100).item()
    print(f'{model.name}:\t {test_ISE=:.4f}')