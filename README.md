# condensité
(Pronounciation: _kon-dahn-see-tay_, like the French word for "density".)

The approach behind `condensité` enables transforming conditional density estimation into a single nonparametric regression task. Our package implements this approach and provides a way of turning regressors from common libraries such as `sklearn` or `torch` into conditional density estimators. See below for a complete example. See `example.py` for further examples using the reference implementations in `src/example_models`.

```python
import torch
import condensite as cde
from sklearn.ensemble import HistGradientBoostingRegressor

# create a condensité predictor based on sklearn's HistGradientBoostingRegressor
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
        self.tree.fit(xtorch.numpy(), ytorch.numpy())

# instantiate and wrap using Condensite
tree_predictor = TreePredictor(params={
    'max_bins': 40,
    'max_leaf_nodes': 100,
    'l2_regularization': 0.1,
})
sklearn_model = cde.Condensite(tree_predictor, h=0.01, M=100, name='sklearn_model')

# generate some data and fit
def sample_data(n_obs=1000, n_features=10):
    x = torch.empty((n_obs, n_features)).normal_(0,1)
    x1 = x[:, 0]
    std = torch.sqrt(0.25 + x1**2)
    noise = torch.distributions.Normal(torch.zeros_like(x1), std).sample()
    y = x1 + noise
    return x, y

# fit conditional density estimator
x_train, y_train = sample_data()
sklearn_model.fit(x_train, y_train, train_frac=0.8, n_grid_ISE=100)

# evaluate fit out-of-sample
x_test, y_test = sample_data()
test_ISE = cde.utils.ISE(sklearn_model, x_test, y_test, n_grid=100).item()
print(f'{sklearn_model.name}:\t {test_ISE=:.4f}')
```

If you find our algorithms useful please consider citing
```
```