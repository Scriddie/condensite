
import condensite as cde
import torch
import torch.distributions as dist
from example_models.sklearnPredictor import TreePredictor
from example_models.LGBMPredictor import LGBMPredictor
from example_models.torchPredictor import NNPredictor
import logging
logging.basicConfig(level=logging.INFO)

# create some example data
def sample_data(n_obs=1000, n_features=10):
    x = torch.empty((n_obs, n_features)).normal_(0,1)
    x1 = x[:, 0]
    std = torch.sqrt(0.25 + x1**2)
    noise = torch.distributions.Normal(torch.zeros_like(x1), std).sample()
    y = x1 + noise
    return x, y

x_train, y_train = sample_data()

# ------------------------------------------------------------------------------

# model 1) condensité with a sklearn predictor
tree_predictor = TreePredictor(params={
    'max_bins': 40,
    'max_leaf_nodes': 100,
    'l2_regularization': 0.1,
})
sklearn_model = cde.Condensite(tree_predictor, h=0.01, M=100, name='sklearn_model')
sklearn_model.fit(x_train, y_train, train_frac=0.8, n_grid_ISE=100)

# model 2) condensité with a lgbm predictor
lgbm_predictor = LGBMPredictor(params={
    'objective': 'regression', 
    'metric': 'mse',
    'min_data_in_leaf': 50,
    'max_bin': 40,
    'num_leaves': 40,
})
lgbm_model = cde.Condensite(lgbm_predictor, h=0.01, M=100, name='lgbm_model')
lgbm_model.fit(x_train, y_train, train_frac=0.8, n_grid_ISE=100)

# model 3) condensité with a torch predictor
torch_predictor = NNPredictor(
    n_in=x_train.shape[1]+1,  # note extra feature (all methods, here explicit)
    n_hidden=20,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4
)
torch_model = cde.Condensite(torch_predictor, h=0.01, M=100, name='torch_model')
torch_model.fit(x_train, y_train, train_frac=0.8, n_grid_ISE=100, epochs=20, patience=5, verbose=1)

# ------------------------------------------------------------------------------

# compare their out-of-sample ISE
x_test, y_test = sample_data()
for model in [sklearn_model, lgbm_model, torch_model]:
    test_ISE = cde.utils.ISE(model, x_test, y_test, n_grid=100).item()
    print(f'{model.name}:\t {test_ISE=:.4f}')