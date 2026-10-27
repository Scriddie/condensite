import torch
import condensite as cde
from test_core import TreePredictor, sample_data


def test_cross_fitted_PITs():
    torch.manual_seed(0)
    x, y = sample_data()
    base = cde.Condensite(TreePredictor(), h=[0.05], Mh=0.5)
    correction = cde.Condensite(TreePredictor(), h=[0.1], Mh=1)
    booster = cde.PITBoosting(base, correction, K=3).fit(x, y)
    assert len(booster.base_models) == 3
    assert torch.bincount(booster.folds).tolist() == [100, 100, 100]
    # fold models share the range of all labels, hence their prediction grid
    assert all((m.y_min, m.y_max) == (y.min().item(), y.max().item()) for m in booster.base_models)
    assert ((booster.PIT >= 0) & (booster.PIT <= 1)).all()


def test_choose_lambda():
    # log(1 + lambda) + log(1 - lambda/2) is maximal at lambda = 1/2
    assert abs(cde.PITBoosting.choose_lambda(torch.tensor([2., 0.5])) - 0.5) < 1e-5
    # a uniform correction density gives no correction
    assert cde.PITBoosting.choose_lambda(torch.ones(10)) == 0.


def test_predict():
    torch.manual_seed(0)
    x, y = sample_data()
    base = cde.Condensite(TreePredictor(), h=[0.05], Mh=0.5)
    correction = cde.Condensite(TreePredictor(), h=[0.1], Mh=1)
    booster = cde.PITBoosting(base, correction, K=2).fit(x, y)
    cdes, y_points = booster.predict(x[:20], 50)
    assert torch.allclose(torch.trapz(cdes, y_points.flatten(), dim=1), torch.ones(20))
    # without correction, the prediction is the average of the base fold models
    booster.lambda_ = 0.
    p0 = torch.stack([m.predict(x[:20], 50)[0] for m in booster.base_models]).mean(dim=0)
    assert torch.allclose(booster.predict(x[:20], 50)[0], p0, atol=1e-5)
