import numpy as np
import pytest
import torch
import condensite as cde
from sklearn.ensemble import HistGradientBoostingRegressor


def test_import():
    import condensite
    assert condensite is not None


class TreePredictor(cde.CondensitePredictor):
    """ Small sklearn predictor for fast tests (c.f. README). """
    def __init__(self):
        self.tree = HistGradientBoostingRegressor(max_iter=20, early_stopping=False)

    def device(self):
        return "cpu"

    def to(self, device):
        return self

    def forward(self, x):
        return self.tree.predict(x)

    def predict(self, x):
        return torch.tensor(self.forward(x), dtype=torch.float32)

    def fit(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.n*dataset.M, collate_fn=cde.concat_collate)
        x, y = (t.numpy() for t in next(iter(loader)))
        self.tree.fit(x, y)
        return np.mean(np.square(self.tree.predict(x) - y))


def sample_data(n=300):
    x = torch.randn(n, 3)
    return x, x[:, 0] + torch.randn(n)


def test_float_h_raises():
    x, y = sample_data()
    with pytest.raises(TypeError):
        cde.Condensite(TreePredictor(), h=0.05, c=0.5).fit(x, y)


def test_several_h_without_validation_raises():
    x, y = sample_data()
    with pytest.raises(ValueError):
        cde.Condensite(TreePredictor(), h=[0.01, 0.1], c=0.5).fit(x, y)


def test_h_selection():
    torch.manual_seed(0)
    x, y = sample_data()
    model = cde.Condensite(TreePredictor(), h=[0.005, 0.05, 0.2], c=0.5)
    summary = model.fit(x, y, val_frac=0.2)
    ises = summary['val_ISE_by_h']
    assert list(ises) == [0.005, 0.05, 0.2]
    assert model.best_h == min(ises, key=ises.get) == summary['h']
    assert model.val_ise == ises[model.best_h]
    assert model.M_best_h == summary['M'] == round(0.5 / model.best_h)


class RecordingPredictor(TreePredictor):
    """ Records the auxiliary samples it is trained on (shared across the copies made for each h). """
    seen = []

    def fit(self, dataset):
        RecordingPredictor.seen.append((dataset.M, dataset.y_aux.clone()))
        return super().fit(dataset)


def test_aux_samples_shared_across_h():
    x, y = sample_data()
    RecordingPredictor.seen.clear()
    cde.Condensite(RecordingPredictor(), h=[0.05, 0.2], c=0.5).fit(x, y, val_frac=0.2)
    (M1, u1), (M2, u2) = RecordingPredictor.seen
    assert (M1, M2) == (10, 2)
    # the second candidate uses the first M2 auxiliary samples of each observation
    assert torch.equal(u2.view(-1, M2), u1.view(-1, M1)[:, :M2])
