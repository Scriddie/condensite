""" Correction of a condensité model's predictive distribution in PIT space. """
import torch
from copy import deepcopy
from sklearn.model_selection import KFold
from . import utils as ut


class PITBoosting:
    """
    Corrects the conditional density p_0 of a base condensité model with the density g of its probability integral
    transforms (PITs) U = F_0(Y | x), i.e. p = p_0 [1 - lambda + lambda g(F_0)]. Base and correction models are
    cross-fitted over K folds.
    """
    def __init__(self, base, correction, K=5, n_grid_u=100, name='pit_boosting'):
        """
        Args:
            base (Condensite): unfitted model of p_0, copied for each fold
            correction (Condensite): unfitted model of g, copied for each fold and fitted with y_range=(0, 1)
            K (int): number of folds
            n_grid_u (int): number of grid points on [0, 1] on which the correction densities are evaluated
            name (str): model name
        """
        self.base = base
        self.correction = correction
        self.K = K
        self.n_grid_u = n_grid_u
        self.name = name
        self.base_models = None  # fitted base model of each fold; set by fit
        self.correction_models = None  # fitted correction model of each fold; set by fit
        self.folds = None  # fold of each training observation; set by fit
        self.PIT = None  # cross-fitted PIT of each training observation; set by fit
        self.lambda_ = None  # weight of the correction; set by fit

    def fit(self, x, y, base_fit_kwargs=None, correction_fit_kwargs=None, n_grid_PIT=100):
        """
        Fit the base model on each fold's complement and compute the PITs of the fold's observations; then fit the
        correction model on each fold's complement of the PITs and choose the weight lambda of the correction.
        Args:
            x (torch.Tensor): (n x d) features
            y (torch.Tensor): (n) labels
            base_fit_kwargs (dict or None): keyword arguments of Condensite.fit for the base models
            correction_fit_kwargs (dict or None): keyword arguments of Condensite.fit for the correction models
            n_grid_PIT (int): number of grid points of the base densities from which the PITs are computed
        """
        base_fit_kwargs = base_fit_kwargs or {}
        correction_fit_kwargs = correction_fit_kwargs or {}
        # random folds; the seed is drawn from torch's generator, so that torch.manual_seed fixes them
        kfold = KFold(self.K, shuffle=True, random_state=int(torch.randint(2**31 - 1, (1,))))
        # fold models share the range of all labels (unless the base fixes its own), hence also their prediction grid
        y_range = (y.min().item(), y.max().item())
        self.folds = torch.empty(len(y), dtype=torch.long)
        self.PIT = torch.empty(len(y))
        self.base_models = []
        for k, (train_idx, test_idx) in enumerate(kfold.split(x)):
            train_idx, test_idx = torch.as_tensor(train_idx), torch.as_tensor(test_idx)
            self.folds[test_idx] = k
            base_k = deepcopy(self.base)
            if base_k.y_range is None:
                base_k.y_range = y_range
            base_k.fit(x[train_idx], y[train_idx], **base_fit_kwargs)
            cdes, y_points = base_k.predict(x[test_idx], n_grid_PIT)
            self.PIT[test_idx] = ut.CDF_at_labels(cdes, y_points, y[test_idx])
            self.base_models.append(base_k)

        # correction models on the same folds; g holds the out-of-fold correction density at each observation's PIT
        g = torch.empty(len(y))
        self.correction_models = []
        for k in range(self.K):
            out = self.folds != k
            correction_k = deepcopy(self.correction)
            correction_k.y_range = (0., 1.)
            correction_k.fit(x[out], self.PIT[out], **correction_fit_kwargs)
            g_grid, u_points = correction_k.predict(x[~out], self.n_grid_u)
            g[~out] = ut.density_at_labels(g_grid, u_points, self.PIT[~out])
            self.correction_models.append(correction_k)
        self.lambda_ = self.choose_lambda(g)
        return self

    @staticmethod
    def choose_lambda(g, tol=1e-6):
        """
        Weight lambda in [0, 1] maximizing the out-of-fold log-likelihood sum_i log(1 - lambda + lambda g_i) of the
        corrected density (up to a term independent of lambda). Its derivative sum_i (g_i - 1)/(1 - lambda + lambda g_i)
        is decreasing in lambda, so the maximizer is found by bisection.
        Args:
            g (torch.Tensor): (n) out-of-fold correction densities at the PITs
            tol (float): tolerance on lambda
        """
        g = g.double()
        def derivative(lam):
            return ((g - 1) / (1 - lam + lam*g)).sum()
        if derivative(0.) <= 0:
            return 0.
        if derivative(1.) >= 0:
            return 1.
        lo, hi = 0., 1.
        while hi - lo > tol:
            mid = (lo + hi)/2
            lo, hi = (mid, hi) if derivative(mid) > 0 else (lo, mid)
        return (lo + hi)/2

    def predict(self, x, n_grid):
        """
        Predict corrected cdes for x on a grid of n_grid points over the range of the training labels, which all fold
        models share (c.f. Condensite.predict).
        Args:
            x (torch.Tensor): features
            n_grid (int): number of grid points for prediction
        """
        # correction density g on [0, 1], averaged over the fold models
        g_preds = [m.predict(x, self.n_grid_u) for m in self.correction_models]
        g_grid = torch.stack([g for g, _ in g_preds]).mean(dim=0)
        u_points = g_preds[0][1]
        # corrected density of each base fold model, p_0 [1 - lambda + lambda g(F_0)], averaged over the folds
        cdes = 0.
        for base_k in self.base_models:
            p0, y_points = base_k.predict(x, n_grid)
            g_at_F0 = ut.density_at_labels(g_grid, u_points, ut.CDF_from_pred(p0, y_points))
            cdes = cdes + p0*(1 - self.lambda_ + self.lambda_*g_at_F0)
        cdes = cdes / self.K
        # normalize with the trapezoid rule; this only corrects the discretization error
        return self.base_models[0].make_density(cdes, y_points), y_points
