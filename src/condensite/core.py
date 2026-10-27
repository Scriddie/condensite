import os
import pickle
import torch
import torch.nn as nn
import torch.distributions as dist
from copy import deepcopy
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from . import utils as ut
import warnings
import logging
logger = logging.getLogger(__name__)


class RepeatDataset(torch.utils.data.Dataset):
    """ 
    Custom dataset for the auxiliary condensite samples (keeps RAM usage in check). 
    """
    def __init__(self, x, y_aux, targets, M):
        """
        Args:
            x (torch.Tensor): the features to be repeated
            y_aux (torch.Tensor): the auxiliary labels
            targets (torch.Tensor): target values
            M (int): number of repetitions
        """
        self.n = len(x)
        # assert dimensions
        assert len(y_aux) == self.n*M
        assert len(targets) == self.n*M
        # parameters
        self.x = x  # (Nxd)
        self.y_aux = y_aux  # (NM)
        self.targets = targets  # (NM)
        self.M = M  # scalar
        # precompute indices
        self.j_map = torch.arange(self.n * self.M) // self.M

    def __len__(self):
        return self.n * self.M

    def __getitem__(self, i):
        return self.x[self.j_map[i]], self.y_aux[i], self.targets[i]


def concat_collate(batch):
    """
    To be used in a DataLoader, as collate_fn for RepeatDataset.
    Args:
        batch (torch.Tensor): batch of features, auxiliary labels, and targets.
    """
    x, y_aux, targets = zip(*batch)  # tuples of tensors
    x = torch.stack(x)            # shape: (B, d)
    y_aux = torch.stack(y_aux).reshape(-1, 1)  # shape: (B, 1)
    features = torch.cat([x, y_aux], dim=1)  # shape: (B, d+1)
    targets = torch.stack(targets)
    return features, targets


class CondensitePredictor(ABC):
    """ Defines the interface predictors have to implement. """
    def __init__(self, device):
        pass

    @abstractmethod
    def device(self):
        """
        Return device (would typically be "cpu" or "cuda").
        """
        return self._device

    @abstractmethod
    def to(self, device):
        """ 
        Port to device if applicable. Otherwise, simply return self. 
        Args:
            device (str): output of self.device()
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: torch.Tensor):
        """ 
        Transform features into targets. 
        Args:
            features (torch.Tensor): feature values
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """ 
        Predict from given features.
        Args:
            features (torch.Tensor): feature values
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, dataset: RepeatDataset) -> torch.Tensor:
        """ 
        Fit the predictor to data. 
        Args:
            dataset (RepeatDataset): virtual dataset of features and targets
        Returns:
            loss (torch.Tensor)
        """
        raise NotImplementedError


class Condensite:
    """ 
    Fits CondensitePredictor objects to training data from a conditional density and provides a prediction function for the fitted object. 
    """

    def __init__(self, predictor: CondensitePredictor, 
                 h: list[float],
                 c: float,
                 name: str = 'condensite',
                 prediction_batch_size: int | None = None,
                 reflect: bool = True,
                 y_range: tuple[float, float] | None = None):
        """
        Args:
            predictor (CondensitePredictor): conditional density estimator.
            h (list of float): candidate kernel bandwidths on the [0,1]-rescaled y; with several, fit selects the one with the lowest validation ISE (requires val_frac > 0).
            c (float): M*h; bandwidth h uses M = round(c/h) auxiliary samples per observation.
            name (str): model name
            prediction_batch_size (int): batch size when predicting (to limit RAM usage)
            reflect (bool): reflect observations at the boundaries of [0,1] when computing targets (c.f. compute_targets).
            y_range (tuple or None): fixed (min, max) for the min-max scaling of y, e.g. for labels with known support; None uses the range of the training labels.
        """
        self.name = name
        self.device = predictor.device()
        self.predictor = predictor.to(self.device)
        self.h = h
        self.best_h = None  # set by fit
        self.c = c
        self.M_best_h = None  # number of auxiliary samples of the selected h; set by fit
        self.reflect = reflect
        self.y_range = y_range
        # parameters for standardization; crucial for training stability
        self.x_mean = None
        self.x_scale = None
        self.y_min = None
        self.y_max = None
        self.t_mean = None
        self.t_scale = None
        self.prediction_batch_size=prediction_batch_size

    def fit(self, x, y, *, val_frac=0., n_grid_ISE=100, verbose=0, epochs=None, early_stopping_epochs=None):
        """
        Train the predictor on the data. With val_frac > 0, the validation ISE selects h (and the best epoch). Predictors
        fitted in a single pass (epochs None) are then refitted on all data with the selected h; with epochs, the
        selected model is kept, since its number of epochs was chosen for the training share only.
        Args:
            x (tensor): (n x d)
            y (tensor): (n x 1)
            val_frac (float): validation data frac in order to compute ISE for selection of h, best epoch selection and early stopping
            n_grid_ISE (int): number of grid points for the validation ISE.
            verbose (int): verbosity of training.
            epochs (int or None): train epochs; None for predictors fitted in a single pass (e.g. trees)
            early_stopping_epochs (int or None): stop after this many epochs without improvement of the validation ISE (requires val_frac > 0); None means no early stopping
        Returns:
            run summary; its validation ISE is that of the selected h before any refit
        """
        if not 0. <= val_frac < 1.:
            raise ValueError("val_frac must be in [0,1).")
        if (early_stopping_epochs is not None) and (val_frac == 0.):
            raise ValueError("early stopping requires a validation set (val_frac > 0).")
        if (len(self.h) > 1) and (val_frac == 0.):
            raise ValueError("selecting among several h requires a validation set (val_frac > 0).")

        # single-pass predictors are refitted on all data after the selection; decided before epochs defaults to 1
        refit = (val_frac > 0.) and (epochs is None)
        n_epochs = 1 if epochs is None else epochs
        unfitted_predictor = deepcopy(self.predictor) if refit else None

        # train-validation split, then selection of h on the validation set
        x_train, x_val, y_train, y_val = ut.train_test_split(x, y, val_frac)
        best, val_ise_by_h = self._fit_candidates(x_train, y_train, x_val, y_val, self.h,
                                                  val_frac, n_grid_ISE, verbose, n_epochs, early_stopping_epochs)
        self.best_h = best['h']
        self.M_best_h = best['M']
        self.val_ise = best['val_ise']
        # refit with the selected h on all data
        if refit:
            self.predictor = unfitted_predictor
            best, _ = self._fit_candidates(x, y, None, None, [self.best_h],
                                           0., n_grid_ISE, verbose, 1, None)

        # return run summary
        return {'h': self.best_h,
                'M': self.M_best_h,
                'epochs': best['epoch'],  # note that count starts from 0; this is total epochs run, not epoch of best model.
                'val_ISE': self.val_ise,
                'val_ISE_by_h': val_ise_by_h,
                'refit': refit}

    def _fit_candidates(self, x_train, y_train, x_val, y_val, hs, val_frac, n_grid_ISE, verbose, epochs, early_stopping_epochs):
        """
        Fit the predictor for each bandwidth candidate in hs on the training data, and keep the one with the lowest
        validation ISE in self.predictor, self.t_mean and self.t_scale (c.f. fit for the arguments). Feature scaling and
        auxiliary samples are shared by all candidates.
        Returns:
            the selected candidate (dict with h, M, val_ise, epoch, ...), validation ISE of each candidate
        """
        # sample auxiliary features; candidate h uses the first M = round(c/h) samples of each observation
        Ms = [max(1, round(self.c / h)) for h in hs]
        sample_aux, aux_mean, aux_scale = self.aux_sampler()
        y_aux_all = sample_aux((len(y_train), max(Ms)))
        # feature-scaling parameters
        self.set_sample_scaling_params(x_train, y_train)
        # min-max scale y
        y_01 = (y_train-self.y_min)/(self.y_max-self.y_min)
        # standardize
        x_train = self.standardize(x_train, self.x_mean, self.x_scale)

        # fit for each bandwidth candidate and keep the one with the lowest validation ISE
        initial_predictor = self.predictor
        val_ise_by_h, best = {}, None
        for h, M in zip(hs, Ms):
            if len(hs) > 1:
                self.predictor = deepcopy(initial_predictor)
            # auxiliary samples in blocks of M per observation (c.f. RepeatDataset)
            y_aux_train = y_aux_all[:, :M].reshape(-1, 1)
            y_aux_std = (y_aux_train - aux_mean) / aux_scale
            # training targets
            targets = self.compute_targets(y_01, y_aux_train, h, M)
            # label-scaling parameters (they depend on h; overwrite at the end using the best selected h)
            self.set_target_scaling_params(targets)
            targets = self.standardize(targets, self.t_mean, self.t_scale)
            # create the condensite RepeatDataset for training
            train_dataset = RepeatDataset(x=x_train,
                                          y_aux=y_aux_std,
                                          targets=targets,
                                          M=M)
            # training
            val_ise, epoch = self._train_epochs(
                train_dataset, x_val, y_val, 
                val_frac, n_grid_ISE,verbose, epochs, early_stopping_epochs
            )
            val_ise_by_h[h] = val_ise
            if (verbose > 0) and (len(hs) > 1):
                logger.info(f"h={h}: validation ISE {val_ise:.4f}")
            if (best is None) or (val_ise < best['val_ise']):
                best = {'h': h, 'M': M, 'val_ise': val_ise, 'epoch': epoch, 'predictor': self.predictor,
                        't_mean': self.t_mean, 't_scale': self.t_scale}

        # select best bandwidth
        self.predictor = best['predictor']
        self.t_mean, self.t_scale = best['t_mean'], best['t_scale']
        return best, val_ise_by_h

    def _train_epochs(self, train_dataset, x_val, y_val, val_frac, n_grid_ISE, verbose, epochs, early_stopping_epochs):
        """
        Train the predictor for up to the given number of epochs (c.f. fit for the arguments). With a validation set, the
        epoch with the lowest validation ISE is kept in self.predictor, and training stops early if requested.
        Returns:
            validation ISE of the kept epoch (inf without validation set), index of the last epoch run
        """
        best_val_ise = float('inf')
        best_epoch_predictor = None
        for epoch in range(epochs):
            # call the predictor's fit method
            mse = self.predictor.fit(train_dataset)
            if mse is not None:
                mse = round(mse, 4)

            # fit on validation set
            if val_frac > 0.:
                # compute ISE on validation set
                val_ise = ut.ISE(self, x_val, y_val, n_grid_ISE)
                val_output = f" validation ISE: {val_ise:.4f}"
            else:
                val_ise = float('inf')
                val_output = ""

            # relay training output
            if verbose > 0:
                if epochs > 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}, train mse: {mse}"+ val_output)
                else:
                    logger.info(f"Train mse: {mse}" + val_output)

            # Early stopping
            if val_frac > 0.:
                if val_ise < best_val_ise:  # expected to apply in first epoch
                    best_val_ise = val_ise
                    best_epoch_predictor = deepcopy(self.predictor)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if (early_stopping_epochs is not None) and (epochs_without_improvement >= early_stopping_epochs):
                        if verbose>0:
                            logger.info(f'Early stopping at epoch {epoch + 1}')
                        break
            else:
                best_epoch_predictor = self.predictor

        # select best network
        self.predictor = best_epoch_predictor
        return best_val_ise, epoch

    def set_sample_scaling_params(self, x_train, y_train):
        """
        Store values for scaling the observations.
        Args:
            x_train (torch.Tensor): training set feature values
            y_train (torch.Tensor): training set labels
        """
        # standard scaling for x
        self.x_mean = x_train.mean(dim=0).to(self.device)
        self.x_scale = x_train.std(dim=0).to(self.device)
        # for min-max scaling of y (observed range unless a fixed range is given)
        if self.y_range is None:
            self.y_min = y_train.min()
            self.y_max = y_train.max()
        else:
            self.y_min, self.y_max = self.y_range
            if (y_train.min() < self.y_min) or (y_train.max() > self.y_max):
                raise ValueError("y_train lies outside y_range.")

    def set_target_scaling_params(self, t_train):
        """
        Store mean and standard deviation of targets for scaling.
        Args:
            t_train (torch.Tensor): targets derived from auxiliary samples
        """
        self.t_mean = t_train.mean(dim=0).to(self.device)
        self.t_scale = t_train.std(dim=0).to(self.device)

    def standardize(self, x, mean, scale):
        """
        Standardize features.
        Args:
            x: features,
            mean (torch.Tensor): mean values for standardization
            scale (torch.Tensor): standard deviations for standardization
        """
        if not x.is_cuda:
            mean, scale = mean.cpu(), scale.cpu()
        return (x - mean) / (scale + 1e-8)
    
    @staticmethod
    def aux_sampler():
        """
        Auxiliary samples on the rescaled support [0,1] of y: a function drawing samples of a given shape, and the mean
        and standard deviation of the samples (used for standardization).
        """
        return (lambda shape: torch.empty(shape).uniform_(0, 1)), 0.5, 12**-0.5

    def compute_targets(self, y_01, y_aux, h, M):
        """
        Compute targets via $K_h(Y_i-Y_{im})$. If self.reflect, each observation is additionally reflected at 0 and 1, i.e. $K_h(Y_i-Y_{im}) + K_h(-Y_i-Y_{im}) + K_h((2-Y_i)-Y_{im})$, so that no kernel mass is lost at the boundaries of the rescaled support [0,1].
        Args:
            y_01 (torch.Tensor): observed y rescaled to [0,1]
            y_aux (torch.Tensor): auxiliary labels for target construction, in blocks of M per observation
            h (float): kernel bandwidth
            M (int): number of auxiliary samples per observation
        """
        # prepare for grid
        y_bar = y_01.repeat_interleave(M, dim=0).view(-1,1)
        # compute auxiliary densities
        Kh = dist.MultivariateNormal(loc=torch.zeros(1),
                                     covariance_matrix=torch.eye(1) * h**2)
        targets = torch.exp(Kh.log_prob(y_bar - y_aux))
        if self.reflect:
            # mirror images of each observation at 0 and 1
            targets = (targets
                       + torch.exp(Kh.log_prob(-y_bar - y_aux))
                       + torch.exp(Kh.log_prob((2 - y_bar) - y_aux)))
        return targets

    def make_density(self, cdes, y):
        """
        Make sure estimated densities are positive and add up to 1 under numerical integration.
        Args:
            cdes (torch.Tensor): (n x n_grid) density estimates
        """
        # force positive
        cdes = torch.where(cdes<0, 0, cdes)
        area = torch.trapz(cdes, y.flatten(), dim=1).reshape(-1, 1)
        # if we predict zero density everywhere, assume a uniform distribution
        empty_pred = torch.argwhere(area.ravel() == 0)
        cdes[empty_pred, :] = 1/cdes.shape[1]
        area[empty_pred] = 1
        # normalize density
        cdes = cdes / area
        return cdes

    def predict(self, x, n_grid):
        """
        Predict cdes for x at a grid across the range of y seen at training; batching to limit RAM usage.
        Args:
            x (torch.Tensor): features
            n_grid (int): number of grid points for prediction
        """
        # create grid on [0,1] scale using RepeatDataset
        x = self.standardize(x, self.x_mean, self.x_scale)
        y_points_01 = torch.linspace(0, 1, n_grid).view(-1,1)        
        grid_y = y_points_01.repeat(len(x), 1)
        _, aux_mean, aux_scale = self.aux_sampler()
        grid_y = (grid_y - aux_mean) / aux_scale
        dataset = RepeatDataset(
            x=x, 
            y_aux=grid_y, 
            targets=torch.zeros_like(grid_y),  # not actually needed here 
            M=n_grid
        )

        # let predictor predict and scale up
        if self.prediction_batch_size is None:
            batch_size = len(grid_y)  # single batch
        else:
            batch_size = self.prediction_batch_size
        predict_DL = DataLoader(
            dataset, 
            batch_size, 
            shuffle=False, 
            collate_fn=concat_collate
        )
        cdes = torch.empty(len(x)*n_grid, device="cpu")
        start = 0
        for feature_batch, _ in predict_DL:
            feature_batch = feature_batch.to(self.device)
            batch_preds = self.predictor.predict(feature_batch)
            batch_preds = batch_preds*self.t_scale+self.t_mean
            cdes[start:(start+len(batch_preds))] = batch_preds.flatten().cpu()
            start += len(batch_preds)
        cdes = cdes.view(len(x), n_grid)

        # min-max upscaling; we want a density on the original scale
        y_points = y_points_01 * (self.y_max - self.y_min) + self.y_min
        cdes = self.make_density(cdes, y_points)        
        return cdes, y_points

    def save(self, path):
        """ Save the model. """
        with open(os.path.join(path, f"{self.name}.pkl"), "wb") as f:
            pickle.dump(self, f)