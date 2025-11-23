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

    def __init__(self, predictor, h, M, 
                 name='condensite', 
                 prediction_batch_size=None):
        """
        Args:
            predictor (CondensitePredictor): conditional density estimator.
            h (int): sharpness; for transformation into regression task.
            M (int): number of auxiliary samples.
            name (str): model name
            prediction_batch_size (int): batch size when predicting (to limit RAM usage)
        """
        self.name = name
        self.device = predictor.device()
        self.predictor = predictor.to(self.device)
        self.h = h
        self.M = M
        # parameters for standardization; crucial for training stability
        self.x_mean = None
        self.x_scale = None
        self.y_min = None
        self.y_max = None
        self.y_aux_mean = None
        self.y_aux_scale = None
        self.t_mean = None
        self.t_scale = None
        self.prediction_batch_size=prediction_batch_size

    def fit(self, x, y, train_frac, n_grid_ISE, verbose=0, epochs=None, patience=None):
        """
        Train the predictor on the data.
        Args:
            x (tensor): (n x d)
            y (tensor): (n x 1)
            train_frac (float): fraction fo data to use for training, rest is used for cross-validation.
            n_grid_ISE: number of grid points for ISE in cross-validation.
            verbose (int): verbosity of training.
            epochs (int or None): train epochs, None if irrelevant
            patience (int or None): early stopping count, None if irrelevant
        """
        # check if epochs relevant
        if epochs is None:
            epochs = 1
        # train-CV split index
        x_train, x_val, y_train, y_val = ut.train_test_split(x, y, train_frac)
        # sample auxiliary features
        y_aux_train = self.sample_y_aux(y_train)        
        # feature-scaling parameters
        self.set_sample_scaling_params(x_train, y_train, y_aux_train)
        # min-max scale y
        y_01 = (y_train-self.y_min)/(self.y_max-self.y_min)
        # training targets
        targets = self.compute_targets(y_01, y_aux_train)
        # label-scaling parameters
        self.set_target_scaling_params(targets)
        # standardize
        x_train = self.standardize(x_train, self.x_mean, self.x_scale)
        y_aux_train = self.standardize(y_aux_train, self.y_aux_mean, self.y_aux_scale)
        targets = self.standardize(targets, self.t_mean, self.t_scale)
        
        # create the condensite RepeatDataset for training
        train_dataset = RepeatDataset(x=x_train, 
                                      y_aux=y_aux_train, 
                                      targets=targets, 
                                      M=self.M)

        # training
        best_val_ise = float('inf')
        best_epoch_predictor = None
        for epoch in range(epochs):
            # call the predictor's fit method
            mse = self.predictor.fit(train_dataset)
            # compute ISE on validation set
            val_ise = ut.ISE(self, x_val, y_val, n_grid_ISE)
            if verbose > 0:
                if epochs > 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train mse: {mse:.4f}, Validation ISE: {val_ise:.4f}")
                else:
                    logger.info(f"Train mse: {mse:.4f}, Validation ISE: {val_ise:.4f}")
            # Early stopping
            if val_ise < best_val_ise:
                best_val_ise = val_ise
                best_epoch_predictor = deepcopy(self.predictor)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose>0:
                        logger.info(f'Early stopping at epoch {epoch + 1}')
                    break
        
        # select best network
        self.predictor = best_epoch_predictor
        self.ise = best_val_ise
        
        # return run summary
        return {'h': self.h,
                'M': self.M,
                'epochs': epoch,
                'ISE': best_val_ise}

    def set_sample_scaling_params(self, x_train, y_train, y_aux_train):
        """
        Store values for scaling the observations and auxiliary samples.
        Args:
            x_train (torch.Tensor): training set feature values
            y_train (torch.Tensor): training set labels
            y_aux_train (torch.Tensor): auxiliary labels 
        """
        # standard scaling for x
        self.x_mean = x_train.mean(dim=0).to(self.device)
        self.x_scale = x_train.std(dim=0).to(self.device)
        # for min-max scaling of y
        self.y_min = y_train.min()
        self.y_max = y_train.max()
        # standard scaling for auxiliary samples y_aux
        self.y_aux_mean = y_aux_train.mean(dim=0).to(self.device)
        self.y_aux_scale = y_aux_train.std(dim=0).to(self.device)

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
        grid_y = self.standardize(grid_y, self.y_aux_mean, self.y_aux_scale)
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
    
    def sample_y_aux(self, y):
        """
        Sample auxiliary labels.
        Args:
            y (torch.Tensor): the (n) y values
        """
        return torch.empty(len(y)*self.M).uniform_(0, 1).view(-1,1)

    def compute_targets(self, y_01, y_aux):
        """ 
        Compute targets via $K_h(Y_i-Y_{im})$.
        Args:
            y_01 (torch.Tensor): observed y rescaled to [0,1]
            y_aux (torch.Tensor): auxiliary labels for target construction
        """
        # prepare for grid
        y_bar = y_01.repeat_interleave(self.M, dim=0).view(-1,1)
        # compute auxiliary densities
        Kh = dist.MultivariateNormal(loc=torch.zeros(1),
                                     covariance_matrix=torch.eye(1) * self.h**2)
        targets = torch.exp(Kh.log_prob(y_bar - y_aux))
        return targets

    def save(self, path):
        """ Save the model. """
        with open(os.path.join(path, f"{self.name}.pkl"), "wb") as f:
            pickle.dump(self, f)