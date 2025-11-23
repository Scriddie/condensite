""" Use a neural network (NN) as predictor in Condensité. """
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from condensite import CondensitePredictor, concat_collate


class NNPredictor(nn.Module, CondensitePredictor):
    def __init__(self, 
        n_in, n_hidden,  # structure params
        batch_size, lr, weight_decay # training params
    ):
        super().__init__()
        # define device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # network architecture
        self.network = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        self.network.apply(self.init_weights)
        # set training params
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        # define optimizer
        self.optimizer = AdamW(params=self.network.parameters(), 
                               lr=lr, weight_decay=weight_decay)

    def device(self):
        return self._device

    def init_weights(self, m):
        # initialization to match ReLU activations
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, features):
        return self.network(features).ravel()

    def predict(self, features):
        """ Predict for given features. """
        # labels don't matter here, set to zeros
        self.eval()
        dataset = TensorDataset(features, torch.zeros(len(features)))
        loader = DataLoader(dataset, self.batch_size, shuffle=False)
        predictions = torch.empty(len(features), device=self.device())
        start = 0
        with torch.no_grad():
            for feature_batch, _ in loader:
                feature_batch = feature_batch.to(self.device())
                batch_preds = self.forward(feature_batch).detach()
                predictions[start:(start+len(batch_preds))] = batch_preds.flatten()
                start += len(batch_preds)
        return predictions

    def fit(self, dataset):
        """
        Perform a fit on the features and targets in the dataset passed.
        """
        self.train()
        ## batching
        train_DL = DataLoader(dataset, self.batch_size, shuffle=True, collate_fn=concat_collate)
        # training
        epoch_losses = []
        loss_fn = nn.MSELoss()
        # loss_fn = nn.BCEWithLogitsLoss()
        for feature_batch, target_batch in train_DL:
            # move to device
            feature_batch = feature_batch.to(self.device())
            target_batch = target_batch.to(self.device())
            # compute MSE and update weights
            self.optimizer.zero_grad()
            pred = self.forward(feature_batch)
            loss = loss_fn(target_batch, pred)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
        # results of epoch
        return sum(epoch_losses) / len(epoch_losses)
