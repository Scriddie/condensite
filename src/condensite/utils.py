import torch


def ISE_from_pred(cdes_grid, y_points, y):
    """
    Separate this from the predict part since sometimes we'd like to load saved predictions.
    Args:
        cdes_grid (torch.Tensor): conditional density estimates over a grid
        y_points (torch.Tensor): the grid points
        y (torch.Tensor): the labels
    """
    ## ISE term 1: volume under squared curve for each X point
    a2 = torch.mean(torch.trapz(cdes_grid**2, y_points.flatten(), dim=1))
    ## ISE term 2: density height at the actual data points.
    # choose the grid value closest to a real value as nearest neighbor
    nn = torch.argmin(torch.abs(y.view(-1,1)-y_points.view(1,-1)), axis=1)
    cdes = cdes_grid[range(len(y)), nn]
    ab = torch.mean(cdes)
    # combine first two terms
    return a2 - 2*ab


def ISE(model, x, y, n_grid):
    """
    ISE approximation up to a constant; on a grid of length n_grid.
    Args:
        model (object): one of our CDE models
        x (torch.Tensor): (n x d)
        y (torch.Tensor): (n x 1)
        n_grid (int): number of grid points for evaluation
    """
    cdes_grid, y_points = model.predict(x, n_grid)
    return ISE_from_pred(cdes_grid, y_points, y)


def train_test_split(x, y, train_frac):
    """ 
    Akin to sklearn.model_selection.train_test_split.
    Args:
        x (torch.Tensor): features
        y (torch.Tensor): labels
        train_frac (float): fraction to be used for training
    """
    split_idx = int(train_frac*len(y))
    mask = torch.randperm(len(y))
    train_mask, test_mask = mask[:split_idx], mask[split_idx:]
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    return x_train, x_test, y_train, y_test