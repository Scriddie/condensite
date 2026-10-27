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


def _grid_cells(y_points, y):
    """
    Grid cell [grid[l], grid[l+1]] containing each point of y; points outside the grid fall into the first or last cell.
    Args:
        y_points (torch.Tensor): the n_grid grid points
        y (torch.Tensor): (n x m) points
    Returns:
        flattened grid, cell indices l (n x m), cell widths (n x m)
    """
    grid = y_points.flatten()
    l = (torch.searchsorted(grid, y.contiguous(), right=True) - 1).clamp(0, len(grid) - 2)
    return grid, l, grid[l+1] - grid[l]


def CDF_from_pred(cdes_grid, y_points):
    """
    CDF at the grid points of a density predicted on a grid: the cumulative trapezoid rule, i.e. the exact CDF of the
    piecewise-linear interpolation of the density. Since Condensite.make_density normalizes with the same rule, the CDF
    is 0 at the first and 1 at the last grid point.
    Args:
        cdes_grid (torch.Tensor): (n x n_grid) conditional density estimates over a grid
        y_points (torch.Tensor): the n_grid grid points
    Returns:
        (n x n_grid) CDF values at the grid points
    """
    F = torch.cumulative_trapezoid(cdes_grid, y_points.flatten(), dim=1)
    # clamping removes rounding errors beyond [0, 1]
    return torch.cat([F.new_zeros(len(F), 1), F], dim=1).clamp(0, 1)


def CDF_at_labels(cdes_grid, y_points, y):
    """
    CDF of each row's predicted density at its own points, e.g. F(y_i | x_i) for probability integral transforms.
    Exact for the piecewise-linear interpolation of the density (c.f. CDF_from_pred), so that its derivative is
    density_at_labels; 0 below and 1 above the grid.
    Args:
        cdes_grid (torch.Tensor): (n x n_grid) conditional density estimates over a grid
        y_points (torch.Tensor): the n_grid grid points
        y (torch.Tensor): (n) or (n x m) points, row i evaluated under the density of row i
    Returns:
        CDF values of the same shape as y
    """
    y2 = y.reshape(len(cdes_grid), -1)
    grid, l, delta = _grid_cells(y_points, y2)
    F = CDF_from_pred(cdes_grid, y_points)
    # clamping the offset within the cell gives F = 0 below and F = 1 above the grid
    s = torch.minimum((y2 - grid[l]).clamp(min=0), delta)
    p_l, p_r = cdes_grid.gather(1, l), cdes_grid.gather(1, l+1)
    return (F.gather(1, l) + p_l*s + (p_r - p_l)*s**2/(2*delta)).clamp(0, 1).reshape(y.shape)


def density_at_labels(cdes_grid, y_points, y):
    """
    Density of each row's prediction at its own points, by linear interpolation between grid points (c.f.
    CDF_from_pred); 0 outside the grid.
    Args:
        cdes_grid (torch.Tensor): (n x n_grid) conditional density estimates over a grid
        y_points (torch.Tensor): the n_grid grid points
        y (torch.Tensor): (n) or (n x m) points, row i evaluated under the density of row i
    Returns:
        density values of the same shape as y
    """
    y2 = y.reshape(len(cdes_grid), -1)
    grid, l, delta = _grid_cells(y_points, y2)
    p_l, p_r = cdes_grid.gather(1, l), cdes_grid.gather(1, l+1)
    p = p_l + (p_r - p_l)*(y2 - grid[l])/delta
    p = torch.where((y2 < grid[0]) | (y2 > grid[-1]), 0., p)
    return p.reshape(y.shape)


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


def train_test_split(x, y, val_frac):
    """
    Akin to sklearn.model_selection.train_test_split.
    Args:
        x (torch.Tensor): features
        y (torch.Tensor): labels
        val_frac (float): fraction to be held out (not used for training)
    """
    split_idx = len(y) - int(val_frac*len(y))
    mask = torch.randperm(len(y))
    train_mask, test_mask = mask[:split_idx], mask[split_idx:]
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    return x_train, x_test, y_train, y_test