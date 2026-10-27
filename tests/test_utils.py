import torch
import condensite as cde

N = torch.distributions.Normal(0., 1.)


def normal_pred(n_rows=3, n_grid=50, dtype=torch.float32):
    """ Standard normal density on a grid over [-4, 4], normalized by the trapezoid rule as in make_density. """
    y_points = torch.linspace(-4, 4, n_grid, dtype=dtype).view(-1, 1)
    cdes = N.log_prob(y_points.flatten().float()).exp().to(dtype).repeat(n_rows, 1)
    cdes = cdes / torch.trapz(cdes, y_points.flatten(), dim=1).view(-1, 1)
    return cdes, y_points


def test_CDF_matches_normal():
    cdes, y_points = normal_pred()
    y = torch.tensor([-1.3, 0.3, 2.1])
    F = cde.utils.CDF_at_labels(cdes, y_points, y)
    assert torch.allclose(F, N.cdf(y), atol=1e-3)


def test_CDF_outside_grid():
    cdes, y_points = normal_pred()
    F = cde.utils.CDF_at_labels(cdes, y_points, torch.tensor([-5., 4., 7.]))
    assert torch.allclose(F, torch.tensor([0., 1., 1.]), atol=1e-6)


def test_CDF_at_grid_points():
    cdes, y_points = normal_pred()
    grid = y_points.flatten().repeat(len(cdes), 1)
    F_grid = cde.utils.CDF_from_pred(cdes, y_points)
    assert torch.allclose(cde.utils.CDF_at_labels(cdes, y_points, grid), F_grid, atol=1e-6)
    assert torch.allclose(F_grid[:, -1], torch.ones(len(cdes)), atol=1e-6)


def test_density_interpolation():
    cdes, y_points = normal_pred()
    grid = y_points.flatten()
    # at grid points, between two of them, and outside the grid; one row of points per density (n x m)
    y = torch.stack([grid[:5], (grid[:5] + grid[1:6])/2, torch.tensor([-9., -5., 4.5, 5., 9.])])
    p = cde.utils.density_at_labels(cdes, y_points, y)
    assert torch.allclose(p[0], cdes[0, :5])
    assert torch.allclose(p[1], (cdes[1, :5] + cdes[1, 1:6])/2)
    assert torch.equal(p[2], torch.zeros(5))


def test_density_is_derivative_of_CDF():
    cdes, y_points = normal_pred(dtype=torch.float64)
    y, eps = torch.tensor([-2.17, 0.03, 1.55], dtype=torch.float64), 1e-6
    dF = (cde.utils.CDF_at_labels(cdes, y_points, y + eps) - cde.utils.CDF_at_labels(cdes, y_points, y - eps))/(2*eps)
    assert torch.allclose(dF, cde.utils.density_at_labels(cdes, y_points, y), atol=1e-6)
