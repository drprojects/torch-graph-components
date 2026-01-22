from typing import TYPE_CHECKING

from . import _deps

torch = _deps.torch
torch_scatter = _deps.torch_scatter

if TYPE_CHECKING:
    import torch as torch
    import torch_scatter as torch_scatter

__all__ = [
    'scatter_mean_weighted',
    'knn_brute_force',
]


def scatter_mean_weighted(x, idx, w, dim_size=None):
    """Helper for scatter_mean with weights"""
    assert w.ge(0).all(), \
        "Only positive weights are accepted"
    assert w.dim() == idx.dim() == 1, \
        "w and idx should be 1D torch.Tensor"
    assert x.shape[0] == w.shape[0] == idx.shape[0], \
        "Only supports weighted mean along the first dimension"

    # Concatenate w and x in the same tensor to only call scatter once
    x = x.view(-1, 1) if x.dim() == 1 else x
    w = w.view(-1, 1).float()
    wx = torch.cat((w, x * w), dim=1)

    # Scatter sum the wx tensor to obtain
    wx_segment = torch_scatter.scatter_sum(wx, idx, dim=0, dim_size=dim_size)

    # Extract the weighted mean from the result
    w_segment = wx_segment[:, 0]
    x_segment = wx_segment[:, 1:]
    w_segment[w_segment == 0] = 1
    mean_segment = x_segment / w_segment.view(-1, 1)

    return mean_segment


def knn_brute_force(
        x_search,
        x_query,
        k,
        r_max=1,
        batch_search=None,
        batch_query=None):
    """Search k-NN of x_query inside x_search, within radius `r_max`.
    Optionally, passing `batch_search` and `batch_query` will ensure the
    neighbor search does not mix up batch items. This is the same as
    `knn_2()` but uses a brute force method, which does not scale.
    """
    assert isinstance(x_search, torch.Tensor)
    assert isinstance(x_query, torch.Tensor)
    assert k >= 1
    assert x_search.dim() == 2
    assert x_query.dim() == 2
    assert x_query.shape[1] == x_search.shape[1]
    assert (batch_search is None) == (batch_query is None)
    assert batch_search is None or batch_search.shape[0] == x_search.shape[0]
    assert batch_query is None or batch_query.shape[0] == x_query.shape[0]

    # To take the batch into account, we add an offset to the Z
    # coordinates. The offset is designed so that any points from two
    # batch different batch items are separated by at least `r_max + 1`
    batch_search_offset = 0
    batch_query_offset = 0
    if batch_search is not None:
        hi = max(x_search[:, 2].max(), x_query[:, 2].max())
        lo = min(x_search[:, 2].min(), x_query[:, 2].min())
        z_offset = hi - lo + r_max + 1
        batch_search_offset = torch.zeros_like(x_search)
        batch_search_offset[:, 2] = batch_search * z_offset
        batch_query_offset = torch.zeros_like(x_query)
        batch_query_offset[:, 2] = batch_query * z_offset

    # Data initialization
    xyz_query = (x_query + batch_query_offset)
    xyz_search = (x_search + batch_search_offset)

    # Brute force compute all distances and neighbors for all query
    # points
    distances = (xyz_search.unsqueeze(0) - xyz_query.unsqueeze(1)).norm(dim=2)
    distances, neighbors = distances.sort(dim=1)
    distances = distances[:, :k]
    neighbors = neighbors[:, :k]
    mask = distances > r_max
    distances[mask] = -1
    neighbors[mask] = -1

    return neighbors, distances
