# Obtained from: https://github.com/HeliosZhao/SHADE/blob/master/utils/fps.py

import numpy as np
import torch


def farthest_point_sample_tensor(point, npoint):
    """A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space".

    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Input:
        point: point data for sampling, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled point index, [npoint, D]
    """
    device = point.device
    N, D = point.shape
    xyz = point
    centroids = torch.zeros((npoint, ), device=device)
    distance = torch.ones((N, ), device=device) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid)**2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    point = point[centroids.long()]
    return point, centroids.long()
