from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torch.jit.annotations import BroadcastingList2
from torchvision.utils import _log_api_usage_once
from torchvision.extension import _assert_has_ops


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """
    Original function is torchvision.ops.roi_align.roi_align
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(roi_align)
    _assert_has_ops()
    #check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    #if not isinstance(rois, torch.Tensor):
    #    rois = convert_boxes_to_roi_format(rois)
    return torch.ops.torchvision.roi_align(
        input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
    )


@torch.jit.unused
def _onnx_merge_levels(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros(
        (levels.size(0), first_result.size(1), first_result.size(2), first_result.size(3)), dtype=dtype, device=device
    )
    for level in range(len(unmerged_results)):
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        index = index.expand(
            index.size(0),
            unmerged_results[level].size(1),
            unmerged_results[level].size(2),
            unmerged_results[level].size(3),
        )
        res = res.scatter(0, index, unmerged_results[level])
    return res


# TODO: (eellison) T54974082 https://github.com/pytorch/pytorch/issues/26744/pytorch/issues/26744
def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level ids
        box_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps).to(s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


def _convert_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [torch.full_like(b[:, :1], i, layout=torch.strided).to(dtype).to(device) for i, b in enumerate(boxes)],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def _infer_scale(feature: Tensor, original_size: List[int]) -> float:
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    s1, s2 = size[0], original_size[0]
    approx_scale = (1.0 * s1) / (1.0 * s2)
    scale = torch.empty(1, ).fill_(approx_scale).log2().round() * 1.0
    scale = 2 ** scale
    return scale.item()


def _setup_scales(
    features: List[Tensor], image_shapes: List[Tuple[int, int]], canonical_scale: int, canonical_level: int
) -> Tuple[List[float], LevelMapper]:
    #if not image_shapes:
    #    raise ValueError("images list should not be empty")
    original_input_shape = image_shapes

    scales = [_infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = torch.empty(1, dtype=torch.float32).fill_(scales[0])
    lvl_min = -torch.log2(lvl_min).to(torch.int64).item()

    lvl_max = torch.empty(1, dtype=torch.float32).fill_(scales[-1])
    lvl_max = -torch.log2(lvl_max).to(torch.int64).item()

    map_levels = initLevelMapper(
        lvl_min,
        lvl_max,
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return scales, map_levels


def _filter_input(x: Dict[str, Tensor], featmap_names: List[str]) -> List[Tensor]:
    x_filtered = []
    for k, v in x.items():
        if k in featmap_names:
            x_filtered.append(v)
    return x_filtered


def _multiscale_roi_align(
    x_filtered: List[Tensor],
    boxes: List[Tensor],
    output_size: List[int],
    sampling_ratio: int,
    scales: Optional[List[float]],
    mapper: Optional[LevelMapper],
) -> Tensor:
    """
    Args:
        x_filtered (List[Tensor]): List of input tensors.
        boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
            (x1, y1, x2, y2) format and in the image reference size, not the feature map
            reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        output_size (Union[List[Tuple[int, int]], List[int]]): size of the output
        sampling_ratio (int): sampling ratio for ROIAlign
        scales (Optional[List[float]]): If None, scales will be automatically infered. Default value is None.
        mapper (Optional[LevelMapper]): If none, mapper will be automatically infered. Default value is None.
    Returns:
        result (Tensor)
    """
    if scales is None or mapper is None:
        raise ValueError("scales and mapper should not be None")

    num_levels = len(x_filtered)
    rois = _convert_to_roi_format(boxes)

    if num_levels == 1:
        return roi_align(
            x_filtered[0],
            rois,
            output_size=output_size,
            spatial_scale=scales[0],
            sampling_ratio=sampling_ratio,
        )

    levels = mapper(boxes)

    dtype, device = x_filtered[0].dtype, x_filtered[0].device
    result = list()

    tracing_results = []
    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = torch.where(levels == level)[0]
        rois_per_level = rois[idx_in_level]

        result_idx_in_level = roi_align(
            per_level_feature,
            rois_per_level,
            output_size=output_size,
            spatial_scale=scale,
            sampling_ratio=sampling_ratio,
        )

        if torchvision._is_tracing():
            tracing_results.append(result_idx_in_level.to(dtype))
        else:
            result.append(result_idx_in_level.to(dtype))

    if torchvision._is_tracing():
        result = _onnx_merge_levels(levels, tracing_results)
    else:
        result = torch.cat(result, dim=0)

    return result


class MultiScaleRoIAlign(nn.Module):
    """
    Original class is torchvision.ops.poolers.MultiScaleRoIAlign
    """

    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        _log_api_usage_once(self)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x_filtered = _filter_input(x, self.featmap_names)
        #if self.scales is None or self.map_levels is None:
        self.scales, self.map_levels = _setup_scales(
            x_filtered, image_shapes, self.canonical_scale, self.canonical_level
        )

        return _multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )
