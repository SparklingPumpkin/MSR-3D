# Adapted inference primitives from OpenSplat3D and Gaussian Splatting.
# Copyright (C) 2023, Inria; GRAPHDECO research group.
# Original ownership and research/evaluation license terms apply.
# See THIRD_PARTY.md and third_party/licenses/.
from __future__ import annotations
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from plyfile import PlyData
from diff_gaussian_rasterization import MAX_FEATURE_DIM
from .general import build_scaling_rotation, inverse_sigmoid, strip_symmetric
from .sh import RGB2SH

def build_covariance_from_scaling_rotation(
    rotation: torch.Tensor, scaling: torch.Tensor, scaling_modifier: float
):
    L = build_scaling_rotation(scaling * scaling_modifier, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

class GaussianModel:
    def __init__(self) -> None:
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # color features
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)

        # other features
        self._features: torch.Tensor | None = torch.empty(0)

        # semantic opacity
        self._sem_opacity: torch.Tensor | None = torch.empty(0)

        # setup functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def num_points(self) -> int:
        return self._xyz.size(0)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_spherical_harmonics(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_features(self) -> torch.Tensor | None:
        return self._features

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.covariance_activation(
            self._rotation, self.get_scaling, scaling_modifier
        )

    def state_dict(self):
        return {
            "xyz": self._xyz,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "features": self._features,
        }

    def load_state_dict(self, state_dict: dict):
        self._xyz = state_dict["xyz"]
        self._scaling = state_dict["scaling"]
        self._rotation = state_dict["rotation"]
        self._opacity = state_dict["opacity"]
        self._features_dc = state_dict["features_dc"]
        self._features_rest = state_dict["features_rest"]
        self._features = state_dict["features"]

def create_from_ply(
    ply_path: Path,
    max_sh_degree: int,
    device: torch.device,
    only_feature_grads: bool = False,
    feature_dim: int = MAX_FEATURE_DIM,
):
    # only called for loading inference model
    plydata = PlyData.read(str(ply_path))

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
    )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    feat_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("feat_")
    ]
    features: npt.NDArray | None = None
    if len(feat_names) > 0:
        feat_names = sorted(feat_names, key=lambda x: int(x.split("_")[-1]))
        features = np.zeros((xyz.shape[0], len(feat_names)))
        for idx, attr_name in enumerate(feat_names):
            features[:, idx] = np.asarray(plydata.elements[0][attr_name])

    model = GaussianModel()
    model._xyz = nn.Parameter(
        torch.tensor(xyz, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )
    model._features_dc = nn.Parameter(
        torch.tensor(features_dc, dtype=torch.float32, device=device)
        .transpose(1, 2)
        .contiguous(),
        requires_grad=not only_feature_grads,
    )
    model._features_rest = nn.Parameter(
        torch.tensor(features_extra, dtype=torch.float32, device=device)
        .transpose(1, 2)
        .contiguous(),
        requires_grad=not only_feature_grads,
    )
    model._opacity = nn.Parameter(
        torch.tensor(opacities, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )
    model._scaling = nn.Parameter(
        torch.tensor(scales, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )
    model._rotation = nn.Parameter(
        torch.tensor(rots, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )

    allow_mismatch = False
    if features is not None and (allow_mismatch or feature_dim == features.shape[1]):
        model._features = nn.Parameter(
            torch.tensor(features, dtype=torch.float32, device=device)
        )
    elif not allow_mismatch:
        if features is not None:
            print(
                f"Feature dimension mismatch, expected {feature_dim} but got {features.shape[1]}"
            )
        model._features = nn.Parameter(
            RGB2SH(
                torch.rand(
                    (xyz.shape[0], feature_dim),
                    dtype=torch.float32,
                    device=device,
                )
            )
        )

    return model
