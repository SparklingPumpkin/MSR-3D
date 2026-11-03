# Adapted inference primitives from OpenSplat3D and Gaussian Splatting.
# Copyright (C) 2023, Inria; GRAPHDECO research group.
# Original ownership and research/evaluation license terms apply.
# See THIRD_PARTY.md and third_party/licenses/.
from __future__ import annotations
from pathlib import Path
from typing import NamedTuple
import imageio.v3 as iio
import torch
from torch import nn
from torchvision.transforms.v2 import InterpolationMode, Resize
from .camera_math import get_projection_matrix, get_world2view, focal2fov
from .colmap import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

class CameraInfo(NamedTuple):
    uid: int
    R: torch.Tensor
    T: torch.Tensor
    fovX: float
    fovY: float
    image: torch.Tensor
    image_path: Path
    image_name: str
    width: int
    height: int
    masks: torch.Tensor | None

WARNED = False

class Camera(nn.Module):
    def __init__(
        self,
        uid: int,
        name: str,
        R: torch.FloatTensor,
        T: torch.FloatTensor,
        fovX: float,
        fovY: float,
        image: torch.FloatTensor,  # CHW
        alpha_mask: torch.FloatTensor | None,
        masks: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.uid = uid
        self.name = name
        self.R = R
        self.T = T
        self.fovX = fovX
        self.fovY = fovY
        self.znear = 0.01
        self.zfar = 100.0

        self.original_image = image.clamp(0.0, 1.0)  # CHW
        self.image_width = self.original_image.size(2)
        self.image_height = self.original_image.size(1)

        self.masks = masks

        if alpha_mask is not None:
            self.original_image *= alpha_mask

        world_view_transform = get_world2view(R, T).transpose(0, 1)
        self.register_buffer("world_view_transform", world_view_transform)
        self.world_view_transform: torch.Tensor = self.world_view_transform

        projection_matrix = get_projection_matrix(
            znear=self.znear, zfar=self.zfar, fovX=fovX, fovY=fovY
        ).transpose(0, 1)
        self.register_buffer("projection_matrix", projection_matrix)
        self.projection_matrix: torch.Tensor = self.projection_matrix

        self.register_buffer(
            "full_proj_transform",
            (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0),
        )
        self.full_proj_transform: torch.Tensor = self.full_proj_transform

        self.register_buffer(
            "camera_center", self.world_view_transform.inverse()[3, :3]
        )
        self.camera_center: torch.Tensor = self.camera_center

def to_camera(
    cam_info: CameraInfo,
    resolution: int,
    resolution_scale: float,
):
    orig_h, orig_w = cam_info.image.shape[:2]

    if resolution in [1, 2, 4, 8]:
        new_resolution = (
            round(orig_h / (resolution_scale * resolution)),
            round(orig_w / (resolution_scale * resolution)),
        )
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        new_resolution = (int(orig_h / scale), int(orig_w / scale))

    resizer = Resize(new_resolution)
    resized_image_rgb: torch.FloatTensor = resizer(
        (cam_info.image.float() / 255.0).permute(2, 0, 1)
    )
    gt_image: torch.FloatTensor = resized_image_rgb[:3, ...]  # type: ignore

    alpha_mask: torch.FloatTensor | None = None
    if resized_image_rgb.size(0) == 4:
        alpha_mask = resized_image_rgb[3:4, ...]  # type: ignore

    gt_masks: torch.Tensor | None = None
    if cam_info.masks is not None:
        mask_resizer = Resize(new_resolution, interpolation=InterpolationMode.NEAREST)
        old_device = cam_info.masks.device
        masks = cam_info.masks.cuda()
        labels: torch.Tensor = torch.unique(masks)
        if masks.ndim == 2:  # merged masked
            masks = mask_resizer(masks.unsqueeze(0)).squeeze(0)
        else:
            masks = mask_resizer(masks)

        # NOTE some masks can be very small and after resizing they are lost, therefore the labels are not consecutive anymore.
        labels2: torch.Tensor = torch.unique(masks)
        if labels.shape != labels2.shape:
            masks = raise_mask_resize_unsupported(masks)
        gt_masks = masks.to(old_device)

    return Camera(
        uid=cam_info.uid,
        name=cam_info.image_name,
        R=cam_info.R,
        T=cam_info.T,
        fovX=cam_info.fovX,
        fovY=cam_info.fovY,
        image=gt_image,
        alpha_mask=alpha_mask,
        masks=gt_masks,
    )

def load_cameras(scene_root, device, resolution=-1, frames=None):
    root = Path(scene_root)
    extrinsics = read_extrinsics_binary(root / 'sparse/0/images.bin')
    intrinsics = read_intrinsics_binary(root / 'sparse/0/cameras.bin')
    cameras = []
    for extr in sorted(extrinsics.values(), key=lambda x: Path(x.name).stem):
        name = Path(extr.name).stem
        if frames is not None and name not in frames:
            continue
        intr = intrinsics[extr.camera_id]
        if intr.model.name == 'SIMPLE_PINHOLE':
            fx = fy = float(intr.params[0])
        elif intr.model.name == 'PINHOLE':
            fx, fy = map(float, intr.params[:2])
        else:
            raise ValueError('Only undistorted pinhole cameras are supported')
        path = root / 'images' / Path(extr.name).name
        info = CameraInfo(extr.id, qvec2rotmat(extr.qvec).transpose(0, 1), extr.tvec,
                          focal2fov(fx, intr.width), focal2fov(fy, intr.height),
                          torch.from_numpy(iio.imread(path)), path, name, intr.width, intr.height, None)
        cameras.append(to_camera(info, resolution, 1.).to(device))
    return cameras

def raise_mask_resize_unsupported(masks):
    raise ValueError('Use masks aligned to the supplied camera resolution')
