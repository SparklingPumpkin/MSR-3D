"""Offline MasQCLIP image/text encoding with explicit weight paths."""
from pathlib import Path
import torch
from .masqclip_network import MasQCLIP
from .crops import CropParams, masks_to_crops


class Encoder:
    img_size = (336, 336)
    dtype = torch.float32

    def __init__(self, encoders_root, device='cuda'):
        root = Path(encoders_root)
        self.device = torch.device(device)
        self.network = MasQCLIP(['ViT-L/14@336px'], root / 'clip/ViT-L-14-336px.pt', device=str(device))
        self.network.from_pretrained(str(root / 'masqclip/base_novel.pth'))
        self.network.to(self.device).eval().requires_grad_(False)

    def preprocess_images(self, images):
        return self.network.preprocess_images(images)

    def crop(self, image, mask, ratio=.3, dynamic_ratio=True):
        if not bool(mask.any()):
            raise ValueError('Cannot encode an empty mask')
        if dynamic_ratio and float(mask.sum()) / mask.numel() >= .0075:
            ratio = .1
        params = CropParams(expansion_ratio=ratio, dynamic_ratio=dynamic_ratio)
        crops, masks, _, _ = masks_to_crops(image, mask.unsqueeze(0), params, self)
        middle = params.levels // 2
        return crops[0, middle], masks[0, middle].unsqueeze(0)

    @torch.inference_mode()
    def images(self, crops, masks):
        images = torch.stack(crops).to(device=self.device, dtype=self.dtype)
        masks = torch.stack(masks).to(device=self.device, dtype=self.dtype)
        value = self.network.get_image_embedding(images, masks, normalize=False).squeeze(1)
        value = value / value.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        return value.cpu()

    @torch.inference_mode()
    def text(self, queries):
        prompts = ['an image of {}'.format(query) for query in queries]
        return self.network.get_text_embedding(prompts, normalize=True).cpu()
