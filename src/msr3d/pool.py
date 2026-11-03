"""Reconstruct semantic observations from preprocessed scene geometry and RGB."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import imageio.v3 as iio
import numpy as np
import torch
import yaml
from .frontend.cameras import load_cameras
from .frontend.gaussian import create_from_ply
from .frontend.render import render
from .frontend.encoder import Encoder


def read_rgba(path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    value = iio.imread(path)
    if value.ndim != 3 or value.shape[-1] not in (3, 4):
        raise ValueError(f'Invalid rendered RGB(A) shape: {path.name}')
    mask = value[..., 3] > 0 if value.shape[-1] == 4 else value[..., :3].any(axis=-1)
    return torch.from_numpy(value[..., :3].copy()).permute(2, 0, 1).contiguous().to(torch.uint8), torch.from_numpy(mask.copy())


@torch.inference_mode()
def build_pool(data_root, dataset, scene, output, instance_ids=None, frames=None, batch_size=8, device='cuda'):
    root, output = Path(data_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene_root = root / 'preprocessed' / dataset / scene
    config = yaml.safe_load((scene_root / 'config.yaml').read_text())
    model = create_from_ply(scene_root / 'scene.ply', config['model']['sh_degree'], torch.device(device), feature_dim=config['model']['mask_dim'])
    model._xyz.requires_grad_(False)
    labels = torch.from_numpy(np.load(scene_root / 'instances.npy', allow_pickle=False)).view(-1)
    if len(labels) != model.num_points:
        raise ValueError('Instance labels are not aligned to the supplied PLY')
    ids = sorted(int(i) for i in labels.unique() if i >= 0)
    if instance_ids is not None:
        requested = {int(i.removeprefix('instance_')) for i in instance_ids}
        if not requested <= set(ids):raise ValueError('Requested instance absent from PLY labels')
        ids = [i for i in ids if i in requested]
    cameras = load_cameras(scene_root, torch.device(device), config['model']['resolution'], frames)
    encoder = Encoder(root / 'encoders', device)
    bg = torch.tensor([1., 1., 1.] if config['model']['white_background'] else [0., 0., 0.], device=device)
    pipe = SimpleNamespace(**config['pipe'])
    rows, features, crops, crop_masks, pending = [], [], [], [], []
    def flush():
        if not pending:return
        values = encoder.images(crops, crop_masks)
        for row, feature in zip(pending, values):rows.append(row);features.append(feature.numpy())
        crops.clear();crop_masks.clear();pending.clear()
    for label in ids:
        iid = f'instance_{label:04d}'
        membership = labels == label
        colors = torch.ones((model.num_points, 3), device=device)
        colors[~membership] = 0.
        for camera in cameras:
            package = render(camera, model, pipe, bg, config['model']['sh_degree'], override_color=colors, render_features=False)
            visible = package.render.clamp(0, 1).permute(1, 2, 0).contiguous().mean(dim=-1) > .3
            visible = visible.cpu()
            visible_count = int((package.visibility_filter.cpu() & membership).sum())
            rgb = camera.original_image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).contiguous()
            sources = []
            if visible.any():sources.append(('sam', rgb, visible))
            rgba = read_rgba(scene_root / 'rendered_instances' / iid / (camera.name + '.png'))
            if rgba is not None:
                rendered_rgb, rendered_mask = rgba
                if rendered_mask.any():
                    composited = rgb.clone();composited[:, rendered_mask] = rendered_rgb[:, rendered_mask]
                    sources.append(('render', composited.contiguous(), rendered_mask))
            for source, image, mask in sources:
                crop, crop_mask = encoder.crop(image, mask, config['lang']['ratio'], config['lang']['dynamic_ratio'])
                crops.append(crop);crop_masks.append(crop_mask)
                pending.append({'instance_id':iid,'frame':camera.name,'source':source,
                                'visible_count':visible_count,'label_count':int(membership.sum()),
                                'mask_area':int(mask.sum()),'height':int(mask.shape[0]),'width':int(mask.shape[1])})
                if instance_ids is not None and frames is not None:
                    iio.imwrite(output / f'{iid}_{camera.name}_{source}_mask.png', mask.numpy().astype(np.uint8) * 255)
                    overlay = image.permute(1,2,0).numpy().copy()
                    overlay[~mask.numpy()] = (overlay[~mask.numpy()].astype(float) * .25).astype(np.uint8)
                    iio.imwrite(output / f'{iid}_{camera.name}_{source}_view.jpg', overlay)
                if len(pending) >= batch_size:flush()
        flush()
        print('INSTANCE_DONE',dataset,scene,iid,len(rows),flush=True)
    if not rows:raise ValueError('No observations generated')
    with (output / 'observations.npz').open('xb') as f:
        np.savez_compressed(f,features=np.stack(features),instance_ids=np.asarray([r['instance_id'] for r in rows]),
                            frame_names=np.asarray([r['frame'] for r in rows]),sources=np.asarray([r['source'] for r in rows]))
    (output / 'metadata.json').write_text(json.dumps({'dataset':dataset,'scene':scene,'observations':rows,
        'input_boundary':'preprocessed PLY, instance labels, posed RGB and rendered instance RGBA',
        'read_observation_or_representation_references':False},indent=2)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--dataset',choices=('lerf67','3dovs61'),required=True)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--instances',nargs='+')
    parser.add_argument('--frames',nargs='+')
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    torch.set_num_threads(1)
    build_pool(args.data_root,args.dataset,args.scene,args.output,args.instances,
               set(args.frames) if args.frames else None,args.batch_size)

if __name__=='__main__':main()
