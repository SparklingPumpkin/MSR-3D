"""Command-line inference and a deterministic synthetic input generator."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import Matcher, load_query


def _write_new(path, text):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(prog='msr3d', description='MSR-3D fixed-weight instance matching')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('example', help='write synthetic cached features, not benchmark data')
    demo.add_argument('--output', required=True, type=Path)
    infer = commands.add_parser('predict', help='match one query against a cached instance feature bank')
    infer.add_argument('--input', required=True, type=Path)
    infer.add_argument('--output', required=True, type=Path)
    infer.add_argument('--weights-dir', type=Path, help='directory containing matcher.json and matcher.npz')
    args = parser.parse_args()
    if args.command == 'example':
        generator = np.random.default_rng(17)
        matrix = np.zeros((4, 6, 13), dtype=np.float32)
        matrix[:, :3, 0] = generator.uniform(.1, .8, (4, 3))
        matrix[:, :3, 1] = 1 / 3
        matrix[:, :3, 2:5] = generator.uniform(0., 1., (4, 3, 3))
        matrix[:, :3, 6:8] = .5
        matrix[:, :3, 8] = 3
        matrix[:, :3, 9:11] = .5
        matrix[:, :3, 12] = 1.
        valid = np.zeros((4, 6), dtype=bool)
        valid[:, :3] = True
        ids = np.asarray(['object_a', 'object_b', 'object_c', 'object_d'])
        with args.output.open('xb') as handle:
            np.savez_compressed(handle, instance_ids=ids, matrix=matrix, valid=valid,
                                anchor_instance_ids=ids, anchor_similarities=np.asarray([.8, .6, .4, .2], dtype=np.float32))
        print('Synthetic feature input written.')
    else:
        torch.set_num_threads(1)
        result = Matcher(args.weights_dir).predict(load_query(args.input))
        _write_new(args.output, json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(f"Selected {len(result['selected_ids'])} of {len(result['instance_ids'])} instances.")


if __name__ == '__main__':
    main()
