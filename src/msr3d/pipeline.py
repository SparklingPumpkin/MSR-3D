"""Build representations and matching inputs from newly encoded observation pools."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import Matcher
from .representation.scene import construct_scene, make_query_features
from .benchmark import evaluate_predictions


def match_scene(pool, representations, output, s1_adapter):
    pool, representations, output = Path(pool), Path(representations), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((pool / "metadata.json").read_text())
    with np.load(pool / "queries.npz", allow_pickle=False) as archive:
        queries = list(
            zip(
                archive["query_ids"].tolist(),
                archive["query_texts"].tolist(),
                archive["features"],
            )
        )
    model = Matcher()
    predictions = []
    for index, (qid, text, embedding) in enumerate(queries):
        features = make_query_features(
            representations, embedding, s1_adapter, pool / "anchors.npz"
        )
        with (output / f"{index:03d}.npz").open("xb") as handle:
            np.savez_compressed(
                handle,
                instance_ids=np.asarray(features.instance_ids),
                matrix=features.matrix,
                valid=features.valid,
                anchor_instance_ids=np.asarray(features.anchor_instance_ids),
                anchor_similarities=features.anchor_similarities,
            )
        predictions.append(
            {"dataset": metadata["dataset"], "query_id": qid, **model.predict(features)}
        )
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": metadata["dataset"],
                "scene": metadata["scene"],
                "s1_adapter": s1_adapter,
                "query_ids": [q[0] for q in queries],
                "matching_inputs": "fresh representations, text embeddings and scene-wide anchors",
                "read_cached_matching_features": False,
            },
            indent=2,
        )
        + "\n"
    )
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=("lerf67", "3dovs61"))
    p.add_argument("--scene", required=True)
    p.add_argument(
        "--pool",
        type=Path,
        help="Reuse a newly generated pool; omit for a full cold start",
    )
    p.add_argument(
        "--construction-threads",
        type=int,
        help="CPU threads for prototype construction (default: 16)",
    )
    p.add_argument(
        "--fast-encoding",
        action="store_true",
        help="Use batched FP32 image encoding instead of historical singleton calls",
    )
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    if a.pool is None:
        from .pool_batch import build_scene

        a.pool = a.output / "pool"
        build_scene(
            a.data_root,
            a.dataset,
            a.scene,
            a.pool,
            reference_precision=not a.fast_encoding,
        )
    representations = a.output / "representations"
    construction_threads = a.construction_threads or 16
    torch.set_num_threads(construction_threads)
    construct_scene(
        a.pool / "observations.npz",
        representations,
        frame_suffix=".pt" if a.dataset == "lerf67" else "",
    )
    torch.set_num_threads(1)
    (a.output / "construction.json").write_text(
        json.dumps(
            {"threads": construction_threads, "torch": torch.__version__}, indent=2
        )
        + "\n"
    )
    contract = json.loads(
        (
            a.data_root / "preprocessed" / a.dataset / a.scene / "runtime.json"
        ).read_text()
    )
    predictions = match_scene(
        a.pool, representations, a.output / "matching", contract["s1_adapter"]
    )
    evaluation = a.data_root / "evaluation/cached-matcher-reference"
    queries = [
        q
        for q in json.loads((evaluation / "queries.json").read_text())
        if q["dataset"] == a.dataset and q["scene"] == a.scene
    ]
    result = evaluate_predictions(
        evaluation,
        queries,
        predictions,
        metadata={"generation": "fresh pool and representations"},
    )
    for key, value in result.items():
        (a.output / f"{key}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
