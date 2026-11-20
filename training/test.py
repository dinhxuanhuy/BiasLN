"""Evaluate a trained detector; writes metrics + per-video scores.

Outputs (under --out_dir):
  metrics.json                  frame/video AUC, EER, AP, ACC per dataset
  scores/<dataset>.npz          per-video mean score + label (for DeLong /
                                bootstrap significance tests in tools/stats.py)
"""

import argparse
from collections import OrderedDict
import os
import pickle
import random
import sys

import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.backends.cudnn as cudnn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

device = torch.device("cpu")
get_test_metrics = None
DETECTOR = None
DeepfakeAbstractBaseDataset = None


parser = argparse.ArgumentParser(description="Evaluate a detector checkpoint")
parser.add_argument(
    "--detector_path",
    type=str,
    default="training/config/detector/BiasLN.yaml",
    help="path to detector YAML file",
)
parser.add_argument("--test_dataset", nargs="+", default=None, help="list of test dataset names")
parser.add_argument("--weights_path", type=str, default=None, help="path to pretrained weights .pth")
parser.add_argument("--save_feat", action="store_true", default=False, help="save feature pkl for t-SNE")
parser.add_argument("--feat_out_dir", type=str, default="tsne_pkls", help="output directory for tsne_dict_*.pkl")
parser.add_argument("--pretrained", action="store_true", default=False,
                    help="evaluate the encoder as initialised (no --weights_path); for feature analysis")
parser.add_argument("--max_samples", type=int, default=None, help="optional cap on samples per dataset")
parser.add_argument("--out_dir", type=str, default=None, help="where to write metrics.json and scores/")
parser.add_argument("--rgb_dir", type=str, default=None)
parser.add_argument("--json_dir", type=str, default=None)
parser.add_argument("--split", type=str, default="test", help="JSON split to evaluate")
parser.add_argument("--amp", type=str, default=None, choices=["none", "fp16", "bf16"])
parser.add_argument("--opts", nargs="*", default=[], help="config overrides key=value")
parser.add_argument(
    "--allow_partial_load",
    action="store_true",
    default=False,
    help="allow missing/unexpected checkpoint keys instead of failing",
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="device override, e.g. cuda, cuda:0, or cpu. Defaults to config cuda setting.",
)
args = parser.parse_args()


def import_runtime_dependencies():
    global get_test_metrics, DETECTOR, DeepfakeAbstractBaseDataset

    from metrics.utils import get_test_metrics as imported_get_test_metrics
    from detectors import DETECTOR as imported_detector

    try:
        from training.dataset.abstract_dataset import (
            DeepfakeAbstractBaseDataset as imported_dataset,
        )
    except Exception:
        from dataset.abstract_dataset import DeepfakeAbstractBaseDataset as imported_dataset

    get_test_metrics = imported_get_test_metrics
    DETECTOR = imported_detector
    DeepfakeAbstractBaseDataset = imported_dataset


def init_seed(config):
    if config.get("manualSeed", None) is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    np.random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["manualSeed"])


def prepare_testing_data(config):
    def get_test_data_loader(cfg, test_name):
        cfg = cfg.copy()
        cfg["test_dataset"] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")

        return torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=int(cfg["test_batchSize"]),
            shuffle=False,
            num_workers=int(cfg["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )

    return {
        one_test_name: get_test_data_loader(config, one_test_name)
        for one_test_name in config["test_dataset"]
    }


AMP_DTYPE = None


@torch.no_grad()
def inference(model, data_dict):
    with torch.autocast(device.type, dtype=AMP_DTYPE or torch.float32,
                        enabled=AMP_DTYPE is not None and device.type == "cuda"):
        out = model(data_dict, inference=True)
    out["prob"] = out["prob"].float()
    out["feat"] = out["feat"].float()
    return out


def _move_batch_to_device(data_dict):
    for key in list(data_dict.keys()):
        if data_dict[key] is None or key == "name":
            continue
        if torch.is_tensor(data_dict[key]):
            data_dict[key] = data_dict[key].to(device)

    if "image" in data_dict and torch.is_tensor(data_dict["image"]) and data_dict["image"].ndim == 5:
        data_dict["image"] = data_dict["image"][:, 0]


def test_one_dataset(model, data_loader, max_samples=None):
    pred_list = []
    label_list = []
    feat_list = []
    label_spe_list = []
    img_names = []

    n_seen = 0
    for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_label_spe = data_dict.get("label_spe", None)

        label_bin = torch.where(data_dict["label"] != 0, 1, 0)
        data_dict["label"] = label_bin
        _move_batch_to_device(data_dict)

        preds = inference(model, data_dict)
        pred_list.append(preds["prob"].detach().cpu().numpy())
        label_list.append(data_dict["label"].detach().cpu().numpy())
        feat_list.append(preds["feat"].detach().cpu().numpy())

        if batch_label_spe is not None:
            label_spe_list.append(batch_label_spe.detach().cpu().numpy())

        if data_dict.get("name", None) is not None:
            try:
                img_names.extend(list(data_dict["name"]))
            except TypeError:
                pass

        n_seen += preds["feat"].shape[0]
        if max_samples is not None and n_seen >= max_samples:
            break

    pred_prob = np.concatenate(pred_list, axis=0)
    label_bin = np.concatenate(label_list, axis=0).astype(int)
    feat = np.concatenate(feat_list, axis=0)
    label_spe = (
        np.concatenate(label_spe_list, axis=0).astype(int)
        if label_spe_list
        else label_bin.astype(int)
    )

    return pred_prob, label_bin, feat, label_spe, img_names


def _strip_known_prefixes(key):
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, (dict, OrderedDict)):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, (dict, OrderedDict)):
                return value
    return checkpoint


def _align_lora_linear_keys(state_dict, model_state_dict):
    """Support checkpoints saved before/after wrapping CLIP Linear layers with LoRA."""
    aligned = OrderedDict()
    model_keys = set(model_state_dict.keys())
    remapped = 0

    for key, value in state_dict.items():
        candidates = [key]

        if ".self_attn." in key:
            if key.endswith(".weight"):
                candidates.append(key[:-len(".weight")] + ".base.weight")
            elif key.endswith(".bias"):
                candidates.append(key[:-len(".bias")] + ".base.bias")
            elif key.endswith(".base.weight"):
                candidates.append(key[:-len(".base.weight")] + ".weight")
            elif key.endswith(".base.bias"):
                candidates.append(key[:-len(".base.bias")] + ".bias")

        target_key = next((candidate for candidate in candidates if candidate in model_keys), key)
        if target_key != key:
            remapped += 1
        aligned[target_key] = value

    if remapped:
        print(f"===> Remapped {remapped} LoRA/base attention keys for checkpoint compatibility.")
    return aligned


def _slice_like_predictions(values, n):
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values[:n]
    if isinstance(values, tuple):
        return list(values[:n])
    if isinstance(values, list):
        return values[:n]
    return values


def _class_accuracy(prob, label):
    pred = (prob > 0.5).astype(int)
    real_idx = label == 0
    fake_idx = label == 1
    acc_real = float((pred[real_idx] == label[real_idx]).mean()) if real_idx.any() else float("nan")
    acc_fake = float((pred[fake_idx] == label[fake_idx]).mean()) if fake_idx.any() else float("nan")
    return acc_real, acc_fake


def save_tsne_pkl(out_dir, model_name, dataset_name, weights_path, pred_prob, label_bin, feat, label_spe, img_names):
    os.makedirs(out_dir, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    safe_data = dataset_name.replace("/", "_")
    out_path = os.path.join(out_dir, f"tsne_dict_{safe_model}_{safe_data}.pkl")

    tsne_dict = {
        "feat": feat,
        "label_spe": label_spe,
        "pred": pred_prob,
        "label": label_bin,
        "img_names": img_names,
        "dataset": dataset_name,
        "model_name": model_name,
        "weights_path": weights_path,
    }

    with open(out_path, "wb") as f:
        pickle.dump(tsne_dict, f)

    print(f"[SAVE_FEAT] {out_path}")
    print(f"           feat={feat.shape}, label_spe_unique={np.unique(label_spe)}")


def load_config(detector_path):
    with open(detector_path, "r") as f:
        config = yaml.safe_load(f)

    test_cfg_path = os.path.join(ROOT, "training", "config", "test_config.yaml")
    if os.path.exists(test_cfg_path):
        with open(test_cfg_path, "r") as f:
            test_config = yaml.safe_load(f)
        if isinstance(test_config, dict):
            config.update(test_config)

    config.setdefault("cuda", True)
    config.setdefault("cudnn", True)
    config.setdefault("workers", 4)
    config.setdefault("test_batchSize", 64)
    config.setdefault("metric_scoring", "auc")
    if args.rgb_dir:
        config["rgb_dir"] = args.rgb_dir
    if args.json_dir:
        config["dataset_json_folder"] = args.json_dir
    config["data_split"] = args.split
    from config_utils import apply_overrides
    apply_overrides(config, args.opts)
    return config


def video_scores(img_names, prob, label):
    """Mean frame score per video; video key = parent directory of the frame."""
    groups = OrderedDict()
    for name, p, y in zip(img_names, prob, label):
        key = os.path.dirname(str(name).replace("\\", "/"))
        groups.setdefault(key, []).append((float(p), int(y)))
    keys = list(groups.keys())
    scores = np.array([np.mean([p for p, _ in groups[k]]) for k in keys])
    labels = np.array([int(round(np.mean([y for _, y in groups[k]]))) for k in keys])
    return keys, scores, labels


def resolve_device(config):
    if args.device is not None:
        return torch.device(args.device)
    if config.get("cuda", True) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model, weights_path, allow_partial_load=False):
    if weights_path is None:
        raise ValueError("--weights_path is required to test a pretrained model.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights_path does not exist: {weights_path}")

    print(f"===> Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(
            "Checkpoint must be a state_dict or contain one of: "
            "state_dict, model_state_dict, model, net."
        )

    normalized_state_dict = {
        _strip_known_prefixes(key): value
        for key, value in state_dict.items()
    }
    normalized_state_dict = _align_lora_linear_keys(
        normalized_state_dict,
        model.state_dict(),
    )

    msg = model.load_state_dict(normalized_state_dict, strict=False)
    trainable_only = isinstance(checkpoint, dict) and checkpoint.get("trainable_only", False)
    if trainable_only:
        # Compact checkpoints store only trained tensors; every missing key must
        # be a frozen parameter/buffer that comes from the pretrained backbone.
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        missing_trainable = [k for k in msg.missing_keys if k in trainable]
        if missing_trainable or msg.unexpected_keys:
            raise RuntimeError(
                f"Compact checkpoint mismatch: missing trainable={missing_trainable[:5]}, "
                f"unexpected={msg.unexpected_keys[:5]}"
            )
        print(f"===> Compact checkpoint: {len(normalized_state_dict)} trained tensors loaded; "
              f"{len(msg.missing_keys)} frozen tensors kept from the pretrained backbone.")
        return
    print("===> Load checkpoint done!")
    print(f"     Missing keys: {len(msg.missing_keys)}")
    print(f"     Unexpected keys: {len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"     Example missing: {msg.missing_keys[:3]} ...")
    if msg.unexpected_keys:
        print(f"     Example unexpected: {msg.unexpected_keys[:3]} ...")

    if not allow_partial_load and (msg.missing_keys or msg.unexpected_keys):
        raise RuntimeError(
            "Checkpoint does not exactly match the model. "
            "This often means --detector_path, LoRA config, or --weights_path is wrong. "
            "Pass --allow_partial_load only if you intentionally want partial loading."
        )


def main():
    global device

    import_runtime_dependencies()

    config = load_config(args.detector_path)
    if args.test_dataset is not None:
        config["test_dataset"] = args.test_dataset
    if "test_dataset" not in config or not config["test_dataset"]:
        raise ValueError("Provide --test_dataset or set test_dataset in the YAML config.")
    if "model_name" not in config:
        raise ValueError("YAML must contain config['model_name'] (e.g., bias_lora_asy).")

    global AMP_DTYPE
    AMP_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp or config.get("amp", "none"))
    device = resolve_device(config)
    print(f"===> Using device: {device}")

    init_seed(config)
    if config.get("cudnn", True) and device.type == "cuda":
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    if not args.pretrained:
        load_checkpoint(model, args.weights_path, allow_partial_load=args.allow_partial_load)

    model.eval()
    all_metrics = {}

    for dataset_name, loader in test_data_loaders.items():
        print(f"\n===== Testing on: {dataset_name} =====")
        data_dict_global = loader.dataset.data_dict if hasattr(loader.dataset, "data_dict") else {}
        img_names_global = data_dict_global.get("image", None)

        pred_prob, label_bin, feat, label_spe, img_names_batch = test_one_dataset(
            model, loader, max_samples=args.max_samples
        )

        img_names_for_metric = img_names_global if img_names_global is not None else img_names_batch
        img_names_for_metric = _slice_like_predictions(img_names_for_metric, len(pred_prob))
        if img_names_for_metric is None:
            raise ValueError("Image names are required for video-level metrics.")
        if len(img_names_for_metric) != len(pred_prob):
            raise ValueError(
                f"Metric input length mismatch for {dataset_name}: "
                f"{len(pred_prob)} predictions but {len(img_names_for_metric)} image names."
            )

        metric_one_dataset = get_test_metrics(
            y_pred=pred_prob,
            y_true=label_bin,
            img_names=img_names_for_metric,
        )
        all_metrics[dataset_name] = metric_one_dataset
        if args.out_dir:
            from sklearn.metrics import roc_auc_score
            keys, v_scores, v_labels = video_scores(img_names_for_metric, pred_prob, label_bin)
            os.makedirs(os.path.join(args.out_dir, "scores"), exist_ok=True)
            np.savez_compressed(
                os.path.join(args.out_dir, "scores", f"{dataset_name}.npz"),
                video=np.array(keys), score=v_scores, label=v_labels,
                frame_score=pred_prob.astype(np.float32), frame_label=label_bin.astype(np.int8),
                **({"frame_feat": feat.astype(np.float16),
                    "frame_name": np.array([str(n) for n in img_names_for_metric])}
                   if args.save_feat else {}),
            )
            metric_one_dataset["video_auc_dir"] = float(roc_auc_score(v_labels, v_scores))
            metric_one_dataset["n_videos"] = int(len(keys))
            metric_one_dataset["n_frames"] = int(len(pred_prob))

        for key, value in metric_one_dataset.items():
            if key in ["pred", "label", "dataset_dict"]:
                continue
            print(f"{key}: {value}")
        acc_real, acc_fake = _class_accuracy(pred_prob, label_bin)
        print(f"acc_real: {acc_real}; acc_fake: {acc_fake}")
        print(
            "pred_prob: "
            f"min={float(pred_prob.min()):.6f}, "
            f"max={float(pred_prob.max()):.6f}, "
            f"mean={float(pred_prob.mean()):.6f}, "
            f">0.5={float((pred_prob > 0.5).mean()):.6f}"
        )

        if args.save_feat:
            save_tsne_pkl(
                out_dir=args.feat_out_dir,
                model_name=config["model_name"],
                dataset_name=dataset_name,
                weights_path=args.weights_path,
                pred_prob=pred_prob,
                label_bin=label_bin,
                feat=feat,
                label_spe=label_spe,
                img_names=(img_names_for_metric if isinstance(img_names_for_metric, list) else []),
            )

    if args.out_dir:
        from config_utils import dump_json
        summary = {
            name: {k: float(v) for k, v in m.items() if k not in ("pred", "label", "dataset_dict")}
            for name, m in all_metrics.items()
        }
        dump_json({"weights_path": args.weights_path, "detector_path": args.detector_path,
                   "split": args.split, "metrics": summary},
                  os.path.join(args.out_dir, "metrics.json"))
        print(f"===> Metrics written to {os.path.join(args.out_dir, 'metrics.json')}")
    print("\n===> Test Done!")
    if args.save_feat:
        print(f"===> Feature pkls saved under: {os.path.abspath(args.feat_out_dir)}")


if __name__ == "__main__":
    main()
