#!/usr/bin/env python3
"""
顺序训练多个小规模 episode 数据集的 SmolVLA policy。

使用方法:
    1. 先运行 `sample_mini_datasets.ipynb` 生成 `demo_data_language_subsets/seed_*/` 目录。
    2. 执行本脚本:
           python train_mini_subsets.py
       或传入自定义子集列表:
           python train_mini_subsets.py --subsets seed_000 seed_001

脚本会为每个子集生成临时配置文件, 基于 `smolvla_episode.yaml` 修改
`dataset.root`、`output_dir`、`job_name` 和 `seed`，然后调用 `train_model.py`
进行顺序训练。原始 yaml 不会被覆盖。
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="顺序训练多个小规模 episode 子集")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("smolvla_episode.yaml"),
        help="基础配置文件路径 (默认: smolvla_episode.yaml)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="*",
        default=None,
        help="要训练的子集目录名称 (默认自动查找 demo_data_language_subsets_nbr15/seed_*)",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("demo_data_language_subsets_nbr15"),
        help="小样本子集根目录 (默认: demo_data_language_subsets_nbr15)",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train_model.py"),
        help="训练脚本路径 (默认: train_model.py)",
    )
    return parser.parse_args()


def discover_subsets(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"未找到子集根目录: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def main() -> None:
    args = parse_args()

    if not args.base_config.exists():
        raise FileNotFoundError(f"基础配置文件不存在: {args.base_config}")
    if not args.train_script.exists():
        raise FileNotFoundError(f"训练脚本不存在: {args.train_script}")

    if args.subsets is None or len(args.subsets) == 0:
        subset_names = discover_subsets(args.subset_root)
    else:
        subset_names = args.subsets

    if len(subset_names) == 0:
        raise ValueError("没有找到任何子集目录，请先运行 sample_mini_datasets.ipynb")

    base_cfg = yaml.safe_load(args.base_config.read_text())
    base_seed = base_cfg.get("seed", 0)

    with tempfile.TemporaryDirectory(prefix="smolvla_subset_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, subset in enumerate(subset_names):
            subset_path = args.subset_root / subset
            if not subset_path.exists():
                raise FileNotFoundError(f"子集目录不存在: {subset_path}")

            cfg = yaml.safe_load(args.base_config.read_text())
            dataset_cfg = cfg.setdefault("dataset", {})
            dataset_cfg["root"] = str(subset_path)
            dataset_cfg["repo_id"] = None
            dataset_cfg["revision"] = None

            cfg["output_dir"] = str(Path(cfg.get("output_dir", "./ckpt/smolvla_omy")) / subset)
            cfg["job_name"] = f"{cfg.get('job_name', 'smolvla_job')}_{subset}"
            cfg["seed"] = base_seed + idx

            tmp_cfg = tmpdir_path / f"{subset}.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

            print(f"[INFO] 开始训练子集: {subset} (使用配置: {tmp_cfg})")
            cmd = ["python", str(args.train_script), "--config", str(tmp_cfg)]
            subprocess.run(cmd, check=True)
            print(f"[INFO] 子集 {subset} 训练完成\\n")


if __name__ == "__main__":
    main()
