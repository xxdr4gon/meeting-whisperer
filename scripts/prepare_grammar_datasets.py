#!/usr/bin/env python3
"""Download and merge Estonian grammar correction datasets."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Dict

import tempfile
import shutil

import requests


HF_REPO_URL = "https://huggingface.co/datasets/TalTechNLP/grammar2_et/resolve/main"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_hf_dataset(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_path = target_dir / "grammar2_et"
    if cache_path.exists():
        print(f"Hugging Face dataset already downloaded at {cache_path}")
        return cache_path

    print("Downloading grammar2_et dataset from Hugging Face")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        response = requests.get(f"{HF_REPO_URL}/grammar2_et.json", timeout=60)
        response.raise_for_status()
        tmp_json = tmpdir_path / "grammar2_et.json"
        tmp_json.write_bytes(response.content)

        cache_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_json, cache_path / "grammar2_et.json")

    return cache_path


def load_jsonl(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_item(item: Dict[str, str]) -> Dict[str, str]:
    return {
        "original": item.get("original") or item.get("source") or item.get("original_string") or "",
        "correct": item.get("correct") or item.get("target") or item.get("correct_string") or ""
    }


def merge_datasets(paths: Iterable[Path], output: Path) -> None:
    seen = set()
    total = 0
    with output.open("w", encoding="utf-8") as out:
        for dataset_path in paths:
            if not dataset_path.exists():
                continue
            for item in load_jsonl(dataset_path):
                pair = normalize_item(item)
                key = (pair["original"], pair["correct"])
                if not pair["original"] or not pair["correct"] or key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1
    print(f"Merged {total} unique pairs into {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare grammar correction datasets")
    parser.add_argument("--models-dir", type=Path, default=Path("models/grammar-correction"), help="Grammar models directory")
    args = parser.parse_args()

    ensure_dir(args.models_dir)

    hf_dataset_path = download_hf_dataset(args.models_dir)
    hf_jsonl = hf_dataset_path / "grammar2_et.json"

    sources = [
        args.models_dir / "grammar_l2_train.jsonl",
        args.models_dir / "grammar_l2_test.jsonl",
        hf_jsonl,
    ]

    combined_path = args.models_dir / "grammar_combined.jsonl"
    merge_datasets(sources, combined_path)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

