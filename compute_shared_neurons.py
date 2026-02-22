#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute shared neurons from two detection JSON files and save to shared_VL_reasoning.json.
Usage:
  python compute_shared_neurons.py FILE_A FILE_B OUTPUT_DIR
  e.g. python compute_shared_neurons.py model_a_gsm.json model_b_multimodal.json ./compare_A_vs_B
"""

import json
import os
import sys


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_shared(file_a: str, file_b: str, output_dir: str) -> str:
    """Compute intersection of neurons from A and B, save shared_VL_reasoning.json. Returns output path."""
    data_a = load_json(file_a)
    data_b = load_json(file_b)
    directions = set(data_a.keys()) | set(data_b.keys())
    shared = {}
    for direction in directions:
        shared[direction] = {}
        a_ids = set()
        for head, neurons in data_a.get(direction, {}).items():
            a_ids.update(neurons)
        b_ids = set()
        for head, neurons in data_b.get(direction, {}).items():
            b_ids.update(neurons)
        both = a_ids & b_ids
        for head, neurons in data_a.get(direction, {}).items():
            s = [n for n in neurons if n in both]
            if s:
                shared[direction][head] = s
    out_path = os.path.join(output_dir, "shared_VL_reasoning.json")
    save_json(shared, out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_shared_neurons.py FILE_A FILE_B OUTPUT_DIR")
        sys.exit(1)
    file_a, file_b, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    path = compute_shared(file_a, file_b, output_dir)
    print(f"Saved: {path}")
