#!/bin/bash
uv run basedpyright \
  denoising/main.py \
  denoising/src/models/gnn.py \
  denoising/src/models/attention.py \
  denoising/src/data/sbm.py \
  2>&1 | grep -A 2 "denoising/"
