#!/usr/bin/env bash
set -euo pipefail

cd /data/filter_code/data_filter
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RUN_DIR="/data/xvla_market_bottle/processing/data_filter_v2_final_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" | tee "$RUN_DIR/run_all.log"

run_one() {
  local name="$1"
  shift
  mkdir -p "$RUN_DIR/$name"
  echo "=== $name ===" | tee -a "$RUN_DIR/run_all.log"
  python scripts/run_filter.py "$@" --out "$RUN_DIR/$name" 2>&1 | tee "$RUN_DIR/$name/run.log"
}

run_one raw_pika_extra \
  --gate raw \
  --source pika \
  --root /data/xvla_market_bottle/pika_extra

run_one raw_umi_scanqr_synced \
  --gate raw \
  --source pika \
  --root /data/xvla_market_bottle/umi/umi_scanqr_synced

run_one raw_market_bottle_tele \
  --gate raw \
  --source teleop \
  --root /data/xvla_market_bottle/market_bottle_tele

run_one raw_nas_teleop_full_raw \
  --gate raw \
  --source teleop \
  --root /data/xvla_market_bottle/nas_teleop/full_raw

run_one processed_all \
  --gate processed \
  --root \
    /data/xvla_market_bottle/nas_teleop/xvla_20d_full \
    /data/xvla_market_bottle/umi/umi_scanqr_synced \
    /data/xvla_market_bottle/pika_extra

echo "DONE" | tee -a "$RUN_DIR/run_all.log"
