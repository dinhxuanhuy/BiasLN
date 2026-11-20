#!/usr/bin/env bash
# Wait until the seed-1024 main comparison is complete, then stop run_all and start run_decide.
cd /content/BiasLN
until [ -f results/val/biasln_asy_soup/seed1024/metrics.json ]; do sleep 60; done
pkill -f run_all.sh; pkill -f run_protocol.sh; pkill -f training/train.py; pkill -f training/test.py
sleep 20
rm -rf logs/val/biasln_s1   # partially started run (run_protocol restarts it cleanly anyway)
exec env SYNC=/content/drive/MyDrive/BiasLN_runs bash scripts/colab/run_decide.sh
