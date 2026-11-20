#!/usr/bin/env bash
# Keeps a Colab VM assigned: Colab reclaims a runtime that receives no request for ~30 min,
# even while a detached job is training on it. Run locally in the background:
#   SESSION=biasln6 bash scripts/colab/keepalive.sh
S=${SESSION:?}; P=${PERIOD:-240}
while :; do
  r=$(echo 'print("alive")' | timeout 120 colab exec -s "$S" --timeout 60 2>&1 | grep -c '^alive')
  echo "$(date '+%F %T') $S alive=$r"
  colab ls 2>&1 | grep -q "$S" || { echo "$(date '+%F %T') session $S gone"; exit 1; }
  sleep "$P"
done
