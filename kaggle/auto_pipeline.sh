#!/usr/bin/env bash
# Prep -> check -> push L4 -> monitor -> download, unattended.
# The L4 kernel is pushed ONLY if the prep report passes every check.
#   bash kaggle/auto_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.."
K=${KAGGLE:-kaggle}
WORK=${WORK:-kaggle/_out}
mkdir -p "$WORK/prep" "$WORK/l4"
st() { $K kernels status "$1" 2>&1 | tail -1; }
wait_kernel() {  # wait_kernel <ref> <poll seconds>
  while :; do s=$(st "$1"); echo "$(date '+%F %T') $1: $s"
    case "$s" in *COMPLETE*) return 0;; *ERROR*|*CANCEL*) return 1;; esac; sleep "$2"; done; }

echo "== 1. wait for prep"
wait_kernel ${PREP:-xuanhuydinh/biasln-prep2} 60 || { echo "!! prep failed"; exit 1; }
$K kernels output ${PREP:-xuanhuydinh/biasln-prep2} -p "$WORK/prep" --file-pattern 'prep_report.json|.*\.log' -q 2>/dev/null \
  || $K kernels output ${PREP:-xuanhuydinh/biasln-prep2} -p "$WORK/prep" -q

echo "== 2. check prep report against the counts of the Colab runs"
python3 - "$WORK/prep/prep_report.json" <<'EOF' || { echo "!! prep check failed: L4 NOT pushed"; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
# test videos / frames of every set in the Colab runs (results metrics.json n_videos, n_frames)
expect = {"Celeb-DF-v2": (518, 16420), "DeepFakeDetection": (3429, 102541), "DFDC": (4704, 132116),
          "DFDCP": (654, 17222), "uniface_ff": (280, 8944), "blendface_ff": (280, 8946),
          "e4s_ff": (255, 8142), "facedancer_ff": (280, 8937), "fsgan_ff": (280, 8934),
          "inswap_ff": (258, 8241), "simswap_ff": (280, 8947), "FaceForensics++": (700, 22388)}
bad = []
for name, (nv, nf) in expect.items():
    e = r.get(name)
    if not isinstance(e, dict):
        bad.append(f"{name}: absent"); continue
    if e["missing_frames"]:
        bad.append(f"{name}: {e['missing_frames']} missing frames")
    got = e["splits"].get("test", [0, 0])
    tol = 0.02 * nf  # small frame-count differences are tolerated, video counts are not
    if got[0] != nv or abs(got[1] - nf) > tol:
        bad.append(f"{name}: test {got} vs Colab {(nv, nf)}")
tr = r.get("FaceForensics++", {}).get("splits", {})
print("FF++ splits:", tr)
if not (3400 <= tr.get("train", [0])[0] <= 3700 and 650 <= tr.get("val", [0])[0] <= 720):
    bad.append(f"FF++ train/val video counts look wrong: {tr}")
print("\n".join(bad) if bad else "all checks passed")
sys.exit(1 if bad else 0)
EOF

echo "== 3. push L4 kernel"
$K kernels push -p kaggle/train || { echo "!! push failed"; exit 1; }
sleep 120

echo "== 4. monitor (smoke test runs first; failure stops the kernel early)"
wait_kernel xuanhuydinh/biasln-l4-seeds 900; rc=$?
echo "== 5. download outputs (rc=$rc)"
$K kernels output xuanhuydinh/biasln-l4-seeds -p "$WORK/l4" -q
ls -R "$WORK/l4" | head -50
echo "== auto_pipeline finished rc=$rc"
