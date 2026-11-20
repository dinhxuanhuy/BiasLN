#!/usr/bin/env bash
# Drive BiasLN experiments on a Colab GPU from the local machine via the `colab` CLI.
#
#   scripts/colab/launch.sh new [GPU]          # allocate VM (default GPU: L4)
#   colab drivemount -s $SESSION               # ONE interactive step, done by a human
#   scripts/colab/launch.sh push               # upload the working tree to /content/BiasLN
#   scripts/colab/launch.sh setup              # background: extract data from Drive, deps, audit
#   scripts/colab/launch.sh start <script> [ENV=VAL ...]
#        e.g. start scripts/run_protocol.sh SEEDS="1024 1 2" AMP=bf16
#   scripts/colab/launch.sh status             # tail job log, GPU usage
#   scripts/colab/launch.sh fetch [dest]       # download results/ (+ small logs) as a tarball
#   scripts/colab/launch.sh stop               # release the VM (stops billing)
#
# Long jobs run detached on the VM (nohup) because `colab exec` returns after
# --timeout seconds; poll them with `status`. Results are also rsynced to
# $DRIVE_SYNC on Drive after every finished (model, seed), so a lost VM only
# loses the run in progress; re-running `start` resumes.
set -euo pipefail

SESSION=${SESSION:-biasln}
REMOTE=${REMOTE:-/content/BiasLN}
DRIVE_SYNC=${DRIVE_SYNC:-/content/drive/MyDrive/BiasLN_runs}
COLAB=${COLAB:-colab}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

rexec() {  # rexec <python code> [timeout]; retries transient "Connection was lost"
  local out attempt
  for attempt in 1 2 3 4; do
    if out=$(printf '%s\n' "$1" | $COLAB exec -s "$SESSION" --timeout "${2:-120}" 2>&1) \
        && ! grep -q "Connection was lost" <<<"$out"; then
      grep -v '^\[colab\]' <<<"$out"
      return 0
    fi
    echo "[launch] exec failed (attempt $attempt), retrying..." >&2
    sleep $((attempt * 10))
  done
  echo "$out" | tail -5 >&2
  return 1
}

background() {  # background <logfile> <bash command>
  rexec "import subprocess
subprocess.Popen(['bash', '-lc', '''cd $REMOTE && ( $2 ) > $1 2>&1'''], start_new_session=True)
print('started; log: $1')"
}

cmd=${1:-help}; shift || true
case "$cmd" in
  new)
    $COLAB new -s "$SESSION" --gpu "${1:-L4}"
    echo "Next: colab drivemount -s $SESSION   (interactive, approve in the browser)"
    ;;
  push)
    tarball=$(mktemp --suffix=.tgz)
    (cd "$ROOT" && git ls-files -co --exclude-standard \
        | grep -vE '\.(zip|pdf|ipynb|pth|pt)$|^(datasets/rgb|logs|results)/' \
        | tar -czf "$tarball" -T -)
    echo "uploading $(du -h "$tarball" | cut -f1) ..."
    $COLAB upload -s "$SESSION" "$tarball" /content/biasln_src.tgz
    rexec "import subprocess
print(subprocess.run('mkdir -p $REMOTE && tar -xzf /content/biasln_src.tgz -C $REMOTE && ls $REMOTE', shell=True, capture_output=True, text=True).stdout)"
    rm -f "$tarball"
    ;;
  setup)
    background /content/setup.log "bash scripts/colab/setup_vm.sh"
    echo "poll with: $0 status setup"
    ;;
  start)
    script=${1:?usage: start <script> [ENV=VAL ...]}; shift
    envs="SYNC=$DRIVE_SYNC"
    for kv in "$@"; do envs="$envs $(printf '%q' "$kv")"; done
    background /content/job.log "env $envs bash $script"
    ;;
  status)
    log=/content/job.log; [ "${1:-}" = setup ] && log=/content/setup.log
    rexec "import subprocess
print(subprocess.run('tail -n ${LINES_TAIL:-25} $log; echo; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; ps aux | grep -E \"train.py|test.py|setup_vm\" | grep -v grep | wc -l | xargs echo running_procs:', shell=True, capture_output=True, text=True).stdout)"
    ;;
  fetch)
    dest=${1:-$ROOT}
    rexec "import subprocess
print(subprocess.run('cd $REMOTE && tar --exclude=\"*.pickle\" --exclude=\"metric_board\" -czf /content/biasln_results.tgz results logs/*/*/run_info.json logs/*/*/config.yaml logs/*/*/training.log logs/*/*/epoch_table.json 2>/dev/null; ls -la /content/biasln_results.tgz', shell=True, capture_output=True, text=True).stdout)" 300
    $COLAB download -s "$SESSION" /content/biasln_results.tgz "$dest/biasln_results.tgz"
    (cd "$dest" && tar -xzf biasln_results.tgz && rm biasln_results.tgz)
    echo "results extracted under $dest/results"
    ;;
  stop)
    $COLAB stop -s "$SESSION"
    ;;
  *)
    sed -n '2,20p' "$0"
    ;;
esac
