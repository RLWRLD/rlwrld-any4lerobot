#!/bin/bash
# Refuse to start an unattended run on a machine that will fail hours in.
#
# Every check here stands for a run that was lost: a truncated image, a short
# sync, a missing tfds metadata file. They are cheap and they all fail loudly.
#
#   ./preflight.sh                 # machine and environment only
#   ./preflight.sh taco_play toto  # also check these datasets are staged
#
# Exit status is the number of failures.
set -uo pipefail

REPO_DIR=${REPO_DIR:-/opt/oxe/any4lerobot}
SCRATCH=${SCRATCH:-/scratch}
UV=${UV:-/usr/local/bin/uv}
MARKER=${BOOTSTRAP_MARKER:-/opt/oxe/BOOTSTRAP_DONE}
# In the image the environment is already on PATH, so there is nothing for `uv run`
# to resolve; on a machine built by user-data.sh there is. robomind_v2 alongside
# openx: robomind_v2_2lerobot/robomind_v2_h5.py needs ray, which openx alone does
# not pull in.
PYTHON=${PYTHON:-"$UV run --extra openx --extra robomind_v2 python"}

failures=0
pass() { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; failures=$((failures + 1)); }
warn() { printf '  warn  %s\n' "$1"; }

echo "== machine"

cores=$(nproc)
gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
per_core=$((gb * 10 / cores))
if [ "$per_core" -ge 35 ]; then
  pass "$cores cores, ${gb} GB (${per_core:0:1}.${per_core: -1} GB per core)"
else
  warn "$cores cores, ${gb} GB -- under 4 GB per core; cap workers below the core count"
fi

free_gb=$(df -BG --output=avail "$SCRATCH" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -ge 200 ]; then
  pass "${free_gb} GB free on $SCRATCH"
else
  fail "${free_gb:-0} GB free on $SCRATCH -- a single source can be 48 GB"
fi

[ -f "$MARKER" ] && pass "bootstrap marker" || fail "no bootstrap marker at $MARKER"
command -v ffmpeg >/dev/null && pass "ffmpeg" || fail "ffmpeg not installed"
[ -x "$UV" ] && pass "uv" || fail "uv not at $UV"

echo "== repo"

if [ -d "$REPO_DIR" ]; then
  pass "repo at $REPO_DIR"
else
  fail "no repo at $REPO_DIR"
  exit $failures
fi

cd "$REPO_DIR" || exit $failures

# An image captured with --no-reboot truncates whatever was in flight. The venv
# imports fine until it does not, so look for the zero-length files directly.
# __init__.py is exempt: this repo has eight that are empty on purpose.
empty=$(find . -name '*.py' -size 0 -not -name '__init__.py' -not -path './.venv/*' | wc -l)
[ "$empty" -eq 0 ] && pass "no truncated source files" \
  || fail "$empty source files are zero bytes -- the image was captured unquiesced"

# Counting every empty file in the virtualenv does not work: a healthy one has
# about 150, nearly all of them dist-info REQUESTED markers, which are empty by
# definition. Compiled libraries never are, and they are what truncation destroys.
empty_libs=$(find .venv -type f \
  \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' -o -name '*.pyd' \) \
  -size 0 2>/dev/null | wc -l)
[ "$empty_libs" -eq 0 ] && pass "venv intact" \
  || fail "$empty_libs compiled libraries in .venv are zero bytes -- rebuild it with uv sync"

if [ -s uv.lock ]; then
  pass "uv.lock ($(stat -c %s uv.lock) bytes)"
else
  fail "uv.lock is missing or empty -- without it the environment is whatever resolved today"
fi

# macOS tar writes these; dataset_registry.available() reads them as specs
apple=$(find . -name '._*' | wc -l)
[ "$apple" -eq 0 ] && pass "no AppleDouble files" \
  || fail "$apple ._* files -- unpacked from a macOS tar; delete them"

echo "== environment"

if $PYTHON -c "
import lerobot, tensorflow, tensorflow_datasets, datatrove
from generic_converter.pipeline import aggregate_tasks
from openx2lerobot.adapter import OpenXAdapter
" >/dev/null 2>&1; then
  pass "imports resolve"
else
  fail "imports do not resolve -- run: $UV sync --locked --extra openx --extra robomind_v2 --group dev"
fi

if $PYTHON -c "import tensorflow_graphics" >/dev/null 2>&1; then
  pass "tensorflow_graphics"
else
  warn "no tensorflow_graphics -- furniture_bench and iamlab_cmu_pickup_insert will fail"
fi

[ $# -eq 0 ] && { echo; echo "$failures failure(s)"; exit $failures; }

echo "== staged datasets"

for name in "$@"; do
  raw="$SCRATCH/raw/$name"
  if [ ! -d "$raw" ]; then
    fail "$name: nothing staged at $raw"
    continue
  fi

  # tfds wants <data_dir>/<name>/<version>/; --raw-dir has to name the version
  version=$(find "$raw" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | head -1)
  if [ -z "$version" ]; then
    fail "$name: no version directory -- --raw-dir must point at <name>/<version>"
    continue
  fi
  dir="$raw/$version"

  for f in dataset_info.json features.json; do
    [ -s "$dir/$f" ] && pass "$name/$version: $f" || fail "$name/$version: $f missing"
  done

  # the shard name says how many there should be: ...-00000-of-00511
  shard=$(find "$dir" -name '*train.*record*' -printf '%f\n' 2>/dev/null | head -1)
  if [ -n "$shard" ]; then
    expected=$(echo "$shard" | sed 's/.*-of-0*\([0-9]*\)$/\1/')
    actual=$(find "$dir" -name '*train.*record*' | wc -l)
    if [ "$actual" -eq "$expected" ]; then
      pass "$name: $actual/$expected train shards"
    else
      fail "$name: $actual of $expected train shards -- the sync did not finish"
    fi
    case "$shard" in
      *array_record*) warn "$name: array_record format; as_dataset() cannot read it" ;;
    esac
  else
    fail "$name: no train shards under $dir"
  fi

  delivered="$SCRATCH/delivered/$name/$name/meta"
  [ -d "$delivered" ] && pass "$name: delivered copy" \
    || warn "$name: no delivered copy -- there is nothing to compare against"
done

echo
echo "$failures failure(s)"
exit $failures
