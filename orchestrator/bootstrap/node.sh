#!/bin/bash
# One node's share of a run: fetch it, build it, publish it, exit.
#
#   docker run --rm -e DATASETS="taco_play toto" -v /scratch:/scratch any4lerobot
#
# DATASETS   this node's share -- names from dataset_registry, space or newline
#            separated. Required; a node with no share is a mistake, not a no-op.
# ENV_NAME   which environment config to use (default: ec2)
# NIC_RATE   the instance's actual NIC rate, e.g. 100Gb/s. Declaring more than the
#            link has measured *slower* (41.4 -> 37.4 Gbps), so it is not guessed.
# SKIP_PREFLIGHT  set to 1 to start anyway. For debugging; a scheduled run should
#            never set it.
set -uo pipefail

# No apostrophe in this message: bash parses a quote inside ${var:?...} even within
# double quotes, and the script then fails to parse at all.
: "${DATASETS:?set DATASETS to the share of the registry this node handles}"
ENV_NAME=${ENV_NAME:-ec2}
REPO_DIR=${REPO_DIR:-/app}
NIC_RATE=${NIC_RATE:-}

cd "$REPO_DIR" || exit 1

# Transfer settings are a property of the machine, not of the code, which is why
# they are set here and not committed in a config. The numbers are measured -- see
# the tables in README.md.
aws configure set default.s3.preferred_transfer_client crt
aws configure set default.s3.multipart_chunksize 64MB
if [ -n "$NIC_RATE" ]; then
  aws configure set default.s3.target_bandwidth "$NIC_RATE"
else
  echo "node: NIC_RATE unset, leaving target_bandwidth alone" >&2
fi

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  # Cheap, and it is the difference between failing now and failing in four hours.
  if ! ./orchestrator/bootstrap/preflight.sh $DATASETS; then
    echo "node: preflight failed; not starting" >&2
    exit 2
  fi
fi

# One --dataset per name. `run` is fetch, build and publish in order, and each step
# records what it did, so a restarted node resumes rather than repeating.
args=()
for name in $DATASETS; do
  args+=(--dataset "$name")
done

echo "node: starting $ENV_NAME on ${#args[@]} argument(s): $DATASETS"
exec python -m orchestrator run --env "$ENV_NAME" "${args[@]}"
