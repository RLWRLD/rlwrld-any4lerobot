#!/bin/bash
# Cloud-init user data: everything a fresh instance needs before the repo arrives.
#
# Deliberately does not fetch the repo. How that gets here depends on where it is
# staged and what the instance profile can read -- see README.md. This script
# prepares the machine and stops.
set -uxo pipefail

REPO_DIR=${REPO_DIR:-/opt/oxe/any4lerobot}
SCRATCH=${SCRATCH:-/scratch}

# ---------------------------------------------------------------- packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends ffmpeg curl ca-certificates

# uv, which is what resolves and runs the environment
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

# ---------------------------------------------------------------- layout
# raw sources, converter output, and the step records that let a run resume
mkdir -p "$SCRATCH"/{raw,out,state,log,delivered}
mkdir -p "$REPO_DIR"
chown -R ubuntu:ubuntu "$SCRATCH" /opt/oxe

# The region has to be set for the machine, not just for a shell: the orchestrator
# shells out to `aws s3 sync` and inherits nothing otherwise.
echo 'AWS_DEFAULT_REGION=us-east-1' >> /etc/environment

# ---------------------------------------------------------------- marker
# preflight.sh looks for this before it believes the machine was ever prepared
date -u +%FT%TZ > /opt/oxe/BOOTSTRAP_DONE
echo "bootstrap complete"
