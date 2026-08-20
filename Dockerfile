# One node of a conversion run: fetch a share of the datasets, rebuild them, publish.
#
# A container rather than an AMI because nothing about a node's *data* is baked in --
# each node downloads its own share at run time -- which leaves only the environment,
# and that is the part an AMI held badly. The AMI this replaces was captured with
# --no-reboot, and came back with 3,945 zero-length files in its virtualenv and a
# source directory missing whole modules, silently. Layers are content-addressed: a
# damaged one fails to pull rather than booting and misbehaving four hours later.
#
# Build:  see orchestrator/bootstrap/README.md -- it needs a foundry build context
# Run:    docker run --rm -e DATASETS="taco_play toto" -v /scratch:/scratch any4lerobot

FROM python:3.12-slim-bookworm

# uv comes from its own published image rather than an install script: a pinned tag
# is one less thing resolved at build time.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

# ffmpeg does the video work; curl and unzip fetch the AWS CLI below. git is not
# optional: lerobot is pinned to a commit rather than a release (see pyproject), so
# uv clones it, and without git the sync fails at resolution.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The orchestrator shells out to `aws s3 sync` (see orchestrator/transfer.py -- a
# coordinated CRT client was built for this once and measured slower than the CLI).
# It has to be v2: the CRT transfer client and target_bandwidth do not exist in v1,
# and Debian ships v1.
ARG TARGETARCH
RUN case "$TARGETARCH" in \
      amd64) arch=x86_64 ;; \
      arm64) arch=aarch64 ;; \
      *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

WORKDIR /app

# Dependencies before source, so editing a converter rebuilds one cheap layer rather
# than resolving and downloading TensorFlow again.
#
# --locked refuses to proceed if uv.lock disagrees with pyproject.toml. That is the
# point of the image: two nodes can only be claimed identical if what decides their
# contents is pinned, so a drifted lock should stop the build, not be quietly fixed.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra openx --no-dev

# Then swap torch for the CPU build. lerobot pins torch to the cu128 index for
# linux in its own [tool.uv.sources], which our pyproject cannot outvote -- uv
# reports the two indexes as a conflict and refuses to resolve. Reinstalling
# afterwards is the escape hatch lerobot itself documents.
#
# Worth 4.7 GB: cuda-toolkit, cudnn, nccl, cusparselt and triton are all there to
# drive a GPU, and m7i has none. Versions are pinned to what uv.lock resolved, so
# this changes the build of torch and nothing else.
#
# The nvidia packages are dependencies of the CUDA build and nothing removes them
# when it goes, so they are uninstalled by name.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv --reinstall \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
 && orphans=$(uv pip list --python /app/.venv --format json \
      | /app/.venv/bin/python -c "import json,sys; print(' '.join(p['name'] for p in json.load(sys.stdin) if p['name'].startswith(('nvidia-','cuda-')) or p['name']=='triton'))") \
 && if [ -n "$orphans" ]; then uv pip uninstall --python /app/.venv $orphans; fi \
 && /app/.venv/bin/python -c "import torch, torchvision; print('torch', torch.__version__)"

# The Foundry CLI and SDK, for the delivered copies a comparison is measured against
# and for publishing a preprocessed dataset back. Not on PyPI and not in an internal
# index, so the source arrives as a named build context:
#
#   docker buildx build --build-context foundry=../rlwrld-foundry \
#     --build-arg FOUNDRY_REVISION=$(git -C ../rlwrld-foundry rev-parse HEAD) ...
#
# The revision is checked against foundry-cli.pin rather than trusted, so which CLI
# an image carries is a reviewed fact and not whatever the builder happened to have
# checked out. Bumping it is an edit to that file.
#
# Installed the way foundry's own CI installs it: the locked dependency set, then each
# package with --no-deps.
#
# The workspace root pyproject.toml is copied with them and the packages/ path is kept,
# which is not tidiness. foundry-cli's pyproject says
# `foundry-client = { workspace = true }`, and uv parses that whatever it is asked to
# do -- --no-deps does not skip it. Without a workspace root above the two packages the
# build fails with "references a workspace ... but is not a workspace member". The root
# globs `packages/*`, so a copy holding only these two resolves to exactly them.
ARG FOUNDRY_REVISION
COPY foundry-cli.pin /tmp/foundry-cli.pin
COPY --from=foundry pyproject.toml /tmp/foundry/pyproject.toml
COPY --from=foundry packages/foundry-cli /tmp/foundry/packages/foundry-cli
COPY --from=foundry packages/foundry-client /tmp/foundry/packages/foundry-client
RUN --mount=type=cache,target=/root/.cache/uv \
    pinned=$(tr -d " \t\r\n" < /tmp/foundry-cli.pin) \
 && if [ "$FOUNDRY_REVISION" != "$pinned" ]; then \
      echo "FOUNDRY_REVISION=${FOUNDRY_REVISION:-(unset)} does not match foundry-cli.pin=$pinned" >&2; \
      exit 1; \
    fi \
 && uv pip install --python /app/.venv \
      --requirement /tmp/foundry/packages/foundry-cli/requirements.lock \
 && uv pip install --python /app/.venv --no-deps /tmp/foundry/packages/foundry-client \
 && uv pip install --python /app/.venv --no-deps /tmp/foundry/packages/foundry-cli \
 && mkdir -p /opt/foundry && echo "$FOUNDRY_REVISION" > /opt/foundry/REVISION \
 && rm -rf /tmp/foundry /tmp/foundry-cli.pin \
 && /app/.venv/bin/foundry --version

COPY . .

# Where the orchestrator stages sources and writes output. Mount a volume over it --
# a node handling taco_play alone needs 48 GB, which is not container-sized.
# FOUNDRY_URL is the *internal* ALB. The name foundry.internal.rlwrld.ai resolves to
# the public ALB's addresses, which do not hairpin from inside the VPC -- a node using
# it gets a 15-second timeout. Both values are the ones rlwrld-foundry's own
# deploy/agents/targets.yaml declares for this cluster, under `skt`. http, not https:
# the internal ALB has one listener and it is plain HTTP on 80.
ENV SCRATCH=/scratch \
    REPO_DIR=/app \
    UV=/usr/local/bin/uv \
    PATH=/app/.venv/bin:$PATH \
    PYTHON=python \
    FOUNDRY_URL=http://internal-rlwrld-foundry-api-425985869.us-east-1.elb.amazonaws.com/api \
    FOUNDRY_HOME_LOCATION=aws-ssot
RUN mkdir -p /scratch/raw /scratch/out /scratch/state /scratch/log /scratch/delivered

# preflight.sh looks for this before believing the machine was prepared.
RUN mkdir -p /opt/oxe && date -u +%FT%TZ > /opt/oxe/BOOTSTRAP_DONE

ENTRYPOINT ["/app/orchestrator/bootstrap/node.sh"]
