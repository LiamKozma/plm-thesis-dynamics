# ==============================================================================
# MULTI-STAGE BUILD
# ------------------------------------------------------------------------------
# A multi-stage build uses MORE THAN ONE `FROM` statement. Each `FROM` starts a
# new, independent stage. We compile/install everything we need in a first
# "builder" stage, then copy ONLY the finished artifacts into a clean second
# "runtime" stage. The heavy build tooling (compilers, git, caches) stays behind
# in the builder and never lands in the final image — giving a smaller, cleaner,
# more secure image to actually ship and run.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
# Pin an EXACT base image (PyTorch + CUDA 12.1 + cuDNN8) so every build is
# byte-for-byte reproducible. `AS builder` names this stage so Stage 2 can copy
# from it by name.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

# Run apt in non-interactive mode so package installs never block waiting for a
# prompt (e.g. tzdata) and hang the build.
ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level build dependencies. Some Python packages compile C/C++
# extensions on install, so they need a compiler (build-essential), git, wget.
# `--no-install-recommends` skips optional extras, and deleting the apt lists in
# the SAME RUN layer keeps that cache out of the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# --- LAYER-CACHE OPTIMIZATION ---
# Copy requirements.txt BY ITSELF, BEFORE the rest of the source code. Docker
# caches each instruction as a layer and invalidates a layer (and everything
# after it) only when its inputs change. Dependencies change rarely; source code
# changes constantly. By copying just requirements.txt first, the expensive
# `pip install` layer is reused on every rebuild UNLESS requirements.txt itself
# changes — so editing your Python code does NOT trigger a full reinstall.
COPY requirements.txt .

# `--user` installs the packages into the per-user prefix (/root/.local) instead
# of the global site-packages. That gathers EVERYTHING pip installs under a
# single self-contained directory tree, which Stage 2 can then copy out in one
# `COPY --from` — the key trick that makes the multi-stage handoff clean.
# `--no-cache-dir` tells pip not to keep its download cache, shrinking the layer.
RUN pip install --user --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Runtime (slimmer final image)
# ------------------------------------------------------------------------------
# Start over from the SAME clean base image. Note: no build-essential/git/wget
# here — none of the builder's toolchain or caches carry over.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Image metadata (queryable via `docker inspect`).
LABEL maintainer="Liam Kozma"
LABEL version="1.0"
LABEL description="ESM-2 Fine-tuning environment for Sapelo2"

# Pull ONLY the installed Python packages from the builder stage. Because Stage 1
# used `pip install --user`, everything lives under /root/.local — so this single
# COPY transplants the full dependency set without any build tooling.
COPY --from=builder /root/.local /root/.local

# Put the `--user`-installed package binaries (e.g. console entry points) on PATH
# so they are runnable as bare commands.
ENV PATH=/root/.local/bin:$PATH

# Minimal runtime utilities: TLS root certs (for HTTPS model/data downloads) and
# curl. Same `--no-install-recommends` + apt-list cleanup discipline as above.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory for all subsequent commands and the default runtime cwd.
WORKDIR /app

# Copy the project source code in last — after dependencies — so frequent code
# edits don't bust the cached dependency layers above.
COPY . /app

# Default command. Nextflow (or `docker run ...`) overrides this per process.
CMD ["python", "src/generate_simulation.py"]
