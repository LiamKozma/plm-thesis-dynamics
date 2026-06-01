# Stage 1: Builder
# We use a specific version of PyTorch to ensure reproducibility
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

# Set non-interactive to avoid apt hangs during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We copy the requirements file first to leverage Docker's cache layers
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (Slimmer image)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Liam Kozma"
LABEL version="1.0"
LABEL description="ESM-2 Fine-tuning environment for Sapelo2"

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Update PATH to include user installed packages
ENV PATH=/root/.local/bin:$PATH

# Install Apptainer compatibility tools (useful for cluster debugging)
# and basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your source code into the container
COPY . /app

# Default command (can be overridden by Nextflow later)
CMD ["python", "src/generate_simulation.py"]
