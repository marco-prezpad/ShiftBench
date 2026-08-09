FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

# Copy only pyproject.toml for dependency resolution
COPY pyproject.toml .

# Create empty README.md to satisfy setuptools if needed
RUN touch README.md

# Install PyTorch first (heavy, cacheable)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies from pyproject.toml without installing the project itself
RUN pip install --no-cache-dir numpy scipy pandas scikit-learn \
    alibi-detect evidently pyyaml tqdm matplotlib seaborn \
    pytest pytest-cov jupyter black ruff

# Copy the rest of the project
COPY . .

# Install the project in editable mode
RUN pip install -e .[all]

CMD ["bash"]