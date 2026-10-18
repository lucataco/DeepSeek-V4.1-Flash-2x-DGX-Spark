# TensorFold's DeepSeek-V4.1 TP2 Python engine (bertholomus/TensorFold, branch deepseek-v41-tp2) on NVIDIA's PyTorch
# container, the base its author serves on. torch / triton come from the container; pip adds only the engine's own deps.
FROM nvcr.io/nvidia/pytorch:26.07-py3
ARG TF_COMMIT=8e0966e2b09250abe2f6d9cc51edc20630e6eb1a
# The engine hardcodes 8 images per request and a 32 MiB body; patches/ makes both configurable, defaults unchanged.
COPY patches/ /tmp/patches/
RUN git clone -q https://github.com/bertholomus/TensorFold /opt/TensorFold && \
    git -C /opt/TensorFold checkout -q ${TF_COMMIT} && \
    git -C /opt/TensorFold apply /tmp/patches/*.patch && \
    pip install --no-cache-dir -e "/opt/TensorFold[vision]" && \
    python3 -c "import torch, tensorfold; print('tensorfold', tensorfold.__file__, 'torch', torch.__version__)"
LABEL org.opencontainers.image.title="tensorfold-dsv41" org.opencontainers.image.revision="${TF_COMMIT}"
