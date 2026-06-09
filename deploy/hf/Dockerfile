# Stage 1: builder
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# Stage 2: runtime
FROM python:3.11-slim

# System libs (still root here)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin            /usr/local/bin

# Switch to the unprivileged user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy application files (owned by user)
COPY --chown=user main.py index.html ./
COPY --chown=user package/config.py \
                  package/visualization.py \
                  package/explainability.py \
                  ./package/

# HF Spaces listens on 7860
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]