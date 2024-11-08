FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libssl-dev \
    ca-certificates \
    git \
    ffmpeg \
    libavutil-dev \
    libavformat-dev \
    libavfilter-dev \
    libavdevice-dev \
    python3-dev \
    libclang-dev \
    python3-rosbag \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /root/.cargo && \
    echo '[source.crates-io]' > /root/.cargo/config.toml && \
    echo 'registry = "https://github.com/rust-lang/crates.io-index"' >> /root/.cargo/config.toml && \
    echo 'replace-with = "ustc"' >> /root/.cargo/config.toml && \
    echo '[source.ustc]' >> /root/.cargo/config.toml && \
    echo 'registry = "https://mirrors.ustc.edu.cn/crates.io-index"' >> /root/.cargo/config.toml

WORKDIR /workspace

COPY . .