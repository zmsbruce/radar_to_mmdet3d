# Use the official ROS Noetic base image (ros-core)
FROM ros:noetic-ros-core

# Install necessary system dependencies
RUN apt-get update && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
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
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Rust with rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Set up the Rust environment
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory to /workspace
WORKDIR /workspace

# Copy the local code into the container
COPY . .

# Source ROS setup script so ROS tools are available in the shell
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Set the default command to bash
CMD ["bash"]