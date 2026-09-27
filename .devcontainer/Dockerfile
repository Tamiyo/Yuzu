# Yuzu dev container — Ubuntu 24.04 LTS + Rust + Python + zsh/starship.
FROM ubuntu:24.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG DEBIAN_FRONTEND=noninteractive

# Base toolchain: a C linker for rustc, protoc for the substrait crate's
# build script, and Python (headers + venv) for the PyO3 bindings.
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  less \
  locales \
  pkg-config \
  protobuf-compiler \
  libzstd-dev \
  zlib1g-dev \
  python3-dev \
  python3-pip \
  python3-venv \
  sudo \
  unzip \
  wget \
  zsh \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*

# LLVM/MLIR 22 from apt.llvm.org — the dialect library (yuzu_ir_sys) and the
# melior bindings build against this exact major version; the pin lives in
# .cargo/config.toml as MLIR_SYS_220_PREFIX / TABLEGEN_220_PREFIX.
RUN wget -qO /etc/apt/trusted.gpg.d/apt-llvm-org.asc https://apt.llvm.org/llvm-snapshot.gpg.key \
  && echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" > /etc/apt/sources.list.d/llvm-22.list \
  && apt-get update && apt-get install -y --no-install-recommends \
  clang-22 \
  libmlir-22-dev \
  libpolly-22-dev \
  llvm-22-dev \
  llvm-22-tools \
  mlir-22-tools \
  && rm -rf /var/lib/apt/lists/*

# Node.js 22 LTS from NodeSource — builds the VS Code extension's language
# client in editors/vscode.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# Starship prompt (matches the host setup).
RUN curl -fsSL https://starship.rs/install.sh | sh -s -- -y

# Ubuntu 24.04 ships with a pre-existing UID 1000 `ubuntu` user — drop it so
# our dev user can claim that UID and match bind-mount permissions.
RUN if getent passwd ubuntu >/dev/null; then userdel -r ubuntu 2>/dev/null || true; fi \
  && groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/zsh ${USERNAME} \
  && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
  && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}

# Oh-my-zsh — unattended so it doesn't drop into an interactive shell.
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Rust via rustup. Installed as the dev user so the toolchain lives in their
# home (~/.cargo, ~/.rustup) and bind-mount permissions line up.
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup \
  CARGO_HOME=/home/${USERNAME}/.cargo \
  PATH=/home/${USERNAME}/.cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --default-toolchain stable --profile default \
  && rustup component add rust-src clippy rustfmt

COPY --chown=${USERNAME}:${USERNAME} zshrc /home/${USERNAME}/.zshrc
COPY --chown=${USERNAME}:${USERNAME} starship.toml /home/${USERNAME}/.config/starship.toml

ENV LANG=en_US.UTF-8 \
  LC_ALL=en_US.UTF-8

WORKDIR /workspaces/yuzu