#!/usr/bin/env bash
# =============================================================
# MoGe (moge_completion) 一键配置脚本
#   适用: Linux + NVIDIA GPU + Anaconda
#   说明: 完整复现本机已验证的 MoGe-3/MoGe-2 环境
#   用法: bash setup.sh
# =============================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-moge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CODE_DIR="${CODE_DIR:-$HOME/code}"
GH_PROXY="${GH_PROXY:-https://ghfast.top/https://github.com/}"

# 固定提交（与官方 pyproject.toml 一致）
UTILS3D_COMMIT="62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
PIPELINE_COMMIT="1c511390d90226c00c101f34b84df26a0f8789b4"
FLEXGEMM_COMMIT="b2fadb29d41846c7981ade6801ffc689fae119cf"

log() { echo -e "\033[1;32m[setup]\033[0m $*"; }

# ---------- 0. 前置检查 ----------
command -v conda >/dev/null || { echo "需要 Anaconda/Miniconda"; exit 1; }
command -v git >/dev/null || { echo "需要 git"; exit 1; }
CONDA_BASE="$(conda info --base)"

# ---------- 1. 创建 conda 环境 ----------
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  log "conda 环境 $ENV_NAME 已存在，跳过创建"
else
  log "创建 conda 环境 $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
log "已激活环境 $ENV_NAME: $(python --version)"

# ---------- 2. 安装 cu128 PyTorch ----------
# 注意: 必须 cu128。cu130 需要驱动>=580；本类机器(驱动 575.x)只能跑 cu128。
#       你的驱动若 >=580，可换成 cu130。
log "安装 torch/torchvision (cu128) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ---------- 3. 克隆 git 依赖(固定 commit) ----------
clone_repo() { # $1=url  $2=dir  $3=commit
  local url="$1" dir="$2" commit="$3"
  if [ -d "$dir/.git" ]; then
    log "已有 $dir，拉取并切到固定 commit"
    ( cd "$dir" && git fetch -q --all && git checkout -q "$commit" )
  elif git clone -q "$url" "$dir"; then
    log "直连克隆成功: $url"
    ( cd "$dir" && git checkout -q "$commit" )
  else
    log "直连失败，改用代理: ${GH_PROXY}${url#https://github.com/}"
    rm -rf "$dir"
    git clone -q "${GH_PROXY}${url#https://github.com/}" "$dir"
    ( cd "$dir" && git checkout -q "$commit" )
  fi
}

mkdir -p "$CODE_DIR"
clone_repo "https://github.com/EasternJournalist/utils3d-moge.git" "$CODE_DIR/utils3d-moge" "$UTILS3D_COMMIT"
clone_repo "https://github.com/EasternJournalist/pipeline.git"      "$CODE_DIR/pipeline"      "$PIPELINE_COMMIT"
clone_repo "https://github.com/JeffreyXiang/FlexGEMM.git"           "$CODE_DIR/FlexGEMM"      "$FLEXGEMM_COMMIT"

# ---------- 4. 安装 git 依赖 ----------
# --no-build-isolation: 直接用当前环境里的 torch，避免隔离环境重复下载 torch
log "安装 utils3d-moge / pipeline / flex-gemm ..."
pip install -e "$CODE_DIR/utils3d-moge" --no-build-isolation
pip install -e "$CODE_DIR/pipeline" --no-build-isolation
pip install -e "$CODE_DIR/FlexGEMM" --no-build-isolation   # 默认纯 Triton 模式，无需编译 CUDA

# ---------- 5. 安装 MoGe 本体 ----------
log "安装 MoGe (editable, 无依赖)..."
pip install -e . --no-deps

# ---------- 6. 运行时依赖 ----------
# opencv 固定 4.10.0.84: 5.0 构建缺少 EXR writer，写 depth.exr 会报错
log "安装运行时依赖..."
pip install -r requirements-runtime.txt

# ---------- 7. 冒烟验证 ----------
log "验证导入 ..."
python -c "import torch; assert torch.cuda.is_available(); from moge.model.v3 import MoGeModel; import flex_gemm, utils3d_moge, pipeline; print('OK  torch', torch.__version__, '| cuda', torch.cuda.get_device_name(0))"

log "配置完成！激活: conda activate $ENV_NAME"
log "推理示例: bash infer.sh <图片> v3"
