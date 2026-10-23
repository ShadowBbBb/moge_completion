#!/usr/bin/env bash
# =============================================================
# MoGe 推理示例（v2 / v3 切换）
#   用法: bash infer.sh <图片路径> [v2|v3] [输出目录]
#   例  : bash infer.sh photo.jpg v3 ./output
# =============================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-moge}"
IMAGE="${1:?用法: infer.sh <图片路径> [v2|v3] [输出目录]}"
VERSION="${2:-v3}"
OUT="${3:-./moge_output}"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 权重未缓存时需要走镜像下载
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ "$VERSION" = "v2" ]; then
  # v2 默认 = Ruicheng/moge-2-vitl-normal (331M, 公制+法线)
  echo ">> MoGe-2 推理 ..."
  moge infer -i "$IMAGE" --version v2 -o "$OUT" --maps --glb --ply --device cuda:0 --fp16
else
  # v3 必须显式传 pretrained
  echo ">> MoGe-3 推理 ..."
  moge infer -i "$IMAGE" --version v3 --pretrained Ruicheng/moge-3-vitl \
    --refine_steps 3 -o "$OUT" --maps --glb --ply --device cuda:0 --fp16
fi

echo ">> 输出目录: $OUT"
