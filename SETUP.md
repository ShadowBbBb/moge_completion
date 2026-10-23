# MoGe 本机配置指南（moge_completion）

本仓库 = MoGe 官方源码（fork 自 [microsoft/MoGe](https://github.com/microsoft/MoGe)，历史完整）+ 本机已验证的一键配置脚本。
可用于在其他机器快速复现 MoGe-3 / MoGe-2 环境，或 clone 后二次开发再 push。

## 一、环境要求

- Linux + NVIDIA GPU（驱动版本决定 PyTorch 构建，见 FAQ）
- Anaconda / Miniconda（Python >= 3.10）
- 网络：GitHub / HuggingFace 直连不通时，脚本自动走代理（见"镜像网络"）

## 二、一键安装

```bash
cd moge_completion
bash setup.sh
```

脚本会依次完成：

1. 创建 conda 环境 `moge`（Python 3.11）
2. 安装 `torch` / `torchvision`（**cu128 构建**，见 FAQ）
3. 克隆 3 个 git 依赖到固定 commit（直连失败自动切 `ghfast.top` 代理）：
   - `utils3d-moge`（纯 Python）
   - `pipeline`（纯 Python）
   - `FlexGEMM`（**默认纯 Triton 模式，无需编译 CUDA**）
4. 依次 `pip install -e . --no-build-isolation` 安装以上依赖
5. 安装本仓库 `pip install -e . --no-deps`
6. 安装运行时依赖（`requirements.txt`，不含 gradio；opencv 固定 4.10.0.84）
7. 导入冒烟验证

环境变量可覆盖：`ENV_NAME` / `CODE_DIR` / `GH_PROXY` / `PYTHON_VERSION`

## 三、模型权重

| 版本 | HF 仓库 | 规模 | 说明 |
|---|---|---|---|
| MoGe-3 | `Ruicheng/moge-3-vitl` | 370M | **默认**，需显式 `--pretrained` |
| MoGe-3 | `Ruicheng/moge-3-vitg` | 1.25B | 更强，更慢 |
| MoGe-2 | `Ruicheng/moge-2-vitl-normal` | 331M | v2 默认（公制+法线） |
| MoGe-2 | `Ruicheng/moge-2-vitl` | 326M | 无法线输出 |
| MoGe-2 | `Ruicheng/moge-2-vitb/vits-normal` | 104M/35M | 轻量变体 |

首次运行会自动下载权重（走 HF 镜像），缓存于 `~/.cache/huggingface`，之后无需联网。

## 四、使用

```bash
conda activate moge

# MoGe-3（项目默认，SSR 3 步细化）
moge infer -i 图.jpg --version v3 --pretrained Ruicheng/moge-3-vitl \
  --refine_steps 3 -o out --maps --glb --ply --device cuda:0 --fp16

# MoGe-2（默认 vitl-normal，无需 pretrained）
moge infer -i 图.jpg --version v2 -o out --maps --glb --ply --device cuda:0 --fp16

# 或用示例脚本
bash infer.sh 图.jpg v3 ./out
bash infer.sh 图.jpg v2 ./out
```

Python API：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2, torch
from moge.model.v3 import MoGeModel   # v2 用 from moge.model.v2 import MoGeModel

model = MoGeModel.from_pretrained("Ruicheng/moge-3-vitl").to("cuda:0").eval()
img = cv2.cvtColor(cv2.imread("图.jpg"), cv2.COLOR_BGR2RGB)
t = torch.tensor(img / 255, dtype=torch.float32, device="cuda:0").permute(2, 0, 1)
with torch.inference_mode():
    out = model.infer(t, refine_steps=3)   # points/depth/normal/intrinsics/mask
```

## 五、镜像网络（本机网络关键）

- GitHub 直连不通 → 脚本先试直连，失败自动改用 `https://ghfast.top/https://github.com/<repo>` 代理克隆。
- HuggingFace 直连不通 → 设 `export HF_ENDPOINT=https://hf-mirror.com`（`infer.sh` / `setup.sh` 已内置）。
- PyTorch wheel 走 `https://download.pytorch.org/whl/cu128`（国内一般可达；也可用 `mirror.sjtu.edu.cn/pytorch-wheels/cu128` 备选）。

## 六、FAQ

**Q: 为什么 torch 必须用 cu128？**
驱动 575.x 最高支持 CUDA 12.9 → cu128（需驱动≥525）可用；cu130（需驱动≥580）装上也跑不了。
驱动 ≥580 的机器可改用 cu130：`pip install torch torchvision --index-url .../whl/cu130`。

**Q: FlexGEMM 需要编译吗？**
不需要。官方 pin 的 `all_triton` 提交默认纯 Triton 运行（只有显式设 `FLEX_GEMM_BUILD_CUDA=1` 才编译 CUDA 扩展）。规避了 nvcc/gcc 版本匹配问题。

**Q: OpenCV 为什么固定 4.10.0.84？**
OpenCV 5.0 的 headless 构建缺少 EXR writer，写 `depth.exr` 会报 "could not find a writer"。
4.10.0.84 的 EXR 需脚本头部 `OPENCV_IO_ENABLE_OPENEXR=1`（`moge` CLI 已自动设置）。

**Q: v2 / v3 切换注意什么？**
通过 `--version` 切换，不要用 v3 模型类加载 v2 权重（会触发 refiner 随机初始化警告）。

**Q: 如何同步上游 MoGe 更新？**
```bash
git remote add upstream https://github.com/microsoft/MoGe.git
git fetch upstream && git merge upstream/main
```
