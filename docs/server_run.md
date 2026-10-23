# ZSDC Phase 0–1 服务器运行说明

## 准备

使用 Linux + NVIDIA GPU 上已兼容的 MoGe 环境。原 `setup.sh` 保留；其默认 cu128 是既有部署选择，服务器应根据实际驱动选择 PyTorch 构建。记录实际依赖版本，不把本机 CPU 测试当作 GPU 验证。

```bash
conda activate moge
pip install -r requirements-zsdc-cpu.txt
pip install -e . --no-deps
```

将两份列表放入 `assets/robby_simval_{train,val}.txt`，或者用命令行传入位置。列表路径是相对于 DATA_ROOT 的，例如 `lingbot-depth_pub/RobbySimVal/val_view/<scene>/<camera>/...`。因此 DATA_ROOT 应是 `lingbot-depth_pub` 的父目录，而不是 RobbySimVal 目录。数据和列表不会自动下载或从邻近项目复制。

复制 `configs/zsdc_base.yaml` 到服务器本地配置，填写 root、列表和 checkpoint。checkpoint 可为实际 `model.pt` 文件或明确的 HF repo ID；离线时用本地权重，或在模型项设置 `offline: true` 使用已缓存的 HF 权重。不要把 v2 checkpoint 配给 v3。

```bash
export ZSDC_CONFIG=/path/to/server_zsdc.yaml
```

所有参数均可由 `--set dotted.key=YAML_VALUE` 覆盖。模型列表使用 YAML 整项配置；`--model v3 --checkpoint /path/model.pt` 可覆盖单个模型。无需改源码。

## 1. 只读预检

```bash
bash launch_zsdc.sh preflight --lists-only
bash launch_zsdc.sh preflight --limit 8
```

第一条检查列表格式、配对及场景隔离；第二条检查两套 split 各最多 8 张的真实 PNG、原尺寸、哨兵值、内参、碰撞和 GT mask。完整预检去掉 `--limit`。也可以给其他入口加 `--dry-run`，不会加载模型或创建输出。

先核实 raw/GT 的 `65535` 含义和 Z/range 定义。默认 `[0,65535]` 为缺测、默认 Z；如数据协议不同，显式修改 `data.raw_invalid`、`data.gt_invalid`、`data.depth_kind`、`data.gt_depth_kind`。单位固定为 uint16 毫米，不按图猜测单位。缺内参时允许估计 FOV；range 转换则必须有 K。

## 2. 官方一致性审计

```bash
bash launch_zsdc.sh audit_model --limit 8
# 单独审计 v3：
bash launch_zsdc.sh audit_model --model v3 --limit 8
```

查看 `results/model_audit/<version>/audit.json` 和 `baseline_audit.md`。仅 passed 才能继续。缺 PyTorch、checkpoint、CUDA 或 refiner backend 将记录 blocked/failed 并返回非零。不要修改 JSON 强行解锁。更改源码、权重、精度、token/FOV/SSR 配置或推理环境后需要重做审计。

审计只读 RGB/raw/K，不使用 GT。可通过 `--split train` 在固定 train 样本做纯推理审计；闸门针对相同模型/配置和覆盖的 FOV 模式，不绑定评价帧。

## 3. 两帧 smoke 与缓存

```bash
bash launch_zsdc.sh evaluate --limit 2 --output results/server_smoke
bash launch_zsdc.sh cache_baseline --model v3 --output results/cache_v3_job
bash launch_zsdc.sh cache_baseline --model v2 --output results/cache_v2_job
```

确认无失败、查看 observed/unobserved 和边界指标、CG 收敛/回退及实际点数。缓存只有 RGB-only 预测和模型元信息，不含 GT、传感器或大中间特征；相同图片的 31 套采样不重复运行 ViT。

## 4. 完整评估与消融

```bash
bash launch_zsdc.sh evaluate --output results/phase1_full
bash launch_zsdc.sh sweep --model v3 --output results/graph_ablation
```

v3 默认 9,521 validation 图，每图 155 个方法条件；CPU 图求解可能耗时较长，先检查两帧 smoke。v2 默认只做 8 图、native/random-100 对照。完整 v2 矩阵请在配置中将 v2 `full_matrix` 设 true。

可在只有 CPU 求解依赖的机器上复制同版本源码、审计目录和缓存再评估：

```bash
bash launch_zsdc.sh evaluate --model v3 --cache-only --output results/phase1_cpu
```

该模式不导入 PyTorch；仍要求源码和模型推理设置与通过的审计一致，且所有所需缓存存在。审计目录记录原推理环境，manifest 单独记录当前 CPU 环境。

分辨率可控：

```bash
bash launch_zsdc.sh evaluate --model v3 --max-pixels 90000 --limit 8 --output results/lowres
bash launch_zsdc.sh evaluate --model v3 --size 240 320 --limit 8 --output results/fixedsize
bash launch_zsdc.sh evaluate --model v3 --original-resolution --limit 2 --output results/native_resolution
```

分辨率改变会生成独立 RGB cache key；尺寸本身不更改模型推理设置，但新的数值不稳定会作为数据/prior failure 报告。先 profile 再扩大到原分辨率。输出目录必须为空，使用不同目录保留各次实验。

单纯导出补全、不读取 GT：

```bash
bash launch_zsdc.sh run_completion --model v3 --limit 2 --output results/completed_maps
```

输出以条件指纹命名的 NPZ，含 metric depth 与样本/条件元信息。

## CPU 验证与交付检查

```bash
python -m pytest -q tests
bash launch_zsdc.sh synthetic_smoke --output results/phase1_synthetic
# 仅在真实服务器配置可用时启动 GPU 测试：
ZSDC_GPU_CONFIG="$ZSDC_CONFIG" python -m pytest -q tests/test_moge_parity.py
```

synthetic 报告不代表 MoGe 实测。服务器交付应保存完整运行配置、manifest、审计、summary、failures 及少量可视化；根据未观测区域收益与延迟决定下一阶段，不只看测量处误差。
