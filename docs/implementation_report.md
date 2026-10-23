# Phase 0–1 实现交付记录

日期：2026-10-23。原始提交：`d7d24c3`。

## 修改内容

- 新增 `zsdc/data`：GT 隔离类型、相机/单位变换、RobbySimVal 加载、碰撞聚合、固定 raw 采样。
- 新增 `zsdc/models`：延迟导入的冻结 MoGe wrapper、逐轮 parity 和冻结状态审计、RGB-only NPZ 缓存、CPU cache-only 重放。
- 新增 `zsdc/solvers`：E1/E2 鲁棒对齐、2D/depth/XYZ/normal 稀疏图、CG/直接解、L2/Huber IRLS、数值回退。
- 新增 `zsdc/evaluation` 与 `zsdc/scripts`：统一指标、流式汇总、可视化、预检/审计/缓存/补全/评估/sweep/synthetic CLI。
- 新增 CPU 测试、GPU opt-in parity 测试、服务器配置与启动脚本、CPU 依赖清单及相关文档。
- 调整包发现以包含 `zsdc`，增加可选 YAML/测试依赖；忽略实验结果及服务器数据列表。`moge/` 无改动，既有 setup/infer 脚本无改动。

## 已执行验证

```text
python -m pytest -q tests
69 passed, 1 skipped

python -m zsdc.scripts.synthetic_smoke --output results/phase1_synthetic_final
executed; explicitly synthetic analytic prior, no model/checkpoint

python -m zsdc.scripts.preflight ... --lists-only
train: 29455 samples / 16 scenes
val:    9521 samples /  5 scenes
no malformed/mismatched/duplicate rows or scene overlap

python -m zsdc.scripts.audit_model ... --audit-dir results/local_blocked_audit
v2: blocked, passed=false, No module named torch
v3: blocked, passed=false, No module named torch

git diff 74fbce0 -- moge
empty
```

GPU 测试跳过的原因是未设置真实服务器 `ZSDC_GPU_CONFIG`；不代表通过。CPU mock runner 测试确认不同方法共享采样、整张图仅调用一次模型 stub、第二次运行复用缓存，以及审计缺失时阻断；stub 不作为官方 parity 证据。

## Synthetic 数值

两个 24×32 解析场景（平面、阶跃），各从解析 sensor 采样 100 点，seed=0；先验人为施加比例和光滑局部误差。以下是合并后 **unobserved** 像素指标，单位米。

| 方法 | Pixel-pooled RMSE | Pixel-pooled MAE | 回退数 |
|---|---:|---:|---:|
| E0 analytic prior | 0.710270 | 0.651140 | 0 |
| E1 scale | 0.223140 | 0.171519 | 0 |
| E2 scale+shift | 0.202288 | 0.151203 | 0 |
| E3 RGB graph | 0.086526 | 0.063607 | 0 |
| E4 XYZ graph | 0.068906 | 0.051291 | 0 |

synthetic 报告包含配置、manifest、per-image/summary 表、热图和准确率/耗时图。图像已目视核查。该结果不能证明真实 MoGe 上的相同排序，也不构成推进 SSR 的依据。

## 尚待服务器完成

1. 核实真实 PNG 的哨兵值与 optical-axis Z/range 定义，并完成文件级预检。
2. 安装与 GPU 驱动兼容的官方运行环境，记录真实 checkpoint hash，完成 v2/v3 parity。
3. 两帧真实 smoke 后运行完整 v3 协议及固定 v2 对照；收集 observed/unobserved、边界、回退率、精度与开销。
4. 根据真实结果更新 decision_log。当前结论保持 **等待验证**。

服务器运行命令见 [server_run.md](server_run.md)。结果目录未纳入 Git；服务器部署和真实基准仍待执行。
