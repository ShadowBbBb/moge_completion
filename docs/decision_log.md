# 阶段决策记录

日期：2026-10-23。

## 已完成的工程边界

Phase 0/1 工程实现包含独立 NumPy 数据接口、RobbySimVal 三列适配、冻结模型 wrapper、官方 parity 审计闸门、GT-free RGB 缓存、E0–E4、稀疏 CG/IRLS、统一指标与服务器入口。没有修改 `moge/`、训练权重或实现 SSR 注入。

CPU 单测和明确标注 synthetic 的平面/阶跃 smoke 用于验证实现。Mock 集成测试只检查接线、缓存复用和公平采样，不能证明 GPU parity。

## 当前决策：等待服务器验证

- **Executed**：本机 CPU 几何、数据临时 PNG、采样、指标、缓存及 mock runner 测试；analytic synthetic smoke；用户列表只读检查。
- **Skipped/blocked**：官方 v2/v3 GPU 推理与 parity、真实数据读取、checkpoint SHA256、真实精度/速度/显存。原因是本机缺少 PyTorch/FlexGEMM/真实数据及可验证权重环境。
- **尚无结论**：E4 是否优于 E3、MoGe-3 上局部修正的真实收益、v2/v3 精度与速度折中。

推进顺序：先服务器 preflight 和 parity，再两帧 smoke，最后固定协议完整评估。若 E4 未观测区域持续不如 E3，检查连接、带宽和遮挡断边；不得仅增大 sensor weight 或按 GT 逐帧选择参数。

本轮不会因 synthetic 成功推进 SSR Bridge，更不会实现或宣称 Measurement-Guided SSR。只有真实 Phase 1 结果支持进一步研究，且未来 bridge metric parity 合格，才评估后续阶段。
