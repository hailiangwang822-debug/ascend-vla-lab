# v0.1.3：修复 Accelerate 与 DeepSpeed 配置冲突

2026-11-20 的训练日志已通过模型和数据 SHA256 校验、每节点 8 卡识别、真实双摄像头处理、432 条轨迹 / 52,970 帧的数据读取预检。随后每个 worker 在 `DeepSpeedPlugin()` 初始化时退出，尚未执行训练参数更新。

根因是本项目生成的 Accelerate JSON 同时声明 `mixed_precision: bf16` 和 `deepspeed_config_file`。Accelerate 1.11.0 将 `mixed_precision` 放入 `ACCELERATE_CONFIG_DS_FIELDS`，插件按其配置规则拒绝。错误文本列出的是全部禁止重复配置的字段，不代表本次每个字段都冲突。

v0.1.3 从 Accelerate JSON 中移除混合精度字段，BF16 只由 DeepSpeed JSON 的 `bf16.enabled: true` 控制。不能将该字段改成字符串 `no`，它依然算显式声明。清理继承的 `ACCELERATE_*` 配置，并在启动 ranks 之前用实际安装的 Accelerate 执行 CLI 解析、worker 环境生成、DeepSpeedPlugin 初始化和精度/批量检查。这个单独的检查进程不启动分布式 workers。

## 覆盖代码

已有 v0.1.0/v0.1.1/v0.1.2 完整 A 文件夹，可以用累计补丁 `ascend-vla-v0.1.3-patch.zip` 覆盖。示例 A 位于 `D:\intern\starvla-v011\A`，补丁位于 `D:\intern`，在 PowerShell 执行一整行：

```powershell
if (!(Test-Path -LiteralPath "D:\intern\starvla-v011\A\run_cloud.sh")) { throw "请确认原来的 A 文件夹位置。" }; Expand-Archive -LiteralPath "D:\intern\ascend-vla-v0.1.3-patch.zip" -DestinationPath "D:\intern\starvla-v011" -Force
```

目标是 A 的父目录，不能解压出 A/A。上传覆盖后的整个 A 至新的代码目录；模型、数据和 wheel 仍使用原 OBS 位置。

```text
obs://YOUR_BUCKET/YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.3/A/
```

训练作业代码目录选择 `ascend-vla-v0.1.3/`（A 上一层），原镜像，8 节点 × 8 卡，自定义 bash，一行启动：

```bash
cd "${MA_JOB_DIR}/A" && STARVLA_OBS_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX/vla RUN_NAME=starvla_spatial_64p_005 TRAIN_STEPS=1000 SAVE_INTERVAL=100 RESUME=none bash run_cloud.sh train
```

本次未产生训练断点。请用未占用的 RUN_NAME；不要用 `RESUME=latest` 恢复一个尚无完整断点的失败作业。

## 观察日志

先确认 `BASE_RUNTIME_PASSED v0.1.3`，再确认新标志：

```text
ACCELERATE_DEEPSPEED_CONFIG_PASSED {"bf16": true, "zero_stage": 2, ...}
```

该记录中 `accelerate_precision: no` 表示精度交由 DeepSpeed 配置管理；`bf16: true` 才是实际插件配置的检查结果。后续仍需观察 `HCCL_64_RANKS_PASSED`、`Step 1, Loss:`、`DURABLE_CHECKPOINT 1`。

本地使用目标 Accelerate 1.11.0 原始代码验证配置及插件路径；不冒称本地 CPU 测试覆盖真实 NPU 算子、跨节点 HCCL 或 3B 模型训练。首个持久化断点和中断恢复仍需云端验收。
