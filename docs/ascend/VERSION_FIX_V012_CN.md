# v0.1.2：修复实际训练镜像的 torch-npu 版本检查

2026-11-20 的用户日志确认：新作业已通过节点入口，实际安装元数据为 torch 2.7.1、torch-npu 2.7.1、torchvision 0.22.1、Accelerate 1.11.0、DeepSpeed 0.18.4。v0.1.1 将之前另一个运行容器报告的 torch-npu 2.7.1.post2 写成唯一值，在下载资产前主动退出。

v0.1.2 将启动元数据、隔离依赖元数据、实际导入模块三处检查合并成同一规则，允许已观察到的 2.7.1 与 2.7.1.post2 进入功能验证，支持本地构建后缀。其他 torch-npu 版本不自动放行。这不宣称两个构建的算子相同。保留实际版本记录与严格断点指纹，不修改镜像包，不重新下载依赖。

模型加载之前，各 rank 在真实 NPU 上检查 FP32/BF16 矩阵乘法及梯度，并检查64进程HCCL求和。这里写的是新增检查流程；尚未获得本版真实NPU执行结果。

## 操作

新用户可用 `ascend-vla-v0.1.2-code.zip` 完整代码包。已有 v0.1.0/v0.1.1 的 A 文件夹时，可只用累计补丁 `ascend-vla-v0.1.2-patch.zip`。补丁包含 v0.1.1 的启动修复。

例如当前本地 A 位于 `D:\intern\starvla-v011\A`，把新补丁复制到 `D:\intern`，在黄区 PowerShell 执行：

```powershell
Expand-Archive -LiteralPath "D:\intern\ascend-vla-v0.1.2-patch.zip" -DestinationPath "D:\intern\starvla-v011" -Force
```

注意 DestinationPath 是 **A 的父目录**，不要解压出 A/A。然后把更新后的整个 A 上传到 OBS 的新代码目录：

```text
obs://YOUR_BUCKET/YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.2/A/
```

训练作业选择 A 的上一层作为代码目录，继续使用原镜像、8节点×8卡、自定义bash启动：

```bash
cd "${MA_JOB_DIR}/A" && STARVLA_OBS_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX/vla RUN_NAME=starvla_spatial_64p_003 TRAIN_STEPS=1000 SAVE_INTERVAL=100 RESUME=none bash run_cloud.sh train
```

已有模型、数据、wheel 原位置复用。旧失败作业已经退出，不需要等待其诊断日志结束后重新下载资产。使用新的 RUN_NAME 避免混淆旧日志；已有完整断点继续用原版本代码恢复。

检查新日志出现 `BASE_RUNTIME_PASSED v0.1.2`，随后观察 `Preflight passed`、`HCCL_64_RANKS_PASSED`、真实 loss 与 `DURABLE_CHECKPOINT 1`。错误如发生，保留最早的 Python traceback；平台后续故障诊断不是训练程序的原始异常。
