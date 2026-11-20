# v0.1.1：修复 ModelArts 节点入口

v0.1.0 的首个云端作业在下载资产和模型加载之前失败：`RTF entry has no ASCEND_DEVICE_ID`。新增代码错误地把 `RANK_TABLE_FILE` 当成了平台逐卡启动的证据。已跑通的 VLM 作业使用每节点一个 bash 入口，再启动每节点 8 个训练进程。

本版恢复这一约定：不因 RTF 或设备元数据存在而要求额外变量，也不把非零变量误当作需要跳过的节点。真正嵌套 torchrun/msrun 会拒绝；同节点重复启动受文件锁保护。Accelerate standard 最终调用 torch.distributed.run，保持 8 节点 × 每节点 8 进程。

本次同时提前检查实际 Python、平台依赖、CANN 和代码哈希；清理旧 rank 元数据，保留平台分配的卡可见范围；按旧脚本设置主节点和运行库路径；检查双图处理器及动作 token；模型加载前执行 64 进程 HCCL/BF16 检查。日志上传失败时保留原始训练异常。

OBS 与训练解释器也按旧作业分开：激活 Conda 前保存平台 Python 用于 MoXing，训练仍使用 PyTorch-2.7.1 及隔离依赖。分布式等待默认改为 240 分钟，与 HCCL 的等待配置对齐，避免完整断点上传期间被旧的 30 分钟设置提前中断。ZeRO2 显式固定同拓扑恢复和 FP32 master weights 恢复配置。

## 只上传新代码

下载 Release 的 **ascend-vla-v0.1.1-code.zip**。已有模型权重、配置、`starvla-spatial.tar` 和 `starvla-wheels.tar` 继续使用。八个节点必须从同一份更新后的代码目录启动。

网络慢且已有旧版 `ascend-vla-v0.1.0-code.zip` 时，可只下载约 55KB 的 `ascend-vla-v0.1.1-patch.zip`。先将旧代码包解压到 `D:\intern\starvla-v011`，再将补丁解压到同一目录并覆盖；补丁合并后的文件已逐一校验，与新版完整代码包一致。随后仍上传合并后的整个 A。

```powershell
Expand-Archive -LiteralPath "D:\intern\ascend-vla-ready\ascend-vla-v0.1.0-code.zip" -DestinationPath "D:\intern\starvla-v011" -Force
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\ascend-vla-v0.1.1-patch.zip" -DestinationPath "D:\intern\starvla-v011" -Force
```

旧包或补丁在其他位置时修改对应 `-LiteralPath`。完整新代码包路线与补丁路线二选一即可。

黄区 Windows PowerShell 执行以下单行命令。文件在 U 盘时只修改 `-LiteralPath`：

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\ascend-vla-v0.1.1-code.zip" -DestinationPath "D:\intern\starvla-v011" -Force
```

确认 `D:\intern\starvla-v011\A\run_cloud.sh` 存在，把完整 `A` 文件夹上传到：

```text
obs://YOUR_BUCKET/YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.1/A/
```

新建训练作业版本：沿用旧成功 VLM 的镜像和自定义 bash 启动方式，8 节点、每节点 8 卡。代码目录选择 **A 的上一层**：

```text
obs://YOUR_BUCKET/YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.1/
```

启动命令（替换 OBS 根目录，整行粘贴）：

```bash
cd "${MA_JOB_DIR}/A" && STARVLA_OBS_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX/vla RUN_NAME=starvla_spatial_64p_002 TRAIN_STEPS=1000 SAVE_INTERVAL=100 RESUME=none bash run_cloud.sh train
```

无需手动设 `ASCEND_DEVICE_ID=0`。先结束旧失败作业并保留日志。首次用本版启动新 RUN_NAME；已存在的旧版本完整断点应继续用其原代码恢复，本次变更会改变代码指纹。

## 成功标志

按顺序查看：

```text
BASE_RUNTIME_PASSED v0.1.1
PROCESSOR_PASSED two_cameras=2 action_tokens=8
Preflight passed: ...
HCCL_64_RANKS_PASSED world_size=64 rank_sum=2016 BF16=passed
Step 1, Loss: ...
OBS_CHECKPOINT_COMMITTED 1 ...
DURABLE_CHECKPOINT 1 ...
```

第 1 步完整状态全部上传到 OBS 后才继续后续步数。首步保存可能较慢；先看各节点 `OBS_NODE_COMPLETE`。只有 `DURABLE_CHECKPOINT` 和 OBS 的 `_COMPLETE.json`/`LATEST.json` 才证明断点持久化完成。

恢复本版作业保持代码、镜像、节点数、RUN_NAME 和 TRAIN_STEPS 不变，只把 `RESUME=none` 改为 `RESUME=latest`。

本地回归与目标库源码核对不能代替真实 910B/HCCL/OBS 集成验证；新增运行检查会给出云端阶段证据。目前不宣称 64 卡已经跑通。
