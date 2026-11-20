# 黄区 Windows 电脑实践：下载 → 贵阳 OBS → 64 卡训练

**更新：准备好本页资产后，请按 [v0.1.2 修复指引](VERSION_FIX_V012_CN.md) 使用新版代码启动，包含节点入口与实际镜像版本检查修复。资产与权重继续使用。**

本指南对应公开版 **v0.1.0**。Windows 只准备文件和操作云平台；训练在贵阳 ModelArts 8节点×8张910B上执行，Windows不需要CUDA、CANN或训练环境。已有π0.5推理环境和本地MuJoCo可以继续用于原来的推理实验。

从项目 [Releases](https://github.com/hailiangwang822-debug/ascend-vla-lab/releases) 下载 `ascend-vla-v0.1.0-transfer.zip`，不是GitHub自动生成的Source code.zip。总包包含代码、完整Spatial数据、离线依赖、配置和下载器；两片Qwen权重需按下文下载。若尚未出现Release，说明上传还未完成。

在资源管理器打开 `D:\intern`，右键“在终端中打开”，使用 **PowerShell**：

```powershell
New-Item -ItemType Directory -Force D:\intern | Out-Null
cd D:\intern
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\ascend-vla-v0.1.0-transfer.zip" -DestinationPath D:\intern -Force
cd D:\intern\ascend-vla-ready
$Manifest = Get-Content .\delivery-manifest.json -Raw | ConvertFrom-Json
foreach ($Item in $Manifest.files) {
    if ((Get-FileHash -LiteralPath $Item.filename -Algorithm SHA256).Hash.ToLower() -ne $Item.sha256) {
        throw "文件校验失败：$($Item.filename)"
    }
}
Write-Host "包内文件全部校验通过"
```

若浏览器另存到其他位置，只修改 `-LiteralPath`。下面第2节下载权重仍在这个PowerShell中运行；第4节的bash启动命令填在ModelArts训练作业页面，不能粘贴到Windows执行。

公开版改变了源码指纹。已有旧版训练作业应继续使用旧版代码恢复，不要混用两个版本的断点。第一次用本版请创建新的RUN_NAME。

---

版本：`ascend-vla-v0.1.0`。8节点×每节点8张910B 64GB，复用旧VLM作业能跑通的LlamaFactory镜像，另开新作业。训练StarVLA QwenOFT：Qwen2.5-VL-3B基座＋连续动作MLP，完整LIBERO Spatial 10任务、432示范、52,970帧，双摄像头、8×7动作块。这是全参数SFT模仿学习；后续再考虑RL，不使用之前的π0.5权重。

公开实现：https://github.com/starVLA/starVLA ，固定commit `261ba5dd4313bbb910a9caeea07433c3d7b1aa34`。

## 1. 本地文件

将 `ascend-vla-v0.1.0-transfer.zip` 总包解压到另一台有网Windows电脑，例如 `D:\intern\ascend-vla-ready`。内容如下：

```text
ascend-vla-v0.1.0-code.zip    新代码，解压后顶层A/
starvla-spatial.tar             完整Spatial数据，已备好
starvla-wheels.tar              55个离线CP311/aarch64依赖wheel，已备好
qwen25-vl-3b-configs.zip         已备好的12个上游配置/tokenizer/许可文件及Notice
download_internvl_shard1.py      requests分块续传下载器
delivery-manifest.json       包大小和SHA256
64卡上传与启动说明.md
```

本指南只对应 **ascend-vla-v0.1.0-code.zip**。旧版训练若已经启动，应使用其原始代码恢复；第一次使用公开版请建新作业、新RUN_NAME。

## 2. Windows终端：下载两片Qwen权重

在ready文件夹打开PowerShell，或先执行 `cd D:\intern\ascend-vla-ready`，然后完整复制：

```powershell
$ErrorActionPreference = "Stop"
$Workers = 32
$RepairScript = Join-Path (Get-Location) "download_internvl_shard1.py"
$ConfigZip = Join-Path (Get-Location) "qwen25-vl-3b-configs.zip"
if (-not (Test-Path -LiteralPath $RepairScript -PathType Leaf)) { throw "当前目录缺少原下载器。" }
if (-not (Test-Path -LiteralPath $ConfigZip -PathType Leaf)) { throw "当前目录缺少模型配置zip。" }
$ModelParent = Join-Path $env:USERPROFILE "Downloads\StarVLA64"
$RepairRoot = Join-Path $ModelParent "Qwen2.5-VL-3B-Instruct"
New-Item -ItemType Directory -Force -Path $ModelParent | Out-Null
Expand-Archive -LiteralPath $ConfigZip -DestinationPath $ModelParent -Force
$RepairFiles = @(
    @{
        Name = "model-00001-of-00002.safetensors"
        Size = "3982649232"
        Hash = "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8"
    },
    @{
        Name = "model-00002-of-00002.safetensors"
        Size = "3526688744"
        Hash = "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4"
    }
)
$RepairBase = "https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-VL-3B-Instruct/repo?Revision=1b5a7674d40a0ac9acef541a46fed12d710bd36b&FilePath="
foreach ($RepairItem in $RepairFiles) {
    Write-Host "开始下载并校验：$($RepairItem.Name)"
    $RepairArgs = @(
        $RepairScript
        "--url", ($RepairBase + $RepairItem.Name)
        "--output", (Join-Path $RepairRoot $RepairItem.Name)
        "--size", $RepairItem.Size
        "--sha256", $RepairItem.Hash
        "--workers", "$Workers"
        "--retries", "4"
        "--insecure"
    )
    python @RepairArgs
    if ($LASTEXITCODE -ne 0) { throw "$($RepairItem.Name) 未完成；已保留分块，重跑相同命令续传。" }
}
Write-Host "两片权重均已校验：$RepairRoot"
explorer.exe $RepairRoot
```

约7.52GB权重，Downloads盘建议至少20GB空闲供分块和合并。下载完整文件会先校验后跳过；中断后重跑可续传。`--insecure`供特定网络环境使用，完整SHA256仍必须通过。

最终模型文件夹应有15个正式文件（含新增的Notice）：两片safetensors，以及Notice、LICENSE、README.md、chat_template.json、config.json、configuration.json、generation_config.json、merges.txt、model.safetensors.index.json、preprocessor_config.json、tokenizer.json、tokenizer_config.json、vocab.json。

## 3. 贵阳OBS上传目录

把全文的 `YOUR_BUCKET` 替换成你的贵阳一桶名，`YOUR_PREFIX` 替换成你的项目目录。`STARVLA_OBS_ROOT` 是必填项；它必须与实际上传目录一致。OBS浏览器可能显示obs://，代码里MoXing用s3://，指同一个桶。你上传一份即可，八个节点自动各自下载到/cache。

| 本地内容 | 上传目标 |
|---|---|
| 新代码zip解压后的整个A文件夹 | `s3://YOUR_BUCKET/YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.0/A/` |
| starvla-spatial.tar，原样上传 | `s3://YOUR_BUCKET/YOUR_PREFIX/vla/assets/starvla-qwen25-spatial-v1/starvla-spatial.tar` |
| starvla-wheels.tar，原样上传 | `s3://YOUR_BUCKET/YOUR_PREFIX/vla/assets/starvla-qwen25-spatial-v1/starvla-wheels.tar` |
| Downloads/StarVLA64下Qwen2.5-VL-3B-Instruct目录中的15个正式文件（含新增的Notice） | `s3://YOUR_BUCKET/YOUR_PREFIX/vla/assets/starvla-qwen25-spatial-v1/models/Qwen2.5-VL-3B-Instruct/` |

**不要上传.parts目录、.download.lock、.assembling。** 配置zip内容已在模型文件夹，不用另传配置zip。两个tar不用你解压，启动器会处理。

代码解压命令：

```powershell
Expand-Archive -LiteralPath .\ascend-vla-v0.1.0-code.zip -DestinationPath .\starvla-code -Force
explorer.exe .\starvla-code
```

上传starvla-code里面的 **A**，避免多套一层。OBS里应直接存在：

```text
YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.0/A/run_cloud.sh
YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.0/A/cann83/bootstrap.py
YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.0/A/starVLA/training/train_starvla.py
```

## 4. ModelArts作业与启动命令

新建独立训练作业，选择旧VLM能跑的镜像，**8节点，每节点8卡**。沿用旧作业自定义bash方式；不要再套平台torchrun/msrun，内部Accelerate已负责启动每节点8进程。

代码目录选 **A的上一层**：

```text
obs://YOUR_BUCKET/YOUR_PREFIX/vla/training_packages/ascend-vla-v0.1.0/
```

这样下载后就是 `${MA_JOB_DIR}/A`，与你旧包相同。完整启动命令：

```bash
cd "${MA_JOB_DIR}/A" && STARVLA_OBS_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX/vla RUN_NAME=starvla_spatial_64p_001 TRAIN_STEPS=1000 SAVE_INTERVAL=100 RESUME=none bash run_cloud.sh train
```

如果页面选的代码目录本身是A，平台可能把run_cloud.sh放在MA_JOB_DIR根下，需相应调整cd。原则是cd到**包含run_cloud.sh的实际A目录**；不要凭空把/code当环境变量，也不用cd到conda目录。

若页面要求输出目录，可填：

```text
obs://YOUR_BUCKET/YOUR_PREFIX/vla/platform-output/starvla_spatial_64p_001/
```

核心断点由代码直接写OBS，**不依赖作业结束后平台才同步/cache**。每节点建议至少32CPU核、128GB内存、100GB可写/cache，实际峰值看作业。使用完整Spatial数据而非几个假样本；数据解码用CPU，64卡不保证比小规模更高效。

自动流程：激活原PyTorch-2.7.1环境→source CANN→每节点OBS写权限探测→下载并核验模型/数据/wheels→隔离venv离线安装→检查每节点8张NPU、包版本与真实双图样本→启动8×8训练。不升级系统torch、torch_npu、torchvision、Accelerate、DeepSpeed。

## 5. 保存位置与成功标志

默认1000优化步，全局batch64，每卡batch1，累积1；全参BF16、ZeRO2、gradient checkpointing、SDPA。1000步是首轮预算，不承诺论文成绩。

第1步保存完整断点，之后每100步和训练结束保存。所有rank保存state，各节点leader上传本节点分片；8节点全部上传成功才发布全局_COMPLETE和LATEST。

```text
s3://YOUR_BUCKET/YOUR_PREFIX/vla/runs/starvla_spatial_64p_001/
  _RUN.json
  LATEST.json
  checkpoints/
    step-00000001/
      node-000/ ... node-007/
      _COMPLETE.json
    step-00000100/ ...
  logs/
    node-000/<启动ID>/ ... node-007/<启动ID>/
  final_model/
    pytorch_model.pt
    config.full.yaml
    dataset_statistics.json
    _COMPLETE.json
```

日志每分钟同步，退出时再同步一次。关注：

```text
Preflight passed: ...               输入与环境检查通过
Step 1, Loss: ...                   已做真实参数更新
OBS_NODE_COMPLETE 1 <节点编号>       某节点上传完成
OBS_CHECKPOINT_COMMITTED 1 ...      八节点断点已提交
DURABLE_CHECKPOINT 1 ...            可从OBS第1步恢复
FULL_STATE_RESUMED 100 ...          从第100步恢复成功
FINAL_MODEL_UPLOADED ...            最终推理权重上传完成
```

只有DURABLE_CHECKPOINT和OBS的LATEST/_COMPLETE才代表断点已持久化；仅本地save_state完成还不够。首步有模型加载、编译及完整state上传，暂时没有下一步时要结合上传日志判断。

## 6. 中断恢复：新建同样64卡作业

保持同一代码、镜像、RUN_NAME、TRAIN_STEPS=1000、8节点×8卡。启动命令改成：

```bash
cd "${MA_JOB_DIR}/A" && STARVLA_OBS_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX/vla RUN_NAME=starvla_spatial_64p_001 TRAIN_STEPS=1000 SAVE_INTERVAL=100 RESUME=latest bash run_cloud.sh train
```

也可指定已完整提交的步骤，例如300：

```bash
cd "${MA_JOB_DIR}/A" && STARVLA_OBS_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX/vla RUN_NAME=starvla_spatial_64p_001 TRAIN_STEPS=1000 SAVE_INTERVAL=100 RESUME=300 bash run_cloud.sh train
```

恢复模型、FP32 master、Adam状态、学习率调度器、每rank Python/NumPy/PyTorch/NPU随机状态、已完成步数和数据游标。下载按SHA256校验。/cache已清空也能恢复，前提是OBS断点仍在且作业服务权限可读写桶。

只能恢复最近**成功上传**的保存点，突然中断前未保存的更新无法找回。第1步提交前失败就重新开始。不要同时运行两个同名run，不要在原作业仍训练时另开latest恢复。当前要求相同64卡拓扑，不支持把64卡优化器状态直接改成8卡。

本地只保留最近2份已提交断点；**OBS保留全部，不自动删除云端断点**。3B量级全参Adam完整断点含BF16模型、FP32 master和两份moment，粗略50–65GB/次，实际按文件核算。11次保存可能占0.5–0.8TB，建议预留约1TB OBS预算；上传也会增加耗时。确认成果后可自行制定OBS保留策略。

## 7. 当前验证和下一步

已完成源码AST、完整数据CPU reader、55个ARM wheel校验、64卡拓扑配置；故障注入验证缺节点不提交、上传失败不更新LATEST、损坏文件拒绝恢复；CPU小模型第7步恢复至20步，数据顺序、最终参数与学习率和连续训练一致。公开版新增OBS根目录配置检查后，8项本地测试全部通过，见 `evidence/public-validation.json`。

**还没有在贵阳执行这个StarVLA包，HCCL、NPU算子、实际显存和收敛需要首次作业确认。** 本地机制测试使用Python3.12、CPU torch2.9；Accelerate1.16和从官方源码加载的目标Accelerate1.11.0均通过7项测试，不冒称与远端NPU环境完全一致。已审查DeepSpeed0.18.4保存加载源码；实际镜像为Python3.11/CANN8.3.RC1/torch2.7.1/torch_npu2.7.1.post2。

训完接StarVLA LIBERO闭环评测，比较固定初态、固定预算的成功率、吞吐、显存和失败案例。π0.5的六回合成功属于独立推理诊断，不计为StarVLA训练收益。完整训练—断点恢复—仿真评测证据比“使用64卡”本身更有简历价值。

参考：

- 原始项目：https://github.com/starVLA/starVLA
- 模型：https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct
- 数据：https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
- ModelArts启动语义：https://support.huaweicloud.com/develop-modelarts/develop-modelarts-1415.html
