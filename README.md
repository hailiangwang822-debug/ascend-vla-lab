# Ascend VLA Lab

**在昇腾 910B 上复现 VLA 训练，把离线部署、完整断点恢复和仿真评估做成可复查的工程项目。**

维护者：[Wang Hailiang / hailiangwang822-debug](https://github.com/hailiangwang822-debug)。这是基于 [StarVLA](https://github.com/starVLA/starVLA) 的复现与工程适配项目，保留上游提交历史、源码许可和引用。当前选择 QwenOFT：Qwen2.5-VL-3B-Instruct 视觉语言基座，加连续动作 MLP，使用 LIBERO Spatial 示范进行全参数监督训练。

**当前状态：本地 CPU 数据读取与恢复测试通过；真实 64 卡 NPU 训练、HCCL 通信和训练后闭环成功率尚待验证。** 请按实际结果使用本项目，不能把配置为 64 卡写成已完成 64 卡训练。

## 开始实践

- **v0.1.0 首次云端启动发现节点入口错误，已有资产用户先更新 [v0.1.1 小代码包](docs/ascend/STARTUP_FIX_V011_CN.md)。模型与数据不用重传。**
- [黄区 Windows → 贵阳 OBS → ModelArts 64 卡操作指南](docs/ascend/YELLOW_ZONE_RUNBOOK_CN.md)
- [模型原理、训练基础和学习路线](docs/ascend/PROJECT_GUIDE_CN.md)
- [下载离线交付包](https://github.com/hailiangwang822-debug/ascend-vla-lab/releases)
- [验证记录与实验边界](evidence/README.md)
- [上游来源与第三方许可](THIRD_PARTY_NOTICES.md)

Release 的 `ascend-vla-v0.1.0-transfer.zip` 包含适配代码、完整 Spatial 数据、ARM/Python3.11 离线依赖、模型配置和分块续传下载器。Qwen 两片权重从官方 ModelScope 下载，不随仓库分发。GitHub 的 Source code.zip 只包含源码。

## 本项目增加的工程能力

1. 面向 CANN 8.3.RC1 / PyTorch 2.7.1 / torch_npu 2.7.1.post2 的独立启动和检查流程。
2. 在无法访问外网的 ModelArts 训练容器中，从 OBS 获取固定版本资产，校验 SHA256，创建隔离的依赖环境。
3. 8 节点 × 8 卡、BF16、DeepSpeed ZeRO2、梯度检查点、全参数 SFT 配置。
4. 每个节点上传其完整状态分片，所有节点完成后才发布 `_COMPLETE.json` 和 `LATEST.json`。
5. 同拓扑恢复模型、FP32 master weights、优化器、学习率调度器、随机状态、训练步数和数据游标；恢复前验证文件哈希和配置指纹。
6. 故障注入测试：缺失节点、失败上传、文件损坏不会被当作可用断点。

训练骨干、QwenOFT 模型与数据接口来源于 StarVLA；本项目新增 `cann83/`、公开配置、离线交付工具、测试和实践文档。代码与文档包含 AI 辅助开发，最终以测试和云端运行证据为准。

## 当前实验

| 项目 | 设置 |
|---|---|
| 原始代码 | StarVLA `261ba5dd4313bbb910a9caeea07433c3d7b1aa34` |
| 基座 | Qwen2.5-VL-3B-Instruct，动作头新初始化 |
| 数据 | LIBERO Spatial，10 任务、432 示范、52,970 帧 |
| 输入 / 输出 | 双摄像头图像与语言 / 未来 8 步 × 7 维连续动作 |
| 训练 | 全参数监督模仿学习，L1 动作损失 |
| 默认预算 | 1000 优化步，全局 batch 64；每卡 1，累积 1 |
| 保存 | 第 1 步、每 100 步、训练结束；OBS 持久化 |
| 硬件 | 8 节点 × 8 张 Ascend 910B 64GB，待首次实测 |

1000 步是首轮工程实验预算，不等同于上游论文的完整训练协议。64 卡也不是运行 3B 模型的最低要求；需要通过实际吞吐、通信和数据解码耗时评估扩展效率。当前从 VLM 基座学习动作能力，不是加载已有机器人 VLA 权重继续做后训练，也未实现强化学习。

## 目录

```text
cann83/                昇腾启动、配置、预检、OBS 持久化和恢复测试
starVLA/               保留并适配的上游框架
tools/                 requests 分块下载器、Release 打包工具
docs/ascend/           Windows/ModelArts 实践指南与模型原理
evidence/              本地验证与独立 π0.5 实验的说明
adaptation-manifest.json  上游版本与已改源码指纹
run_cloud.sh           ModelArts 每节点入口
```

训练页面选择代码目录 `A/` 的上一层，使用自定义 bash 启动，每节点入口内部会启动 8 个进程：

```bash
cd "${MA_JOB_DIR}/A" && STARVLA_OBS_ROOT=s3://YOUR_BUCKET/YOUR_PREFIX/vla RUN_NAME=starvla_spatial_64p_001 TRAIN_STEPS=1000 SAVE_INTERVAL=100 RESUME=none bash run_cloud.sh train
```

修改桶名和目录后使用。中断时保持相同代码、训练配置、节点数和 RUN_NAME，将 `RESUME=none` 改为 `RESUME=latest`。完整上传目录和成功标志见操作指南。

## 下一阶段验收

- 真实 64 卡完成参数更新，并记录每步耗时、峰值显存和吞吐。
- 首个 OBS 完整断点成功提交，在新作业中清空本地缓存后恢复。
- 完成首轮训练，保存可加载的推理权重与数据归一化统计。
- 接通 StarVLA 与 LIBERO 闭环评测，固定任务、初态、种子和步数预算。
- 用失败类型与对照实验分析动作执行长度、数据和训练设置的影响。

之前 π0.5 推理的 24/27 及定向复测 6/6 不属于 StarVLA 训练成绩，证据级别单独说明。项目的结果应持续补充到 `evidence/`。

## 来源与许可

StarVLA 上游说明见 [原 README](docs/ascend/UPSTREAM_README.md)，原许可保留在 [LICENSE](LICENSE)，原引用保留在 [CITATION.cff](CITATION.cff)。本项目新增工程文件适用 [LICENSE-ADAPTATIONS](LICENSE-ADAPTATIONS)。Qwen3B 模型适用 Qwen RESEARCH LICENSE AGREEMENT，不应按整个仓库的代码许可理解。数据、依赖及模型各自的来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
