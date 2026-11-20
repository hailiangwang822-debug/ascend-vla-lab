# v0.1.2 — Actual image version check

修复实际作业 torch-npu 2.7.1 被唯一允许值 2.7.1.post2 拦截的问题。启动、隔离环境、导入模块共用一个版本规则，两个已报告构建进入功能检查；不修改平台包。保留镜像记录、源码哈希和断点兼容检查。增加真实64rank上的FP32/BF16反向运算检查，尚待云端执行验证。

完整代码包或累计小补丁二选一；补丁支持覆盖 v0.1.0/v0.1.1 的 A。按 [v0.1.2 操作说明](VERSION_FIX_V012_CN.md) 更新代码目录。

# v0.1.1 — ModelArts node startup repair

修复首个64卡作业中的 `RTF entry has no ASCEND_DEVICE_ID`：RTF 存在不再被当成逐卡启动；统一为旧成功作业的每节点一个 bash 入口。提前验证镜像和源码，补齐旧作业运行库路径，检查双图处理器，并在加载模型前检查64进程通信和BF16运算。

下载 `ascend-vla-v0.1.1-code.zip`，上传完整 A 到新的代码目录；复用 v0.1.0 的模型、数据和离线依赖。详见 [修复操作指引](STARTUP_FIX_V011_CN.md)。真实 NPU/OBS 集成仍待云端验证。

本地验证：目标 Accelerate 1.11.0 下 25/25 项测试通过；55 个离线 wheel 与 1312 个非权重资产校验通过；目标 Transformers 4.57.1 双图处理及缩小随机 QwenOFT CPU 前反向通过。另提供约55KB覆盖补丁，合并旧代码包后与新完整代码包逐文件相同。

# v0.1.0 — Ascend / ModelArts training integration

首个公开交付版本。基于固定 StarVLA 源码，提供 QwenOFT / LIBERO Spatial 的 8×8 昇腾训练配置、离线资产校验、OBS 完整状态持久化与同拓扑恢复。

**下载 `ascend-vla-v0.1.0-transfer.zip`**：包含代码、完整Spatial数据、Python3.11/aarch64依赖、配置及下载器；两片模型权重按操作指南从官方ModelScope下载。仅需看代码时浏览仓库即可。

配套文件：`SHA256SUMS.txt`、`release-manifest.json`、`YELLOW_ZONE_RUNBOOK_CN.md`。在Windows准备文件，然后上传OBS，在ModelArts启动。

验证：8项本地CPU/故障注入测试通过，Python语法和离线ZIP CRC校验通过。真实64卡NPU训练、HCCL、OBS集成、训练后仿真成绩尚待验证。

本版将OBS根目录改为必填的 `STARVLA_OBS_ROOT`，请修改桶名/目录；启动新RUN_NAME。之前私有原型的已启动作业应继续使用原代码进行恢复，不能混用不同代码指纹。

上游模型、数据及依赖许可保持独立，详见 THIRD_PARTY_NOTICES.md。项目不包含公司桶名、工号、私有VLM训练数据或凭据。

发布历史已清理旧上游提交中的令牌字符串，保留作者、提交结构及来源映射；当前训练源码未因此改变。
