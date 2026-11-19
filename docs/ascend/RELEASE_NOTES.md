# v0.1.0 — Ascend / ModelArts training integration

首个公开交付版本。基于固定 StarVLA 源码，提供 QwenOFT / LIBERO Spatial 的 8×8 昇腾训练配置、离线资产校验、OBS 完整状态持久化与同拓扑恢复。

**下载 `ascend-vla-v0.1.0-transfer.zip`**：包含代码、完整Spatial数据、Python3.11/aarch64依赖、配置及下载器；两片模型权重按操作指南从官方ModelScope下载。仅需看代码时浏览仓库即可。

配套文件：`SHA256SUMS.txt`、`release-manifest.json`、`YELLOW_ZONE_RUNBOOK_CN.md`。在Windows准备文件，然后上传OBS，在ModelArts启动。

验证：8项本地CPU/故障注入测试通过，Python语法和离线ZIP CRC校验通过。真实64卡NPU训练、HCCL、OBS集成、训练后仿真成绩尚待验证。

本版将OBS根目录改为必填的 `STARVLA_OBS_ROOT`，请修改桶名/目录；启动新RUN_NAME。之前私有原型的已启动作业应继续使用原代码进行恢复，不能混用不同代码指纹。

上游模型、数据及依赖许可保持独立，详见 THIRD_PARTY_NOTICES.md。项目不包含公司桶名、工号、私有VLM训练数据或凭据。

发布历史已清理旧上游提交中的令牌字符串，保留作者、提交结构及来源映射；当前训练源码未因此改变。
