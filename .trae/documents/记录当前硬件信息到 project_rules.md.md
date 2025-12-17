## 目标
- 采集并汇总当前机器的硬件与系统关键信息（OS/CPU/内存/GPU/CUDA/驱动/Python/PyTorch/磁盘）。
- 将摘要以 Markdown 形式追加到 `/.trae/rules/project_rules.md`（Linux 路径：`/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/.trae/rules/project_rules.md`）。

## 将采集的信息
- OS 与内核：`uname -a`、`/etc/os-release`（发行版名与版本）。
- CPU：型号、架构、核心与线程数、插槽数、厂商（`lscpu` 关键字段）。
- 内存：总量与交换分区（`free -h`、`/proc/meminfo`）。
- GPU：型号、显存、驱动版本、CUDA 版本（`nvidia-smi`）。
- CUDA 工具链：`nvcc --version`（若可用）。
- Python/PyTorch：`python` 查询 `torch.__version__`、`torch.version.cuda`、`torch.backends.cudnn.version()`、`compute capability`、`python` 版本。
- 磁盘：块设备与挂载点（`lsblk`）、工作盘使用情况（`df -h /share`）。
- 时间戳：`date -Iseconds`（ISO 8601）。

## 实现步骤
1. 在当前工作目录下执行读取命令，收集上述信息；针对可能缺失的工具（如 `nvcc`），自动跳过并标注“未安装/不可用”。
2. 将原始输出进行轻度清洗与提取关键字段，生成简洁的一行摘要项。
3. 以追加方式写入 `/.trae/rules/project_rules.md`，新增章节名为：`## 硬件信息（环境指纹） — <timestamp>`。
4. 保留原文件其他内容不变；如文件为空则创建基本标题后再追加。

## 采集命令（示例）
- OS：
  - `uname -a`
  - `cat /etc/os-release`
- CPU：
  - `lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread|Core|Socket|Vendor ID"`
- 内存：
  - `free -h`
  - `grep -E "MemTotal|SwapTotal" /proc/meminfo`
- GPU/CUDA/驱动：
  - `nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv,noheader`
  - `nvidia-smi -L`
  - `nvcc --version`（若可用）
- Python/PyTorch：
  - `python -c "import torch, platform; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cudnn', torch.backends.cudnn.version()); print('compute_cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); print('python', platform.python_version())"`
- 磁盘：
  - `lsblk -o NAME,TYPE,SIZE,MOUNTPOINT`
  - `df -h /share`
- 时间戳：
  - `date -Iseconds`

## 输出格式（写入示例）
```
## 硬件信息（环境指纹） — 2025-11-29T12:34:56+00:00
- OS: Ubuntu 22.04, kernel 6.8.0-xx-generic
- CPU: Intel(R) Xeon(R) Gold 6226R, 32 cores / 64 threads, x86_64
- Memory: 256 GB RAM, Swap 64 GB
- GPU: 4× NVIDIA A100 40GB, driver 550.54.15, CUDA 12.4
- CUDA toolkit: 12.4 (nvcc)
- Python: 3.10.12
- PyTorch: 2.1.2 (CUDA 12.1), cuDNN 9.1, compute cap 8.0
- Disk: /share 10.0T total, 6.3T used, 3.7T free
- Workspace: /share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full
```

## 后续维护
- 每次升级驱动/CUDA/重大库版本时追加一条快照；保留历史记录。
- 若需要严格遵循项目“环境指纹”规范，可同时导出 `runs/<exp>/env_fingerprint.json`（含 `pip freeze` 与 CUDA/Driver/Torch 指纹），本次按需求仅在 `project_rules.md` 记录硬件摘要。