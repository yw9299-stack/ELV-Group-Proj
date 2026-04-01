# CJEPA PushT 新电脑部署流程

这份文档记录的是这次在新电脑上把 PushT 的 CJEPA 训练链路真正跑起来的实际流程。

目标不是“只让 import 成功”，而是让 PushT 的 slot-based 训练脚本能够进入真实训练。

## 适用范围

这份流程适用于：

- Windows 主机
- WSL2 Ubuntu
- WSL 可见 NVIDIA GPU
- Windows 侧用 PyCharm，解释器指向 WSL 里的 Conda 环境
- `team9-model-code/external/cjepa-main`
- PushT 的 slot-based 训练入口：
  - `src/train/train_causalwm_AP_node_pusht_slot.py`

这份文档**不**覆盖从零开始重训 object-centric model。对于 PushT 的 CJEPA 训练，我们采用的是：

- 官方预提取的 slot embeddings
- 官方 Videosaur checkpoint

## 1. 需要准备的东西

你需要以下组件：

- 当前这个 workspace 仓库
- WSL Ubuntu
- WSL 里的 Miniconda
- 一个用于 CJEPA 的 Conda 环境
- 足够新的 Windows NVIDIA 驱动
- DINO-WM 的 PushT 数据集：
  - `team9-model-code/external/dino_wm/datasets/pusht_noise`
- 官方预计算的 PushT slots
- 官方 PushT Videosaur checkpoint
- 从 `pusht_noise` 生成出来的 action/proprio/state 三个 metadata pickle

## 2. 目录结构

相关目录最终应当大致如下：

```text
Embodied Vision group proj/
├── docs/
│   ├── cjepa_pusht_setup.md
│   └── cjepa_pusht_setup_zh.md
├── scripts/
│   ├── prepare_cjepa_pusht_meta.py
│   ├── extract_pusht_slots_videosaur.py
│   └── run_cjepa_pusht_1epoch.sh
└── team9-model-code/
    └── external/
        ├── dino_wm/
        │   └── datasets/
        │       └── pusht_noise/
        └── cjepa-main/
            └── data/
                └── pusht_precomputed/
```

最终 `pusht_precomputed` 目录应包含：

```text
team9-model-code/external/cjepa-main/data/pusht_precomputed/
├── pusht_expert_action_meta.pkl
├── pusht_expert_proprio_meta.pkl
├── pusht_expert_state_meta.pkl
├── pusht_slots.pkl
└── pusht_videosaur_model.ckpt
```

## 3. Windows 和 WSL 前置准备

### 3.1 安装 WSL2

使用 WSL2，并安装 Ubuntu。

### 3.2 在 WSL 里安装 Miniconda

为 CJEPA 单独准备 Conda 环境。

### 3.3 更新 Windows NVIDIA 驱动

这一点非常重要。即使训练是从 WSL 里启动，WSL 使用的也是 Windows 主机上的 NVIDIA 驱动。

这次部署里，推荐优先使用 **Studio Driver**，因为当前机器主要是用于 CUDA / PyTorch / 复现实验，不是游戏。

更新驱动后，重启 Windows，然后检查：

Windows 下：

```cmd
nvidia-smi
```

WSL 下：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

如果 WSL 里不能正常用 CUDA，训练后续很可能会在 checkpoint 加载或张量搬运阶段失败。

## 4. 克隆并打开仓库

例如把仓库放在 Windows：

```text
D:\Embodied Vision group proj
```

那么在 WSL 里对应路径是：

```text
/mnt/d/Embodied Vision group proj
```

## 5. 准备 CJEPA 环境

激活你希望 PyCharm 和 WSL 共用的环境，例如：

```bash
conda activate cjepa
```

环境里至少需要能正常使用：

- `torch`
- `torchvision`
- `lightning`
- `transformers`
- `timm`
- `stable_pretraining`
- `stable_worldmodel`

仓库里还 vendored 了第三方代码：

- `src/third_party/videosaur`
- `src/third_party/stable-worldmodel`
- `src/third_party/stable-pretraining`

## 6. 把 PushT 数据集放到位

这条流程要求 PushT 数据在：

```text
team9-model-code/external/dino_wm/datasets/pusht_noise
```

这个目录至少要包含：

```text
pusht_noise/
├── train/
│   ├── states.pth
│   ├── rel_actions.pth
│   ├── velocities.pth
│   ├── seq_lengths.pkl
│   └── obses/
└── val/
    ├── states.pth
    ├── rel_actions.pth
    ├── velocities.pth
    ├── seq_lengths.pkl
    └── obses/
```

这份文档默认你已经把 `pusht_noise` 放好了。

## 7. 生成 metadata pickle

我们通过 `pusht_noise` 直接生成 CJEPA 需要的三个 metadata：

- [prepare_cjepa_pusht_meta.py](/mnt/d/Embodied%20Vision%20group%20proj/scripts/prepare_cjepa_pusht_meta.py)

运行：

```bash
cd "/mnt/d/Embodied Vision group proj"
python scripts/prepare_cjepa_pusht_meta.py
```

它会生成：

- `pusht_expert_action_meta.pkl`
- `pusht_expert_proprio_meta.pkl`
- `pusht_expert_state_meta.pkl`

输出目录是：

```text
/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed
```

## 8. 下载官方 PushT slots

正常部署时，不建议自己提 slot。

对于 PushT 的 CJEPA 训练，最快最稳的方法是直接使用官方预提取好的：

- `pusht_videosaur_slots.pkl`

下载并保存为：

```bash
mkdir -p "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed" && \
wget -O "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed/pusht_slots.pkl" \
"https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_videosaur_slots.pkl"
```

## 9. 下载官方 Videosaur checkpoint

这条 PushT slot-based 训练链路还需要 object-centric checkpoint，因为模型构建时仍然会加载 Videosaur 结构和参数。

下载：

- `pusht_videosaur_model.ckpt`

保存为：

```bash
wget -O "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed/pusht_videosaur_model.ckpt" \
"https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_videosaur_model.ckpt"
```

## 10. 检查预计算文件是否齐全

运行：

```bash
ls -lh "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed"
```

理论上应看到：

- `pusht_expert_action_meta.pkl`
- `pusht_expert_proprio_meta.pkl`
- `pusht_expert_state_meta.pkl`
- `pusht_slots.pkl`
- `pusht_videosaur_model.ckpt`

## 11. 启动训练

使用已经准备好的 WSL 启动脚本：

- [run_cjepa_pusht_1epoch.sh](/mnt/d/Embodied%20Vision%20group%20proj/scripts/run_cjepa_pusht_1epoch.sh)

运行：

```bash
bash "/mnt/d/Embodied Vision group proj/scripts/run_cjepa_pusht_1epoch.sh"
```

这个脚本会：

- 检查 metadata pickle 是否存在
- 如果不存在则自动生成
- 检查 `pusht_slots.pkl`
- 检查 `pusht_videosaur_model.ckpt`
- 启动：
  - `src/train/train_causalwm_AP_node_pusht_slot.py`

## 12. PyCharm 配置

推荐做法：

- 在 Windows 侧 PyCharm 打开项目
- 解释器使用 **WSL interpreter**
- 指向 WSL 里的 Conda 环境，例如：
  - `/home/<user>/miniconda3/envs/cjepa/bin/python`

为了实现一键运行，建议：

1. 建一个 Shell Script / Bash 的 Run Configuration
2. Script path:
   - `/mnt/d/Embodied Vision group proj/scripts/run_cjepa_pusht_1epoch.sh`
3. Interpreter:
   - `/bin/bash`
4. Working directory:
   - `/mnt/d/Embodied Vision group proj`

## 13. 为什么这次需要这么多步骤

这条链路在新电脑上并不是开箱即用的，主要踩到了以下问题：

- PushT slot 训练依赖的 metadata 不能直接从 `pusht_noise` 现成使用，需要先生成
- PushT slot 训练依赖官方预提取的 `slots.pkl`
- Videosaur 模型构建时仍然需要 object-centric checkpoint
- 当前环境里的 `torchcodec` 已安装但无法正常使用
- 当前安装的 `stable_worldmodel` 已经不再暴露 `swm.wm.dinowm.Embedder`
- 初始阶段还遇到了 CUDA 驱动与 checkpoint 加载的兼容问题

因此这次真正跑通依赖了一批本地兼容性修改。

## 14. 为了部署成功做过的本地改动

以下文件是为了让这条链路在当前机器上真正可用而新增或修改的：

- `scripts/prepare_cjepa_pusht_meta.py`
  - 新增，用来从 `pusht_noise` 生成三个 metadata pickle
- `scripts/extract_pusht_slots_videosaur.py`
  - 新增，用于可选的 slot 提取
  - 支持多种视频解码回退
- `scripts/run_cjepa_pusht_1epoch.sh`
  - 新增，给 WSL / PyCharm 一键启动使用
- `src/third_party/videosaur/videosaur/data/pipelines.py`
  - 改为允许 `torchcodec` 导入失败
- `src/third_party/videosaur/videosaur/models.py`
  - checkpoint 加载时改为 `map_location="cpu"`
- `src/train/train_causalwm_AP_node_pusht_slot.py`
  - 从失效的 `swm.wm.dinowm.Embedder` 切换到仓库自带的 `Embedder`

这些改动的目标是部署兼容性，不改变整体的 PushT slot-based 训练任务定义。

## 15. 可选：自己重新提 slots

这是可选路径，不是正常部署所必需。

如果你之后确实想复现 slot 提取流程，可以使用：

- [extract_pusht_slots_videosaur.py](/mnt/d/Embodied%20Vision%20group%20proj/scripts/extract_pusht_slots_videosaur.py)

但这条路更脆弱，因为会依赖：

- 视频解码后端是否正常
- object-centric checkpoint 是否可用
- CUDA / PyTorch / TorchCodec 的版本兼容性

因此对于新电脑部署，优先使用官方预提取好的 PushT slots。

## 16. 成功判据

当日志中同时出现以下现象时，可以认为这套部署已经成功：

- metadata pickle 成功加载
- slot embeddings 成功加载
- Videosaur checkpoint 成功加载
- Lightning 进入 `Epoch 0/0`
- step 计数器开始前进，不再停留在 `0/...`

典型成功信号例如：

```text
Epoch 0/0 ... 865/119186
```

看到这种日志，就说明训练链路已经真实开始运行。
