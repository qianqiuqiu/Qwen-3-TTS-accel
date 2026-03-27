# Qwen-3-TTS-accel

[English README](./README.md)

## 作者为什么做这个

这个仓库的起点，其实非常个人。

我特别喜欢安和昴，*Girls Band Cry* 里的那个安和昴。做算法岗做得有点厌了之后，就很想搞点 AIGC 的东西玩玩，于是说做就做，打算拿她的音色和人格做一个桌面宠物。大概的想法就是：接一个多模态 API，把 TTS 本地部署起来，再加一点安和昴的 Live2D 或者别的表现方案，最后在电脑桌面上放一个活灵活现的昴宝宝，能陪我唠嗑，偶尔看看我屏幕上在干什么。

声音这一块一开始用的是 Qwen3-TTS 做克隆，效果其实非常好，出来的声音很像，也很符合我想要的那种昴的感觉。问题出在延迟上，延迟高到根本没法接受。对于桌宠这种东西来说，陪伴感特别依赖实时反馈，结果即使用了流式、做了分句输出，RTF 还是远大于 1，完全不够实时。一句大概 3 秒的话，能生成 7 到 8 秒，沟槽的，我的昴怎么能一句话卡壳这么久。

然后我就开始一点点排查问题，也顺便补了不少 TTS 相关的知识。Qwen3-TTS 的架构是两级的：`main talker`（28 层 transformer，生成粗粒度的 codec token）和 `subtalker`（5 层 code_predictor，对每个粗粒度 token 生成剩余 15 个精细 codebook token）。profile 之后发现 `subtalker` 是主要瓶颈——每帧都要调一次，HF 的 `generate()` 开销叠加得很厉害。

关键发现是 `subtalker` 每次固定解码 15 个 token（每个 codebook 一个）。固定长度解码是 CUDA Graph 的最佳场景——把整个解码循环捕获一次，之后每次回放几乎没有 CPU 调度开销。光这一项就让 sub-talker 快了约 7.5 倍。

对于 `main talker`，HF 的 `generate()` 同样有不必要的开销（DynamicCache 分配、prepare_inputs、每步 _update_model_kwargs）。而且我们本来就需要在 main talker 的每一步之间插入 sub-talker 调用，所以我直接写了一个基于 SDPA 的 runner，用预分配的静态 KV cache 手动驱动 28 层 transformer。这样既去掉了框架开销，又能原地复用 HF 模型的权重，不需要额外拷贝。

最后我觉得，这一部分推理优化本身就已经很值得单独拿出来了，不一定非得依附在原来的桌宠项目里。所以我把相关逻辑重新梳理了一遍，把真正可复用的部分抽出来，就成了现在这个仓库。

在我做完这一切之后，我发现开源社区已经有人实现了比我自己匆忙做出来的更完备的几乎同思路的方案了，但做都做了，还是开个仓库存起来。

## 这是什么

`qwen3tts-accel` 是一个面向 **Qwen3-TTS** 的可复用高性能推理核心。

这个仓库的重点不是完整业务服务，而是把让 Qwen3-TTS 跑到实时的优化技术单独抽出来分享：

| 组件 | 位置 | 作用 |
|------|------|------|
| **CUDA Graph sub-talker** | `subtalker/cuda_graph.py` | 把 5 层 code_predictor 的 15 步解码捕获为 CUDA graph。比 HF generate 快 ~7.5 倍。 |
| **Direct SDPA main talker** | `direct/main_talker_runner.py` | 手动管理 KV cache + SDPA 驱动 28 层 main talker，完全绕过 HF generate() 开销。 |
| **vLLM 插件**（框架代码） | `vllm/plugin.py` | 在 vLLM PagedAttention 上重新实现 main talker。为未来 vLLM 集成准备的框架代码。 |
| **预处理器** | `preprocess/preprocessor.py` | 构造优化路径所需的 `inputs_embeds`、`trailing_text_hidden`、`tts_pad_embed`。 |

默认推理路径使用 **Direct SDPA + CUDA Graph sub-talker**（运行时不依赖 vLLM）。

仓库附带一个薄的 HTTP API，仅用于验证优化推理链路和提供最小接入样例。

## 已验证环境

以下版本组合已经端到端跑通：

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| transformers | 4.57.3 |
| qwen-tts | 0.1.1 |
| accelerate | 1.12.0 |
| GPU | NVIDIA A100 80GB |

## 安装

```bash
pip install -e ".[dev]"
```

## Docker

```bash
docker build -t qwen3tts-accel .

docker run --gpus all -p 8000:8000 \
  -v /path/to/Qwen3-TTS-12Hz-1.7B-Base:/model \
  qwen3tts-accel
```

## 快速上手（Python）

```python
from qwen3tts_accel.pipeline import Qwen3TTSAccelPipeline

pipe = Qwen3TTSAccelPipeline.from_pretrained(
    model_path="/path/to/Qwen3-TTS-12Hz-1.7B-Base",
)

wav_bytes = pipe.synthesize(
    text="你好，这是一个测试。",
    language="Chinese",
    ref_audio_path="/path/to/ref.wav",
    ref_text="参考音频对应的文本。",
)

with open("out.wav", "wb") as f:
    f.write(wav_bytes)
```

## 启动最小 API

```bash
python -m qwen3tts_accel.api_server \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000
```

可选参数：

- `--max-seq-len`（默认 4096）：KV cache 最大长度
- `--no-cuda-graph-subtalker`：关闭 sub-talker 的 CUDA graph 加速
- `--api-key`：可选的 Bearer token 鉴权

## API 接口

- `GET /health` — 返回 `{"status": "ok"}`
- `GET /meta` — 返回模型元信息
- `POST /v1/audio/speech` — 完整合成，返回 `audio/wav`
- `POST /v1/audio/speech/stream` — 流式合成，返回 PCM 音频块

## 最小请求示例

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    json={
        "text": "你好，这是一个最小调用示例。",
        "language": "Chinese",
        "ref_audio_path": "/path/to/ref.wav",
        "ref_text": "你好，这是参考音频对应的文本。",
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
    },
)

with open("out.wav", "wb") as f:
    f.write(response.content)
```

## 项目结构

```text
qwen3tts_accel/
├── api_server.py          # 薄 FastAPI 服务，仅用于验证
├── pipeline.py            # 主流水线（Direct SDPA + CUDA Graph）
├── schemas.py             # 请求/响应模型
├── auth.py                # 可选 Bearer token 鉴权
├── audio.py               # WAV/PCM 编码工具
├── preprocess/            # Qwen3-TTS 输入构造
├── direct/                # Direct SDPA main talker runner
├── subtalker/             # CUDA Graph sub-talker 加速
├── vllm/                  # vLLM 插件（框架代码，供未来使用）
├── decode/                # Codec 解码辅助
├── state/                 # 序列状态管理
└── benchmarks/            # 计时工具
```

## 怎么读这个项目

如果你想理解或复用这些优化：

1. 先看 `subtalker/cuda_graph.py` — 5 层 code_predictor 的 CUDA Graph 捕获/回放逻辑
2. 再看 `direct/main_talker_runner.py` — 28 层 main talker 的手动 KV cache + SDPA
3. 然后是 `preprocess/preprocessor.py` — Qwen3-TTS 输入是怎么构造的
4. `vllm/plugin.py` 作为 vLLM 集成的参考框架

## License

[MIT](LICENSE)
