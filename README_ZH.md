# qwen3tts-accel

[![English README](https://img.shields.io/badge/README-English-blue)](./README.md)

## 作者为什么做这个

这个仓库的起点，其实非常个人。

我特别喜欢安和昴，*Girls Band Cry* 里的那个安和昴。做算法岗做得有点厌了之后，就很想搞点 AIGC 的东西玩玩，于是说做就做，打算拿她的音色和人格做一个桌面宠物。大概的想法就是：接一个多模态 API，把 TTS 本地部署起来，再加一点安和昴的 Live2D 或者别的表现方案，最后在电脑桌面上放一个活灵活现的昴宝宝，能陪我唠嗑，偶尔看看我屏幕上在干什么。

声音这一块一开始用的是 Qwen3-TTS 做克隆，效果其实非常好，出来的声音很像，也很符合我想要的那种昴的感觉。问题出在延迟上，延迟高到根本没法接受。对于桌宠这种东西来说，陪伴感特别依赖实时反馈，结果即使用了流式、做了分句输出，RTF 还是远大于 1，完全不够实时。一句大概 3 秒的话，能生成 7 到 8 秒，沟槽的，我的昴怎么能一句话卡壳这么久。

然后我就开始一点点排查问题，也顺便补了不少 TTS 相关的知识。后来才真正意识到，TTS 的架构和 GPT 那种单一路径生成很不一样，瓶颈主要在 `subtalker` 这部分。说来也挺丢人的，我一个在垃圾公司算法岗实习的垃圾大学生，之前居然对这种 subtalker 竖着自回归预测 token 的结构没什么概念。最开始我的想法是直接重写 vLLM 的调用逻辑，让 `main talker` 和 `subtalker` 都走 vLLM。继续往下看之后发现，`subtalker` 的长度其实是固定的，那就意味着它非常适合用 CUDA Graph 来做加速，而且实现起来比我最开始设想的方案简单得多，也没必要搞得那么大动干戈。

最后我觉得，这一部分推理优化本身就已经很值得单独拿出来了，不一定非得依附在原来的桌宠项目里。所以我把相关逻辑重新梳理了一遍，把真正可复用的部分抽出来，就成了现在这个仓库。

`qwen3tts-accel` 是一个面向 **Qwen3-TTS** 的可复用高性能推理核心仓库。

这个仓库的重点不是完整业务服务，而是把最有价值、最容易复用的优化路径单独抽出来：

- `main talker`: 基于 **vLLM** 的自定义插件路径
- `subtalker`: 基于 **CUDA Graph** 的加速解码
- `prefill`: 从 Qwen3-TTS 中抽出的预处理逻辑

仓库中附带了一个非常薄的 HTTP API，只用于：

- 验证这条推理链可以独立工作
- 给外部系统提供最小接入样例
- 方便后续二次封装成自己的服务

它不是这个项目的核心价值，也不是要覆盖完整生产业务能力的服务层。

## 仓库保留了什么

- Qwen3-TTS 输入构造与 prefill 逻辑
- vLLM `main talker` 集成
- CUDA Graph `subtalker` 加速
- codec frame 收集与解码
- 同步与流式音频输出路径

如果你要做上传接口、声音库、鉴权、存储、任务编排，这些都建议在当前推理核心外面再包一层服务。

## 最小 API

当前仓库只提供最基础的验证接口：

- `GET /health`
- `GET /meta`
- `POST /v1/audio/speech`
- `POST /v1/audio/speech/stream`

其中：

- `/v1/audio/speech` 返回完整 `audio/wav`
- `/v1/audio/speech/stream` 返回流式二进制音频块

## 安装

```bash
pip install -e ".[vllm,dev]"
```

## 环境说明

这个项目依赖面向 GPU 的运行时环境。实际部署时，下面这些组件的可用版本组合根据目标机器单独确认：

- NVIDIA 驱动版本
- CUDA 版本
- PyTorch 版本
- vLLM 版本
- `qwen_tts` 版本
- `transformers` 版本


## 启动最小 API

```bash
python -m qwen3tts_accel.api_server \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000
```

可选鉴权：

```bash
python -m qwen3tts_accel.api_server \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000 \
  --api-key your-secret
```

如果设置了 `--api-key`，客户端需要携带：

```http
Authorization: Bearer your-secret
```

## 最小请求示例

当前接口按请求传入参考音频路径与参考文本，适合作为最基础的声音克隆验证入口。

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
├── api_server.py
├── pipeline.py
├── schemas.py
├── auth.py
├── audio.py
├── preprocess/
├── subtalker/
├── vllm/
├── decode/
├── state/
└── benchmarks/
```

## 使用建议

如果你的目标是复用 Qwen3-TTS 的高性能推理能力，那么当前仓库已经足够：

- 核心推理路径已经抽离
- 最小 API 已经可用于联调和验证
- 外围服务能力可以按你的业务需求自由扩展

## License

[MIT](LICENSE)
