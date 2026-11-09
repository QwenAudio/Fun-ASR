# Hướng dẫn engine suy luận vLLM của FunASR

「[简体中文](vllm_guide_zh.md)」|「[English](vllm_guide.md)」|「Tiếng Việt」

---

## Benchmark

**Bộ kiểm thử**: 184 file, tổng cộng 11.541 giây. Mô hình: Fun-ASR-Nano / GLM-ASR-Nano.

| Mô hình | Engine | VAD | RTFx | CER | Ghi chú |
|-------|--------|-----|------|-----|-------|
| Fun-ASR-Nano | PyTorch | động | 21 | 8,06% | Đường cơ sở |
| Fun-ASR-Nano | **vLLM theo lô** | động | **340** | **8,20%** | Tăng tốc 16 lần |
| Fun-ASR-Nano | **Dịch vụ ngoại tuyến (không SPK)** | động | **102** | 8,14% | |
| Fun-ASR-Nano | **Dịch vụ ngoại tuyến (+SPK)** | động | **46** | 8,19% | SPK tắt theo mặc định |
| GLM-ASR-Nano | **vLLM theo lô** | cố định | **265** | 12,93% | Không hỗ trợ audio dài |

> vLLM cho CER khớp với PyTorch (chênh lệch < 0,2%) trong khi đạt tốc độ nhanh hơn 16–340 lần.

---

## Mục lục

1. [Cài đặt và môi trường](#1-cài-đặt-và-môi-trường)
2. [Kiến trúc engine vLLM](#2-kiến-trúc-engine-vllm)
3. [Suy luận ngoại tuyến bằng SDK](#3-suy-luận-ngoại-tuyến-bằng-sdk)
4. [Suy luận streaming bằng SDK](#4-suy-luận-streaming-bằng-sdk)
5. [Dịch vụ nhận dạng giọng nói ngoại tuyến](#5-dịch-vụ-nhận-dạng-giọng-nói-ngoại-tuyến)
6. [Dịch vụ nhận dạng giọng nói streaming](#6-dịch-vụ-nhận-dạng-giọng-nói-streaming)
7. [VAD động](#7-vad-động)
8. [Tham chiếu API](#8-tham-chiếu-api)
9. [Câu hỏi thường gặp](#9-câu-hỏi-thường-gặp)

---

<a id="1-cai-dat" name="1-cai-dat"></a>

## 1. Cài đặt và môi trường

```bash
pip install "funasr>=1.3.26"
pip install "vllm>=0.12.0"
pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart

git clone --depth 1 https://github.com/modelscope/FunASR.git
cd FunASR
pip install -e .
```

> Repository mô hình này không đóng gói kèm các script dịch vụ được dùng bên dưới. Các file chạy được `demo_vllm.py`, `serve_vllm.py`, `serve_realtime_ws.py` và các client được duy trì trong [thư mục ví dụ chính thức của FunASR](https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/fun_asr_nano). Hãy chạy các lệnh shell còn lại từ bản clone FunASR vừa tạo ở trên.

**Phần cứng**: GPU ≥ 8 GB VRAM, CUDA ≥ 11.8. Khuyến nghị 16 GB trở lên.

---

<a id="2-kien-truc" name="2-kien-truc"></a>

## 2. Kiến trúc engine vLLM

### Kiến trúc tổng thể

Phần tích hợp vLLM của FunASR tách mô hình ASR thành hai thành phần chạy độc lập:

```
┌──────────────────────────────────────────────────────────────┐
│                  FunASR + vLLM Inference Architecture          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────── PyTorch (single GPU) ───────────┐          │
│  │                                                │          │
│  │  Audio ──→ Frontend ──→ Audio Encoder ──→ Adaptor         │
│  │            (fbank)      (SenseVoice/     (Transformer/    │
│  │                          Whisper)         MLP)            │
│  │                              │                            │
│  │                              ▼                            │
│  │                     Audio Embeddings                      │
│  │                              │                            │
│  │  Text Prompt ──→ Tokenize ──→ Embed                      │
│  │  (system/user/                  │                         │
│  │   hotwords/language)            │                         │
│  │                                 ▼                         │
│  │                          [Concat Embeddings]              │
│  └─────────────────────────────────┼─────────────┘          │
│                                    │                         │
│                                    ▼ EmbedsPrompt            │
│  ┌─────────────── vLLM Engine ────────────────────┐          │
│  │                                                │          │
│  │   PagedAttention + Continuous Batching         │          │
│  │   KV Cache management + CUDA Graph             │          │
│  │   Tensor Parallel (multi-GPU)                  │          │
│  │                                                │          │
│  │   Qwen3-0.6B / Llama-2B (LLM decoding)        │          │
│  │                                                │          │
│  └────────────────────┬───────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│                Generated Text                                │
│                       │                                      │
│  ┌────────────────────┼──────────────────────────┐           │
│  │  (Optional) CTC Decoder ──→ Forced Alignment  │           │
│  │           ──→ Character-level timestamps       │           │
│  └───────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

### Vì sao dùng vLLM?

| Tính năng | PyTorch generate() | vLLM |
|---------|-------------------|------|
| Quản lý KV Cache | Cấp phát cố định, lãng phí bộ nhớ | PagedAttention, cấp phát theo nhu cầu |
| Gom lô | Phải padding thủ công | Continuous Batching, tự động lập lịch |
| Tối ưu CUDA | Không có | CUDA Graph + hợp nhất toán tử |
| Song song nhiều GPU | Tự cài đặt thủ công | Tensor Parallel, cấu hình một dòng |
| Thông lượng | RTFx ~20 | **RTFx 340+** |

### Các mô hình được hỗ trợ

| Mô hình | Thành phần LLM | Bộ mã hóa audio | Tăng tốc vLLM |
|-------|--------------|---------------|-------------|
| **Fun-ASR-Nano** | Qwen3-0.6B | SenseVoice | ✓ 21,7x |
| **GLM-ASR-Nano** | Llama-2B | Kiểu Whisper | ✓ 7,6x |
| LLMASR | Qwen/Vicuna | Whisper | ✓ |
| Paraformer | Không có LLM | — | ✗ Không tự hồi quy (non-autoregressive) |
| SenseVoice | Không có LLM | — | ✗ Encoder-decoder |

### Các chi tiết cài đặt quan trọng

1. **Tách trọng số**: Trọng số LLM được trích xuất từ `model.pt` và chuyển sang định dạng HuggingFace để vLLM nạp
2. **EmbedsPrompt**: Embedding audio và embedding văn bản được nối lại và đưa vào vLLM dưới dạng một prompt embedding duy nhất
3. **use_low_frame_rate**: Đầu ra adaptor của Fun-ASR-Nano phải được cắt về đúng số token theo một công thức (yếu tố then chốt để đảm bảo tính nhất quán)
4. **Mã hóa theo lô**: Nhiều file audio đi qua `extract_fbank` → `audio_encoder` → `audio_adaptor` trong một lượt forward duy nhất
5. **Dấu thời gian CTC**: Đầu ra của encoder được giữ lại; sau khi sinh văn bản, căn chỉnh cưỡng bức (forced alignment) cho ra dấu thời gian cấp ký tự

---

<a id="3-sdk-ngoai-tuyen" name="3-sdk-ngoai-tuyen"></a>

## 3. Suy luận ngoại tuyến bằng SDK

Phù hợp nhất cho phiên âm audio quy mô lớn và xử lý theo lô ngoại tuyến. Khả năng gom lô của vLLM phát huy ưu thế lớn nhất trong kịch bản này.

### Nguyên lý thiết kế

Suy luận ngoại tuyến bằng SDK chia pipeline ASR thành hai giai đoạn thực thi độc lập:

```
┌─────────────────────────────────────────────────────────────────────┐
│            Stage 1: Audio Encoding (PyTorch, single GPU)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Audio file list ──→ Group (batch of 8) ──→ Frontend (Fbank)        │
│       │                                          │                  │
│       │                                          ▼                  │
│       │                                 SenseVoice Encoder          │
│       │                                          │                  │
│       │                                          ▼                  │
│       │                                 Audio Adaptor               │
│       │                                 (dim transform + LFR trunc) │
│       │                                          │                  │
│       └─── Shared text prompt encoding ────┐     ▼                  │
│            (system/hotwords/language)       │  audio_embeds          │
│                     │                      │     │                  │
│                     ▼                      │     ▼                  │
│                prefix_emb ──→ [concat: prefix | audio | suffix]     │
│                                                  │                  │
│                                                  ▼                  │
│                                        EmbedsPrompt (N samples)     │
└──────────────────────────────────────────────────┼─────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│        Stage 2: LLM Decoding (vLLM, multi-GPU Tensor Parallel)      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EmbedsPrompt × N ──→ vLLM Continuous Batching                      │
│                        (PagedAttention + CUDA Graph)                 │
│                              │                                      │
│                              ▼                                      │
│                     Generated token_ids × N                          │
│                              │                                      │
│                              ▼                                      │
│                     Decode + post-processing (strip special tokens)  │
│                              │                                      │
│                              ▼                                      │
│                     (Optional) CTC Forced Alignment → char timestamps│
└─────────────────────────────────────────────────────────────────────┘
```

**Các quyết định thiết kế chính:**

1. **Tách trọng số**: Ở lần chạy đầu tiên, các trọng số có tiền tố `llm.*` được trích xuất từ `model.pt` và lưu ở định dạng HuggingFace safetensors cho vLLM (lưu đệm trong thư mục `Qwen3-0.6B-vllm/`)
2. **Nối embedding**: Prompt văn bản được mã hóa qua lớp `embed_tokens` của LLM thành embedding, sau đó nối với đầu ra của audio adaptor theo chiều chuỗi: `[prefix_emb | audio_emb | suffix_emb]`, rồi gửi tới vLLM dưới dạng `EmbedsPrompt`
3. **Cắt theo tốc độ khung thấp (Low Frame Rate)**: Đầu ra adaptor phải được cắt về đúng độ dài theo công thức: `fake_token_len = ((((fbank_len - 3 + 2) // 2 - 3 + 2) // 2) - 1) // 2 + 1`, đảm bảo nhất quán với pipeline huấn luyện PyTorch
4. **Mã hóa audio theo lô**: Nhiều file audio được gom thành lô 8 file để đi qua lượt forward encoder + adaptor, giảm chi phí khởi chạy kernel GPU
5. **Dùng chung prompt văn bản**: Khi hotword và ngôn ngữ giống nhau trong một lô, prefix_emb và suffix_emb chỉ được tính một lần
6. **Dấu thời gian CTC**: Đầu ra của encoder được giữ lại; sau khi LLM sinh văn bản, căn chỉnh cưỡng bức tạo ra dấu thời gian cấp ký tự

**Vì sao nhanh hơn PyTorch generate()?**

| Khía cạnh | PyTorch | vLLM |
|-----------|---------|------|
| KV Cache | Cấp phát trước cố định (lãng phí bộ nhớ) | PagedAttention cấp phát theo nhu cầu |
| Gom lô | Căn padding thủ công | Continuous Batching tự động lập lịch |
| CUDA | Chạy tuần tự từng mẫu | CUDA Graph + hợp nhất toán tử |
| Nhiều GPU | Tự cài đặt thủ công | Tensor Parallel cấu hình một dòng |
| Kết quả | RTFx ~20 | **RTFx 340+** (tăng tốc 16 lần) |

### Giao diện chung (khuyến nghị)

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",                    # hoặc "hf"
    tensor_parallel_size=2,      # song song nhiều GPU
    gpu_memory_utilization=0.8,
)

results = model.generate(
    ["audio1.wav", "audio2.wav"],
    language="中文",
    hotwords=["张三", "北京"],
)
for r in results:
    print(f"[{r['key']}] {r['text']}")
```

### Giao diện trực tiếp

```python
from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

engine = FunASRNanoVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=4,
)

results = engine.generate(
    inputs="wav.scp",  # hỗ trợ scp/jsonl/danh sách file
    hotwords=["开放时间"],
    language="中文",
    max_new_tokens=512,
)
```

### Dòng lệnh

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# Một file
python demo_vllm.py --input audio.wav --language 中文

# Theo lô + nhiều GPU
python demo_vllm.py --input wav.scp --tensor-parallel-size 4 --batch-size 32

# Kèm hotword + lưu kết quả
python demo_vllm.py --input audio.wav --hotwords 张三 北京 --output results.jsonl
```

---

<a id="4-sdk-streaming" name="4-sdk-streaming"></a>

## 4. Suy luận streaming bằng SDK

Xử lý audio tăng dần theo từng khối 720 ms, xuất ra kết quả nhận dạng ổn định dần. Phù hợp cho kịch bản phụ đề thời gian thực tích hợp qua SDK.

### Nguyên lý thiết kế

```
Audio stream (720 ms chunks)
    │ Cumulative re-encoding (each chunk covers all audio from the start)
    ▼
┌──────────────────────────┐
│ Stage 1: First 10 chunks │  ← No prev_text; batch generation
│ Identify stable output   │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Stage 2: Subsequent      │  ← Use stable output as prev_text
└──────────┬───────────────┘
           ▼
Each chunk: [fixed region (confirmed)] + [8-char unfixed (may change)]
```

Tóm tắt sơ đồ: mỗi khối được mã hóa lại cộng dồn từ đầu audio; 10 khối đầu tiên chạy không có `prev_text` để xác định phần đầu ra ổn định, các khối sau dùng phần ổn định đó làm `prev_text`. Mỗi kết quả gồm vùng cố định (đã xác nhận) và 8 ký tự cuối chưa cố định (có thể thay đổi).

### Cách dùng

```python
from funasr.models.fun_asr_nano.inference_vllm_streaming import FunASRNanoStreamingVLLM

engine = FunASRNanoStreamingVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    chunk_ms=720,
    rollback_chars=8,
)

for result in engine.streaming_generate("audio.wav", language="中文"):
    if result["is_final"]:
        print(f"Final: {result['text']}")
    else:
        print(f"[{result['audio_duration_ms']:.0f}ms] Confirmed: {result['fixed_text']}")
```

### Đặc điểm đầu ra

| Lượng audio tích lũy | Chất lượng đầu ra |
|-------------------|---------------|
| < 1,5 giây | Rỗng hoặc nhiễu |
| 1,5–3,0 giây | Đúng một phần |
| > 3,0 giây | Đầu ra chính xác |

> Lưu ý: `repetition_penalty=1.3` được cố định bên trong để tránh suy giảm chất lượng do lặp từ ở các khối ngắn.

---

<a id="5-dich-vu-ngoai-tuyen" name="5-dich-vu-ngoai-tuyen"></a>

## 5. Dịch vụ nhận dạng giọng nói ngoại tuyến

### 5.1 Kiến trúc dịch vụ

```
Client                                  serve_vllm.py
  │                                        │
  │── HTTP / OpenAI / WebSocket ─────────→│
  │                                        │
  │                                   ┌────┴────────────────────────┐
  │                                   │ 1. Receive complete audio    │
  │                                   │ 2. Dynamic VAD (≤60 s/seg)  │
  │                                   │ 3. vLLM batch all segments  │
  │                                   │ 4. CTC timestamps (per-char)│
  │                                   │ 5. Speaker diarization (opt)│
  │                                   └────┬────────────────────────┘
  │                                        │
  │←── JSON result ───────────────────────│
```

Các bước xử lý: (1) nhận toàn bộ audio, (2) VAD động (≤60 giây/đoạn), (3) vLLM xử lý theo lô mọi đoạn, (4) dấu thời gian CTC (theo từng ký tự), (5) phân tách người nói (tùy chọn).

**Đặc điểm**:
- Chỉ xử lý khi audio đã đến đầy đủ — lý tưởng cho phiên âm file
- VAD động giữ lại các đoạn dài (≤60 giây), giảm tổn thất do cắt ở ranh giới đoạn
- Suy luận theo lô trên tất cả các đoạn VAD giúp tối đa hóa thông lượng
- Tự động xuất dấu thời gian cấp ký tự
- Phân tách người nói tắt theo mặc định; client có thể bật

### 5.2 Khởi động dịch vụ

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py \
    --port 8899 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --gpu-memory-utilization 0.5
```

### 5.3 Giao thức 1: HTTP REST — `POST /asr`

Giao diện đầy đủ tính năng nhất, hỗ trợ phân tách người nói, dấu thời gian và hotword.

**Yêu cầu**: `multipart/form-data`

| Tham số | Kiểu | Mặc định | Mô tả |
|-----------|------|---------|-------------|
| `file` | file | bắt buộc | File audio (wav/mp3/flac) |
| `language` | string | None | Ngôn ngữ ("中文" / "English" / ...), None để tự nhận diện |
| `hotwords` | string | "" | Hotword, phân tách bằng dấu phẩy |
| `spk` | bool | false | Bật phân tách người nói |
| `timestamp` | bool | true | Xuất dấu thời gian cấp ký tự |

**Phản hồi**:

```json
{
    "text": "Full transcription text",
    "segments": [
        {
            "text": "Segment text",
            "start": 1.7,
            "end": 14.8,
            "speaker": "SPK0",
            "words": [
                {"word": "砸", "start": 2.02, "end": 2.08},
                {"word": "了", "start": 2.26, "end": 2.32}
            ]
        }
    ],
    "duration": 227.4,
    "processing_time": 3.422,
    "rtf": 0.015
}
```

**Ví dụ client**:

```bash
# cURL
curl -X POST http://localhost:8899/asr \
    -F "file=@meeting.wav" -F "language=中文" -F "spk=true"
```

```python
# Python requests
import requests
resp = requests.post("http://localhost:8899/asr",
    files={"file": open("audio.wav", "rb")},
    data={"language": "中文", "spk": "true"})
result = resp.json()
```

```javascript
// JavaScript fetch
const form = new FormData();
form.append("file", audioBlob, "audio.wav");
form.append("language", "中文");
form.append("spk", "true");
const resp = await fetch("http://localhost:8899/asr", { method: "POST", body: form });
const result = await resp.json();
```

### 5.4 Giao thức 2: Tương thích OpenAI Whisper — `POST /v1/audio/transcriptions`

Tương thích với chuẩn API OpenAI Whisper; dùng trực tiếp được với OpenAI SDK.

**Yêu cầu**: `multipart/form-data`

| Tham số | Kiểu | Mặc định | Mô tả |
|-----------|------|---------|-------------|
| `file` | file | bắt buộc | File audio |
| `model` | string | "fun-asr-nano" | Tên mô hình (trường để tương thích) |
| `language` | string | None | Ngôn ngữ |
| `response_format` | string | "json" | "json" / "text" / "verbose_json" |
| `timestamp_granularities` | string | "word" | "word" / "segment" |
| `spk` | bool | false | Phân tách người nói (mở rộng của FunASR) |

**Phản hồi** (`verbose_json`):

```json
{
    "task": "transcribe",
    "language": "zh",
    "duration": 5.17,
    "text": "我一直没有照顾孩子，但是我想要抚养权。",
    "segments": [
        {
            "id": 0, "start": 0.0, "end": 5.15,
            "text": "我一直没有照顾孩子，但是我想要抚养权。",
            "words": [{"word": "我", "start": 0.42, "end": 0.48}, ...]
        }
    ]
}
```

**Ví dụ client**:

```python
# OpenAI SDK (khuyến nghị)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8899/v1", api_key="none")
result = client.audio.transcriptions.create(
    model="fun-asr-nano",
    file=open("audio.wav", "rb"),
    response_format="verbose_json",
)
print(result.text)
```

```bash
# cURL
curl -X POST http://localhost:8899/v1/audio/transcriptions \
    -F "file=@audio.wav" -F "model=fun-asr-nano" -F "response_format=verbose_json"
```

### 5.5 Giao thức 3: WebSocket — `ws://host:port/ws`

Giao diện WebSocket của dịch vụ ngoại tuyến. Gửi toàn bộ audio rồi nhận kết quả. Việc phân cụm người nói được thực hiện tự động khi nhận STOP, và kết quả có kèm trường `spk`.

**Client → Server**:

| Thông điệp | Mô tả |
|---------|-------------|
| `"START"` | Bắt đầu phiên |
| `"LANGUAGE:中文"` | Đặt ngôn ngữ (tùy chọn) |
| `"HOTWORDS:word1,word2"` | Đặt hotword (tùy chọn) |
| `[binary]` | Dữ liệu audio PCM16 16 kHz đơn kênh (mono) |
| `"STOP"` | Kết thúc phiên; yêu cầu kết quả nhận dạng |

**Server → Client**:

```json
{"event": "started"}
{"event": "language_set", "language": "中文"}
{"sentences": [{"text":"...","start":..,"end":..}], "is_final": true, "duration_ms": 5170}
{"event": "stopped"}
```

**Ví dụ client**:

```python
import asyncio, websockets, json, numpy as np, soundfile as sf

async def offline_ws(audio_path):
    audio, sr = sf.read(audio_path)
    pcm = (audio * 32768).astype(np.int16)

    async with websockets.connect("ws://localhost:8899/ws") as ws:
        await ws.send("START")
        await ws.recv()
        await ws.send("LANGUAGE:中文")
        await ws.recv()

        # Gửi toàn bộ audio
        await ws.send(pcm.tobytes())
        await ws.send("STOP")

        # Nhận kết quả
        async for msg in ws:
            data = json.loads(msg)
            if data.get("is_final"):
                for s in data["sentences"]:
                    print(f"[{s['start']/1000:.1f}s] {s['text']}")
                break

asyncio.run(offline_ws("audio.wav"))
```

---

<a id="6-dich-vu-streaming" name="6-dich-vu-streaming"></a>

## 6. Dịch vụ nhận dạng giọng nói streaming

### 6.1 Kiến trúc dịch vụ

```
Client (microphone / audio stream)     serve_realtime_ws.py
  │                                      │
  │── WebSocket PCM16 16 kHz ──────────→│
  │   (~100 ms per frame, continuous)    │
  │                                      │
  │                                 ┌────┴─────────────────────────┐
  │                                 │ Real-time loop:               │
  │                                 │  ├─ Dynamic VAD (60 ms chunk) │
  │                                 │  ├─ Endpoint → vLLM decode    │
  │                                 │  ├─ No endpoint → partial     │
  │                                 │  └─ Streaming SPK assignment  │
  │                                 └────┬─────────────────────────┘
  │                                      │
  │←── JSON real-time push ─────────────│
```

Vòng lặp thời gian thực: VAD động (khối 60 ms); khi phát hiện điểm kết thúc câu → vLLM giải mã; chưa có điểm kết thúc → trả kết quả tạm (partial); đồng thời gán người nói theo luồng.

**Đặc điểm**:
- Audio đến theo từng khung; xử lý bắt đầu ngay lập tức
- Ngắt câu tự nhiên dựa trên điểm kết thúc của VAD
- Văn bản của đoạn đã xác nhận được khóa và không bao giờ thay đổi; văn bản tạm được cập nhật theo thời gian thực
- Gán người nói theo luồng + phân cụm lại toàn cục khi nhận STOP
- Độ trễ từ đầu tiên ~480 ms

### 6.2 Khởi động dịch vụ

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 --hotword-file hotword_list \
    --ws-ping-interval 30 --ws-ping-timeout 120
```

Với các cuộc họp dài hoặc kết nối đi qua Nginx/bộ cân bằng tải, hãy điều chỉnh các thiết lập keepalive của WebSocket để luôn thấp hơn thời gian chờ nhàn rỗi (idle timeout) mà gateway áp dụng:

| Tùy chọn | Mặc định | Mô tả |
|--------|---------|-------------|
| `--ws-ping-interval` | `30.0` | Số giây giữa các lần ping theo giao thức WebSocket; đặt `0` để tắt ping |
| `--ws-ping-timeout` | `120.0` | Số giây chờ pong trước khi đóng kết nối; đặt `0` để tắt thời gian chờ ping |
| `--ws-close-timeout` | `10.0` | Số giây chờ bắt tay đóng kết nối WebSocket |
| `--ws-max-size` | `10485760` | Kích thước tối đa (byte) của một thông điệp WebSocket gửi đến |

### 6.3 Giao thức WebSocket

**Kết nối**: `ws://host:10095`

**Client → Server**:

| Thông điệp | Định dạng | Mô tả |
|---------|--------|-------------|
| Bắt đầu | `"START"` | Khởi tạo phiên |
| Hotword | `"HOTWORDS:word1,word2"` | Tùy chọn |
| Ngôn ngữ | `"LANGUAGE:中文"` | Tùy chọn |
| Audio | `binary` | PCM16 16 kHz đơn kênh |
| Kết thúc | `"STOP"` | Giải mã cuối + phân cụm lại người nói |

**Server → Client**:

```json
{"event": "started"}
{"sentences": [{"text":"你好","start":300,"end":1200,"spk":0}], "partial": "世界", "is_final": false}
{"sentences": [...], "is_final": true}
{"event": "stopped"}
```

**Các trường**: `sentences[]` = các đoạn đã khóa, `partial` = văn bản đang được nói (có thể thay đổi), `is_final` = true sau khi STOP.

**Sơ đồ tuần tự**:
```
Client              Server
  │── START ───────→│
  │←─ started ──────│
  │── [audio] ─────→│
  │←─ {partial} ────│
  │── [audio] ─────→│
  │←─ {sentences+partial} ─│  (VAD cut a sentence)
  │── STOP ────────→│
  │←─ {is_final:true} ────│
  │←─ stopped ─────│
```

### 6.4 Sử dụng client

**Python CLI**:
```bash
python client_python.py --server ws://localhost:10095 --mic
python client_python.py --server ws://localhost:10095 --file audio.wav
```

**Trình duyệt**: Mở `client_mic.html`

**Python tự viết**:
```python
import asyncio, websockets, numpy as np, json

async def stream(audio_path):
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    pcm = (audio * 32768).astype(np.int16)

    async with websockets.connect("ws://localhost:10095") as ws:
        await ws.send("START")
        await ws.recv()

        for i in range(0, len(pcm), 1600):
            await ws.send(pcm[i:i+1600].tobytes())
            await asyncio.sleep(0.05)

        await ws.send("STOP")
        async for msg in ws:
            data = json.loads(msg)
            if data.get("is_final"):
                for s in data["sentences"]:
                    print(f"[{s['start']/1000:.1f}s] {s['text']}")
                break

asyncio.run(stream("audio.wav"))
```

---

<a id="7-vad-dong" name="7-vad-dong"></a>

## 7. VAD động

fsmn-vad bật ngưỡng im lặng động theo mặc định. Chế độ ngoại tuyến và streaming dùng cấu hình khác nhau.

| Thời lượng tích lũy | Ngoại tuyến (giữ đoạn dài ≤60 giây) | Streaming (cân bằng độ trễ) |
|---------------------|-----------------------------------|-----------------------------|
| ≤ 5 giây | 2000 ms | 2000 ms |
| 5–10 giây | 2000 ms | 1500 ms |
| 10–15 giây | 1000 ms | 1000 ms |
| 15–20 giây | 1000 ms | 800 ms |
| 20–30 giây | 800 ms | 800 ms |
| 30–45 giây | 600 ms | 400 ms |
| 45–60 giây | 200–400 ms | 100 ms |
| > 60 giây | 100 ms | 100 ms |

Chế độ ngoại tuyến ưu tiên các đoạn dài hơn để giảm tổn thất do cắt ở ranh giới; chế độ streaming siết ngưỡng nhanh hơn để giảm độ trễ.

### Tùy chỉnh

```python
model.generate(input="audio.wav", silence_schedule=[(5000,1500), (20000,800), (float('inf'),300)])
```

> GLM-ASR không hỗ trợ suy luận đoạn dài; hãy truyền `dynamic_silence=False` khi dùng mô hình này.

---

<a id="8-tham-chieu-api" name="8-tham-chieu-api"></a>

## 8. Tham chiếu API

| Tham số | AutoModelVLLM | serve_vllm.py | serve_realtime_ws.py |
|-----------|--------------|---------------|---------------------|
| model | ✓ | --model | --model |
| gpu_memory_utilization | ✓ | --gpu-memory-utilization | --gpu-memory-utilization |
| tensor_parallel_size | ✓ | — | --tensor-parallel-size |
| max_model_len | ✓ | --max-model-len | --max-model-len |
| language | tham số của generate() | tham số API | --language / LANGUAGE: |
| hotwords | tham số của generate() | tham số API | --hotword-file / HOTWORDS: |

---

<a id="9-faq" name="9-faq"></a>

## 9. Câu hỏi thường gặp

**H: Nên dùng ngoại tuyến hay streaming?**
File hoàn chỉnh → ngoại tuyến (thông lượng cao). Micro / luồng trực tiếp → streaming (độ trễ thấp).

**H: GLM-ASR có dùng được VAD động không?**
Mô hình này không hỗ trợ suy luận đoạn dài. Hãy dùng `dynamic_silence=False`.

**H: SPK ảnh hưởng hiệu năng thế nào?**
RTFx giảm từ 102 xuống 46. CER không đổi. Tắt theo mặc định.

**H: Tôi nhận đầu ra rỗng trên GPU đời cũ (V100, P100)?**
Đường vLLM được thiết kế cho GPU NVIDIA Ampere trở lên (compute capability >= 8.0). `bf16` yêu cầu Ampere trở lên, còn `fp16` trên GPU trước Ampere có thể cho đầu ra kém chất lượng hoặc rỗng. Với V100, P100 và các phần cứng trước Ampere khác, hãy dùng đường AutoModel (PyTorch) với `demo1.py` / `demo2.py`.

**H: Điểm vào cho phát triển tùy biến?**
Ngoại tuyến: `serve_vllm.process_audio()` / `FunASRNanoVLLM.generate()`
Streaming: `serve_realtime_ws.RealtimeASRSession`

**H: Khởi động lần đầu chậm?**
Khởi tạo vLLM mất 60–90 giây (khởi động KV Cache + CUDA Graph). Các lần suy luận tiếp theo diễn ra ngay lập tức.
