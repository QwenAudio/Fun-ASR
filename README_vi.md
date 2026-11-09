# Fun-ASR

「[简体中文](README_zh.md)」|「[English](README.md)」|「[日本語](README_ja.md)」|「[한국어](README_ko.md)」|「Tiếng Việt」

Fun-ASR là họ mô hình nhận dạng giọng nói đầu-cuối (end-to-end) do Tongyi Lab phát triển. Mỗi checkpoint có phạm vi khả năng riêng: Fun-ASR-Nano-2512 được huấn luyện trên hàng chục triệu giờ dữ liệu giọng nói và hỗ trợ tiếng Trung, tiếng Anh, tiếng Nhật cùng các phương ngữ và giọng địa phương của tiếng Trung; Fun-ASR-MLT-Nano-2512 là checkpoint đa ngôn ngữ 800M tham số, được huấn luyện trên hàng trăm nghìn giờ dữ liệu và hỗ trợ 31 ngôn ngữ. Cả hai checkpoint đều tích hợp với FunASR để suy luận (inference) và triển khai dịch vụ.

<div align="center">
<img src="images/funasr-v2.png">
</div>

<div align="center">
<h4>
<a href="https://www.funasr.com/en/"> Trang chủ </a>
｜<a href="#tinh-nang-chinh"> Tính năng chính </a>
｜<a href="#danh-gia-hieu-nang"> Đánh giá hiệu năng </a>
｜<a href="#cai-dat-moi-truong"> Cài đặt môi trường </a>
｜<a href="#huong-dan-su-dung"> Hướng dẫn sử dụng </a>

</h4>

Kho mô hình: **Fun-ASR-Nano** ([ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512), [HF / Transformers](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf) · [HF / FunASR](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)) · **Fun-ASR-MLT-Nano** ([ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512), [Hugging Face](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512))

Dùng thử trực tuyến:
[ModelScope Community Space](https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano), [Hugging Face Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenAudio/Fun-ASR/blob/main/examples/colab/fun_asr_nano_transformers.ipynb)

[Các ví dụ chạy được](examples/README.md) bao gồm suy luận nhanh (quickstart), suy luận trực tiếp, phân tách người nói (speaker diarization), suy luận theo lô với vLLM và SDK streaming.

</div>

|                                                                           Tên mô hình                                                                           |                                                                                                                                                                                                       Chi tiết tác vụ                                                                                                                                                                                                       |   Dữ liệu huấn luyện    | Số tham số |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------: | :--------: |
|       Fun-ASR-Nano <br> ([⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [HF / Transformers](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf) · [HF / FunASR](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512))       | Nhận dạng giọng nói hỗ trợ tiếng Trung, tiếng Anh và tiếng Nhật. Tiếng Trung bao gồm 7 phương ngữ (Ngô, Quảng Đông, Mân, Khách Gia, Cám, Tương, Tấn) và 26 giọng địa phương (Hà Nam, Sơn Tây, Hồ Bắc, Tứ Xuyên, Trùng Khánh, Vân Nam, Quý Châu, Quảng Đông, Quảng Tây và hơn 20 vùng khác). Tiếng Anh và tiếng Nhật bao phủ nhiều giọng vùng miền. Các tính năng bổ sung gồm nhận dạng lời bài hát và nhận dạng giọng rap. |   Hàng chục triệu giờ   |    800M    |
| Fun-ASR-MLT-Nano <br> ([⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)) | Nhận dạng giọng nói hỗ trợ tiếng Trung, tiếng Anh, tiếng Quảng Đông, tiếng Nhật, tiếng Hàn, tiếng Việt, tiếng Indonesia, tiếng Thái, tiếng Mã Lai, tiếng Philippines, tiếng Ả Rập, tiếng Hindi, tiếng Bulgaria, tiếng Croatia, tiếng Séc, tiếng Đan Mạch, tiếng Hà Lan, tiếng Estonia, tiếng Phần Lan, tiếng Hy Lạp, tiếng Hungary, tiếng Ireland, tiếng Latvia, tiếng Litva, tiếng Malta, tiếng Ba Lan, tiếng Bồ Đào Nha, tiếng Romania, tiếng Slovakia, tiếng Slovenia và tiếng Thụy Điển: tổng cộng 31 ngôn ngữ. | Hàng trăm nghìn giờ |    800M    |

<a id="What's News" name="What's News"></a>

# Có gì mới 🔥

- **FunASR 1.4.15** là bản phát hành Python hiện tại dành cho cài đặt từ mã nguồn, khám phá MOSS và triển khai thời gian thực hoặc công nghiệp. Cài đặt bằng `python -m pip install -U "funasr==1.4.15"`. [Bản phát hành ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.15)
- **MOSS-Transcribe-Diarize** là mô hình bên thứ ba của OpenMOSS dành cho phiên âm ngoại tuyến audio dài, dấu thời gian (timestamp) và nhãn người nói ẩn danh, có sẵn các đường triển khai qua dịch vụ FunASR, Docker, Kubernetes, vLLM, SGLang, LocalAI và FunClip. [Triển khai MOSS ->](https://www.funasr.com/deploy/moss-transcribe-diarize.html)
- **Triển khai production** bao gồm dịch vụ WebSocket thời gian thực, đường suy luận theo lô/streaming với vLLM gốc, và các gói llama.cpp / GGUF đã được kiểm chứng cho Linux, macOS và Windows. [Runtime v0.2.6 ->](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) · [Hướng dẫn vLLM ->](docs/vllm_guide_vi.md)

# Bắt đầu nhanh với Transformers gốc

Phiên âm bằng gói Transformers 5.17.0 đã phát hành. Không cần cài thêm bộ công cụ hay mã Python từ xa. Bản Nano cơ bản hỗ trợ tiếng Trung, tiếng Anh và tiếng Nhật; checkpoint MLT 31 ngôn ngữ là một checkpoint riêng.

```bash
python -m pip install 'transformers==5.17.0' 'torch==2.10.0' 'torchaudio==2.10.0' 'librosa==0.11.0' 'soundfile==0.13.1'
```

[Công thức Python đầy đủ](https://www.funasr.com/en/docs/native-transformers.html) · [Audio cục bộ, xử lý theo lô và từ khóa](examples/transformers/) · [Notebook](examples/colab/fun_asr_nano_transformers.ipynb) · [Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano)

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

torch.set_num_threads(4)
model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
revision = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
audio = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/272c57b82523ada6fd87095e955f8e29100979ab/example/en.mp3"

processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False
)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False,
    dtype=torch.float32,
).to("cpu").eval()
inputs = processor.apply_transcription_request(
    audio=audio, language="en",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": 16000},
        "text_kwargs": {"padding": True},
    },
)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
new_tokens = generated[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

<a id="tinh-nang-chinh" name="tinh-nang-chinh"></a>

# Tính năng chính 🎯

**Fun-ASR** tập trung vào nhận dạng giọng nói độ chính xác cao, hỗ trợ đa ngôn ngữ theo từng checkpoint và khả năng tùy biến cho từng ngành.

- **Nhận dạng trường xa, nhiễu cao:** Được tối ưu sâu cho thu âm ở khoảng cách xa và môi trường nhiễu cao (như phòng họp, trong xe, công trường công nghiệp, v.v.), nâng độ chính xác nhận dạng lên **93%**.
- **Phương ngữ và giọng địa phương tiếng Trung:**
  - Hỗ trợ **7 phương ngữ chính**: Ngô, Quảng Đông, Mân, Khách Gia, Cám, Tương, Tấn
  - Bao phủ **26 giọng địa phương**: bao gồm Hà Nam, Thiểm Tây, Hồ Bắc, Tứ Xuyên, Trùng Khánh, Vân Nam, Quý Châu, Quảng Đông, Quảng Tây và hơn 20 vùng khác
- **Phạm vi ngôn ngữ theo checkpoint:** Fun-ASR-Nano hỗ trợ tiếng Trung, tiếng Anh, tiếng Nhật cùng các phương ngữ và giọng địa phương tiếng Trung. Fun-ASR-MLT-Nano hỗ trợ **31 ngôn ngữ**, tập trung vào các ngôn ngữ Đông Á và Đông Nam Á.
- **Nhận dạng lời bài hát có nhạc nền:** Cải thiện hiệu năng nhận dạng giọng nói khi có nhạc nền gây nhiễu, hỗ trợ nhận dạng chính xác lời bài hát.

<a id="cai-dat-moi-truong" name="cai-dat-moi-truong"></a>

# Cài đặt môi trường 🐍

```shell
git clone https://github.com/QwenAudio/Fun-ASR.git
cd Fun-ASR
pip install -r requirements.txt
```

<a id="huong-dan-su-dung" name="huong-dan-su-dung"></a>

# Giới hạn khả năng

- [x] Dấu thời gian cấp ký tự gốc của checkpoint (phụ thuộc trạng thái checkpoint trên từng hub)
  > Checkpoint `FunAudioLLM/Fun-ASR-Nano-2512` hiện tại trên ModelScope chứa đầy đủ 86 tensor `ctc_decoder.*` / `ctc.*` đã huấn luyện (`model.pt` SHA-256 `81fec8616083c69377f3ceef36aba3655660ee0ca69a5d4a1e9810cd340ca499`) và tạo ra dấu thời gian CTC gốc. Checkpoint trên Hugging Face ở revision `272c57b82523ada6fd87095e955f8e29100979ab` vẫn là artifact cũ chỉ có văn bản (`model.pt` SHA-256 `55ae0d2fee369f0f11cce0795f6927934ad17cf11b278a7e56a51272074160bb`), không có tensor CTC. Khi dùng `model.py` hiện tại của repository này với `funasr>=1.3.26`, các checkpoint không đầy đủ sẽ được xử lý theo hướng an toàn (fail closed): vẫn phiên âm được, nhưng trường `timestamps` sẽ bị bỏ qua thay vì trả về các căn chỉnh 60 ms ngẫu nhiên. Hãy dùng `hub="ms"` để có dấu thời gian gốc của checkpoint cho đến khi artifact và mã từ xa trên Hugging Face được đồng bộ. Xem [issue #70](https://github.com/QwenAudio/Fun-ASR/issues/70) và [FunASR #3496](https://github.com/modelscope/FunASR/issues/3496).
- [ ] Phân tách người nói gốc của checkpoint
  > Bản thân Fun-ASR-Nano và Fun-ASR-MLT-Nano không xuất nhãn người nói. Hãy kết hợp chúng trong FunASR với các mô hình riêng `fsmn-vad` và `cam++`, như minh họa bên dưới.
  > Để phân tách người nói ẩn danh kèm phiên âm và dấu thời gian chỉ trong một lần chạy, hãy dùng [hướng dẫn triển khai MOSS-Transcribe-Diarize](https://www.funasr.com/deploy/moss-transcribe-diarize.html) của bên thứ ba OpenMOSS. Đây là một mô hình riêng, không phải tính năng của checkpoint Fun-ASR-Nano.
- [x] Huấn luyện mô hình

# Cách sử dụng 🛠️

## Suy luận

### Chạy trên CPU / thiết bị biên — llama.cpp / GGUF (không cần GPU, không cần Python)

Chạy Fun-ASR-Nano dưới dạng **một file thực thi duy nhất, độc lập** — giống [whisper.cpp](https://github.com/ggml-org/whisper.cpp) nhưng dành cho FunASR, với độ chính xác tiếng Trung cao. Tích hợp sẵn FSMN-VAD, không cần Python khi chạy.

```bash
bash runtime/llama.cpp/download-funasr-model.sh nano ./gguf
llama-funasr-cli --enc ./gguf/funasr-encoder-f16.gguf -m ./gguf/qwen3-0.6b-q8_0.gguf -a audio.wav --vad ./gguf/fsmn-vad.gguf
```

`fsmn-vad.gguf` được lưu trữ trong repo dùng chung [FunAudioLLM/fsmn-vad-GGUF](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF), không nằm trong repo Nano GGUF. Trình tải `nano` ở trên sẽ tự động tải file này; để chỉ tải VAD từ giao diện web/CLI của Hugging Face, dùng:

```bash
hf download FunAudioLLM/fsmn-vad-GGUF --include "*.gguf" --local-dir ./gguf
```

**File thực thi dựng sẵn:** [Releases](https://github.com/QwenAudio/Fun-ASR/releases) · **Tải về & bắt đầu nhanh:** [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html) · **GGUF:** [Nano encoder/LLM](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) · [FSMN-VAD](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF) · **Tài liệu & benchmark:** [runtime/llama.cpp/](./runtime/llama.cpp/)

### Suy luận bằng funasr

```python
from funasr import AutoModel


def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0",
        # hub: tải mô hình từ ms (ModelScope) hoặc hf (Hugging Face).
        hub="hf"
    )

    wav_path = f"{model.model_path}/example/zh.mp3"
    res = model.generate(
        input=[wav_path],
        cache={},
        batch_size=1,
        hotwords=["开放时间"],
        # 中文、英文、日文 for Fun-ASR-Nano-2512
        # 中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
        # 印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
        # 匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
        # 斯洛伐克语、斯洛文尼亚语、瑞典语 for Fun-ASR-MLT-Nano-2512
        language="中文",
        itn=True, # hoặc False
    )
    text = res[0]["text"]
    print(text)

    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        remote_code="./model.py",
        device="cuda:0",
    )
    res = model.generate(input=[wav_path], cache={}, batch_size=1)
    text = res[0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

> Lưu ý: tham số `language` nhận tên ngôn ngữ bằng tiếng Trung như trong chú thích ở trên; ví dụ tiếng Việt là `language="越南语"` (chỉ có trên Fun-ASR-MLT-Nano-2512).

### Phiên âm theo lô nhanh hơn (không dùng vLLM)

Khi phiên âm audio dài hoặc nhiều file trên đường `funasr` (PyTorch), hãy truyền
`batch_size_s` để gom các đoạn VAD thành lô và đưa qua bộ giải mã LLM cùng lúc. Cách này
cải thiện đáng kể hiệu suất sử dụng GPU:

```python
res = model.generate(
    input=[wav_path],
    cache={},
    language="中文",
    itn=True,
    batch_size_s=120,   # gom các đoạn VAD, tối đa ~120 giây audio cho mỗi lần gọi LLM
)
```

Trên Fun-ASR-Nano-2512 (184 file tiếng Trung / 11.539 giây, một GPU H100), cách này
**nhanh hơn khoảng 1,6 lần** so với giải mã từng đoạn mặc định (RTFx 19,8 -> 31,8) mà không
giảm độ chính xác. Để có thông lượng cao nhất, hãy dùng đường vLLM bên dưới.

### Phân tách người nói

Ví dụ này là một pipeline FunASR được ghép từ nhiều mô hình: FSMN-VAD chia đoạn audio,
Fun-ASR-Nano phiên âm, CAM++ gán nhãn người nói và CT-Punc khôi phục
dấu câu. Các giá trị `start` và `end` là ranh giới đoạn VAD, không phải
dấu thời gian cấp ký tự gốc đáng tin cậy của checkpoint.

```python
from funasr import AutoModel


def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model="cam++",
        punc_model="ct-punc",
        device="cuda:0",
        hub="hf",
    )

    wav_path = f"{model.model_path}/example/zh.mp3"
    res = model.generate(input=[wav_path], cache={}, batch_size=1, language="中文")

    # Kết quả theo từng câu kèm nhãn người nói
    for sent in res[0]["sentence_info"]:
        print(f"Speaker {sent['spk']}: [{sent['start']}ms - {sent['end']}ms] {sent['sentence']}")


if __name__ == "__main__":
    main()
```

### Suy luận trực tiếp

```python
from model import FunASRNano


def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()

    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

<details><summary> Mô tả tham số (bấm để mở rộng) </summary>

- `model_dir`: Tên mô hình hoặc đường dẫn tới mô hình trên ổ đĩa cục bộ.
- `trust_remote_code`: Có tin cậy mã từ xa để tải phần cài đặt mô hình tùy chỉnh hay không.
- `remote_code`: Chỉ định vị trí mã của mô hình cụ thể (ví dụ `model.py` trong thư mục hiện tại), hỗ trợ cả đường dẫn tuyệt đối và tương đối.
- `device`: Chỉ định thiết bị sử dụng, ví dụ "cuda:0" hoặc "cpu".

</details>


<a id="vllm-suy-luan-thong-luong-cao" name="vllm-suy-luan-thong-luong-cao"></a>

# Suy luận thông lượng cao với vLLM 🚀

Fun-ASR tích hợp sẵn engine [vLLM](https://github.com/vllm-project/vllm) cho suy luận theo lô thông lượng cao và dịch vụ streaming thời gian thực cấp production.

> Hướng dẫn đầy đủ: [docs/vllm_guide_vi.md](docs/vllm_guide_vi.md) | Tài liệu API: [modelscope.github.io/FunASR/vllm.html](https://modelscope.github.io/FunASR/vllm.html)

### Ba chế độ

| Chế độ | Trường hợp sử dụng | Điểm vào |
|------|----------|-------|
| **Lô ngoại tuyến** | Phiên âm quy mô lớn | `AutoModelVLLM` |
| **SDK streaming** | Phụ đề thời gian thực | `FunASRNanoStreamingVLLM` |
| **Dịch vụ WebSocket** | Triển khai production | `serve_realtime_ws.py` |

### Suy luận theo lô ngoại tuyến (nhanh hơn 3-5 lần)

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=2,      # Nhiều GPU
    gpu_memory_utilization=0.8,
)

results = model.generate(
    ["audio1.wav", "audio2.wav", "audio3.wav"],
    language="中文",
    hotwords=["张三", "北京"],
)
for r in results:
    print(f"[{r['key']}] {r['text']}")
```

> **Audio dài:** `AutoModelVLLM` giải mã mỗi đầu vào trong một lượt duy nhất, nên một
> bản ghi dài (ví dụ cuộc họp kéo dài nhiều phút) có thể bị cắt cụt — hãy chia đoạn trước bằng
> VAD rồi truyền các đoạn vào, hoặc dùng API cấp cao
> `AutoModel(model=..., vad_model="fsmn-vad")`, vốn tự động chia đoạn
> audio dài.

### Dịch vụ WebSocket thời gian thực

```bash
# Khởi động server (với VAD động + phân tách người nói)
python serve_realtime_ws.py --port 10095 --language 中文 --tensor-parallel-size 2

# Client trên trình duyệt
open client_mic.html

# Client Python
python client_python.py --server ws://localhost:10095 --mic
```

**Giao thức WebSocket:**
```
Client: "START" → Server: {"event":"started"}
Client: [audio bytes] → Server: {"sentences":[...], "partial":"..."}
Client: "STOP" → Server: {"sentences":[...], "is_final":true}
```

### SDK streaming

```python
from funasr.models.fun_asr_nano.inference_vllm_streaming import FunASRNanoStreamingVLLM

engine = FunASRNanoStreamingVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512", chunk_ms=720
)

for result in engine.streaming_generate("audio.wav", language="中文"):
    print(f"[{result['audio_duration_ms']:.0f}ms] {result['fixed_text']}")
```

### Hiệu năng

| Phương pháp | Thời gian (184 file, 11.541 giây) | RTFx | CER |
|--------|---------------------|------|-----|
| PyTorch gốc | 550 giây | 21x | 8,06% |
| **vLLM (của chúng tôi)** | **34 giây** | **340x** | **8,20%** |

> **Nhanh hơn 16 lần** so với PyTorch với độ chính xác gần như tương đương (chênh lệch CER < 0,2%)

### Cài đặt

```bash
pip install "funasr>=1.3.26" "vllm>=0.12.0"
```

# Tinh chỉnh (Finetune)

Vui lòng tham khảo [docs/finetune_vi.md](docs/finetune_vi.md)

<a id="danh-gia-hieu-nang" name="danh-gia-hieu-nang"></a>

# Đánh giá hiệu năng 📝

Chúng tôi đã đánh giá Fun-ASR so với các mô hình tiên tiến khác trên các bộ benchmark mã nguồn mở, bộ dữ liệu phương ngữ tiếng Trung và các bộ kiểm thử chuyên ngành. Kết quả cho thấy Fun-ASR đạt hiệu năng vượt trội trong nhiều kịch bản khác nhau.

### 1. Hiệu năng trên bộ dữ liệu mã nguồn mở (WER %)

| Bộ kiểm thử         | GLM-ASR-nano | GLM-ASR-nano\* | Whisper-large-v3 | Seed-ASR | Seed-ASR\* | Kimi-Audio | Step-Audio2 | FireRed-ASR | Fun-ASR-nano | Fun-ASR |
| :------------------ | :----------: | :------------: | :--------------: | :------: | :--------: | :--------: | :---------: | :---------: | :----------: | :-----: |
| **Kích thước mô hình** |     1.5B     |      1.5B      |       1.6B       |    -     |     -      |     -      |      -      |    1.1B     |     0.8B     |  7.7B   |
| **Mã nguồn mở**     |      ✅      |       ✅       |        ✅        |    ❌    |     ❌     |     ✅     |     ✅      |     ✅      |      ✅      |   ❌    |
| AIShell1            |     1.81     |      2.17      |       4.72       |   0.68   |    1.63    |    0.71    |    0.63     |    0.54     |     1.80     |  1.22   |
| AIShell2            |      -       |      3.47      |       4.68       |   2.27   |    2.76    |    2.86    |    2.10     |    2.58     |     2.75     |  2.39   |
| Fleurs-zh           |      -       |      3.65      |       5.18       |   3.43   |    3.23    |    3.11    |    2.68     |    4.81     |     2.56     |  2.53   |
| Fleurs-en           |     5.78     |      6.95      |       6.23       |   9.39   |    9.39    |    6.99    |    3.03     |    10.79    |     5.96     |  4.74   |
| Librispeech-clean   |     2.00     |      2.17      |       1.86       |   1.58   |    2.8     |    1.32    |    1.17     |    1.84     |     1.76     |  1.51   |
| Librispeech-other   |     4.19     |      4.43      |       3.43       |   2.84   |    5.69    |    2.63    |    2.42     |    4.52     |     4.33     |  3.03   |
| WenetSpeech Meeting |     6.73     |      8.21      |      18.39       |   5.69   |    7.07    |    6.24    |    4.75     |    4.95     |     6.60     |  6.17   |
| WenetSpeech Net     |      -       |      6.33      |      11.89       |   4.66   |    4.84    |    6.45    |    4.67     |    4.94     |     6.01     |  5.46   |

> _Ghi chú: Kết quả Seed-ASR\* được đánh giá bằng API chính thức trên volcengine; kết quả GLM-ASR-nano\* được đánh giá bằng checkpoint mã nguồn mở._

### 2. Hiệu năng trên bộ dữ liệu chuyên ngành (WER %)

| Bộ kiểm thử        | GLM-ASR-Nano | Whisper-large-v3 | Seed-ASR  | FireRed-ASR | Kimi-Audio | Paraformer v2 | Fun-ASR-nano |  Fun-ASR  |
| :----------------- | :----------: | :--------------: | :-------: | :---------: | :--------: | :-----------: | :----------: | :-------: |
| **Kích thước mô hình** |     1.5B     |       1.6B       |     -     |    1.1B     |     8B     |     0.2B      |     0.8B     |   7.7B    |
| **Mã nguồn mở**    |      ✅      |        ✅        |    ❌     |     ✅      |     ✅     |      ✅       |      ✅      |    ❌     |
| Trường gần         |    16.95     |      16.58       |   7.20    |    10.10    |    9.02    |     8.11      |     7.79     |   6.31    |
| Trường xa          |     9.44     |      22.21       |   4.59    |    7.49     |   10.95    |     9.55      |     5.79     |   4.34    |
| Nền phức tạp       |    23.79     |      32.57       |   12.90   |    15.56    |   15.56    |     15.19     |    14.59     |   11.45   |
| Tiếng Anh tổng quát |    16.47     |      18.56       |   15.65   |    21.62    |   18.12    |     19.48     |    15.28     |   13.73   |
| Mã nguồn mở        |     4.67     |       7.05       |   3.83    |    5.31     |    3.79    |     6.23      |     4.22     |   3.38    |
| Phương ngữ         |    54.21     |      66.14       |   29.45   |    52.82    |   71.94    |     41.16     |    28.18     |   15.21   |
| Giọng địa phương   |    19.78     |      36.03       |   10.23   |    14.05    |   27.20    |     17.80     |    12.90     |   10.31   |
| Lời bài hát        |    46.56     |      54.82       |   30.26   |    42.87    |   65.18    |     50.14     |    30.85     |   21.00   |
| Hiphop             |    43.32     |      46.56       |   29.46   |    33.88    |   57.25    |     43.79     |    30.87     |   28.58   |
| **Trung bình**     |  **26.13**   |    **33.39**     | **15.95** |  **22.63**  | **31.00**  |   **23.49**   |  **16.72**   | **12.70** |

<div align="center">
<img src="images/compare_en.png" width="800" />
</div>

## Các dự án nổi bật từ bên thứ ba

- **[Fun-ASR-vllm](https://github.com/yuekaizhang/Fun-ASR-vllm)** ([@yuekaizhang](https://github.com/yuekaizhang)) — bản cài đặt vLLM của cộng đồng cho Fun-ASR (nhanh hơn ~50% so với PyTorch), có suy luận theo lô và tích hợp NVIDIA Triton Inference Server để triển khai production với số lượng truy cập đồng thời cao. Xem [#34](https://github.com/QwenAudio/Fun-ASR/issues/34).

> Hỗ trợ vLLM gốc cũng đã được tích hợp sẵn — xem mục [Suy luận thông lượng cao với vLLM 🚀](#vllm-suy-luan-thong-luong-cao) ở trên để biết về engine theo lô `AutoModelVLLM`, SDK streaming và dịch vụ WebSocket.

## Hệ sinh thái

Fun-ASR-Nano là một phần của họ **FunAudioLLM**:

| Dự án | Mô tả | Stars |
|---------|-------------|-------|
| [FunASR](https://github.com/modelscope/FunASR) | Bộ công cụ nhận dạng giọng nói cấp công nghiệp — VAD, ASR, dấu câu, phân tách người nói | [![](https://img.shields.io/github/stars/modelscope/FunASR?style=social)](https://github.com/modelscope/FunASR) |
| [SenseVoice](https://github.com/QwenAudio/SenseVoice) | Hiểu giọng nói đa ngôn ngữ — ASR + cảm xúc + sự kiện âm thanh | [![](https://img.shields.io/github/stars/QwenAudio/SenseVoice?style=social)](https://github.com/QwenAudio/SenseVoice) |
| [CosyVoice](https://github.com/QwenAudio/CosyVoice) | Tổng hợp giọng nói tự nhiên — đa ngôn ngữ, nhân bản giọng zero-shot | [![](https://img.shields.io/github/stars/QwenAudio/CosyVoice?style=social)](https://github.com/QwenAudio/CosyVoice) |
| [FunClip](https://github.com/modelscope/FunClip) | Cắt ghép video bằng AI dựa trên nhận dạng giọng nói | [![](https://img.shields.io/github/stars/modelscope/FunClip?style=social)](https://github.com/modelscope/FunClip) |

<a href="https://star-history.com/#QwenAudio/Fun-ASR&modelscope/FunASR&QwenAudio/SenseVoice&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=QwenAudio/Fun-ASR,modelscope/FunASR,QwenAudio/SenseVoice&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=QwenAudio/Fun-ASR,modelscope/FunASR,QwenAudio/SenseVoice&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=QwenAudio/Fun-ASR,modelscope/FunASR,QwenAudio/SenseVoice&type=Date" />
  </picture>
</a>

## Giấy phép

- Mã nguồn trong repository này được cấp phép theo [Apache License 2.0](./LICENSE).
- Trọng số mô hình được phân phối riêng và tuân theo thông tin giấy phép trên model card tương ứng. Model card chính thức của [Fun-ASR-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) và [Fun-ASR-MLT-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) hiện ghi Apache-2.0; hãy xem model card của đúng artifact bạn tải về.

## Trích dẫn

```bibtex
@misc{an2025funasrtechnicalreport,
      title={Fun-ASR Technical Report},
      author={Keyu An and Yanni Chen and Zhigao Chen and Chong Deng and Zhihao Du and Changfeng Gao and Zhifu Gao and Bo Gong and Xiangang Li and Yabin Li and Ying Liu and Xiang Lv and Yunjie Ji and Yiheng Jiang and Bin Ma and Haoneng Luo and Chongjia Ni and Zexu Pan and Yiping Peng and Zhendong Peng and Peiyao Wang and Hao Wang and Haoxu Wang and Wen Wang and Wupeng Wang and Yuzhong Wu and Biao Tian and Zhentao Tan and Nan Yang and Bin Yuan and Jieping Ye and Jixing Yu and Qinglin Zhang and Kun Zou and Han Zhao and Shengkui Zhao and Jingren Zhou and Yanqiao Zhu},
      year={2025},
      eprint={2509.12508},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.12508},
}
```
