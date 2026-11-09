# Tinh chỉnh (Finetune)

「[简体中文](finetune_zh.md)」|「[English](finetune.md)」|「Tiếng Việt」

## Yêu cầu

```
pip install "funasr>=1.3.26"
```

## Chuẩn bị dữ liệu

Ví dụ dữ liệu

```
head -n1 data/train_example.jsonl | jq

{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "语音转写：<|startofspeech|>!https://modelscope.cn/datasets/FunAudioLLM/funasr-demo/resolve/master/audios/IT0011W0002.wav<|endofspeech|>"
    },
    {
      "role": "assistant",
      "content": "几点了？"
    }
  ],
  "speech_length": 145,
  "text_length": 3
}
```

Xem đầy đủ tại `data/train_example.jsonl`

Mô tả:

- Nội dung của `system` cố định là `You are a helpful assistant.`
- Nội dung của `user` gồm câu lệnh (prompt) và đường dẫn tới file audio (đặt giữa `<|startofspeech|>!` và `<|endofspeech|>`).
  - Prompt mặc định là `语音转写：` và `Speech transcription: `.
  - Với từng ngôn ngữ tương ứng, có thể kết hợp prompt, ví dụ `语音转写成英文：` và `Transcribe speech into Chinese: `.
  - Khi nhãn văn bản của file audio không chứa chữ số Ả Rập hoặc dấu câu, có thể dùng `语音转写，不进行文本规整：` và `Speech transcription without text normalization: `.
- Nội dung của `assistant` là nhãn văn bản (bản phiên âm) của file audio.
- speech_length: Số khung fbank của file audio (mỗi khung 10ms).
- text_length: Số token trong nhãn văn bản của file audio (mã hóa bằng `Qwen/Qwen3-0.6B`).

- `messages[2]["content"]`: bản phiên âm
- `speech_length`: số khung fbank của file audio
- `text_length`: số token của bản phiên âm (tách token bằng `Qwen3-0.6B`)

Chúng tôi cung cấp công cụ chuyển đổi định dạng dữ liệu `scp2jsonl.py`, có thể chuyển các định dạng dữ liệu huấn luyện nhận dạng giọng nói phổ biến như wav scp và file phiên âm sang định dạng ChatML.

`train_wav.scp`

```
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
```

`train_text.txt`

```
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0916W0489 湖北一公司以员工名义贷款数十员工负债千万
```

```
python tools/scp2jsonl.py \
  ++scp_file=data/train_wav.scp \
  ++transcript_file=data/train_text.txt \
  ++jsonl_file=data/train_example.jsonl
```

## Tinh chỉnh

Sửa các tham số `audio_encoder_conf.freeze`, `audio_adaptor_conf.freeze` và `llm_conf.freeze` trong `finetune.sh`.

Đặt tham số `freeze` của các mô-đun cần tinh chỉnh thành false (mặc định chỉ tinh chỉnh LLM).

Để biết thêm chi tiết về các tham số, tham khảo: [Huấn luyện và kiểm thử mô hình SenseVoice](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md#Model%20Training%20and%20Testing)

```
bash finetune.sh
```

### Cấu hình khuyến nghị

- Với dữ liệu huấn luyện dưới 1000 giờ, nên tinh chỉnh audio_adaptor.
- Với dữ liệu huấn luyện dưới 5000 giờ, nên tinh chỉnh audio_encoder và audio_adaptor.
- Với dữ liệu huấn luyện trên 10000 giờ, nên tinh chỉnh toàn bộ tham số.

## Đánh giá mô hình

Sau khi tinh chỉnh xong, bạn có thể giải mã bằng mô hình mới thông qua script decode.py:

```
python decode.py \
  ++model_dir=/path/to/finetuned \
  ++scp_file=data/val_wav.scp \
  ++output_file=output.txt
```

Sau khi giải mã xong, cần chuẩn hóa văn bản cho cả nhãn và kết quả nhận dạng, rồi tính WER:

```
python tools/whisper_mix_normalize.py data/val_text.txt data/val_norm.txt
python tools/whisper_mix_normalize.py output.txt output_norm.txt
compute-wer data/val_norm.txt output_norm.txt cer.txt
tail -n8 cer.txt
```

Với tiếng Nhật, hãy chuẩn hóa cả hai file sang katakana bằng OpenJTalk trước khi tính
CER. Quy trình chạy bằng một lệnh và các tùy chọn tái lập kết quả được mô tả trong
[Đánh giá CER tiếng Nhật](japanese_cer.md) (tiếng Anh):

```bash
python tools/compute_ja_cer.py data/val_text.txt output.txt japanese_cer.txt
```
