# ComfyUI NSFW Guard (Viddexa)

Node chặn NSFW cho ComfyUI, giữ logic như `comfyui-nsfw-guard` nhưng dùng model:

- `viddexa/nsfw-detection-2-nano`
- `viddexa/nsfw-detection-2-mini`
- Backend ưu tiên: `moderators` (PyPI)
- Fallback: `transformers`

## Cài đặt

1. Copy thư mục này vào:
   - `ComfyUI/custom_nodes/comfyui-nsfw-guard-viddexa`
2. Cài dependency:

```bash
pip install -r requirements.txt
```

3. Restart ComfyUI.

## Dùng node

- Node name: `NSFW Check (HF Classifier)`
- Input: `image`
- Output: `image` (pass-through nếu safe)

### Dùng chung 1 model cho nhiều điểm chặn

Node mới:
- `NSFW Load Model (HF)` -> output `NSFW_GUARD_MODEL`
- `NSFW Check (HF, Shared Model)` -> nhận `NSFW_GUARD_MODEL + image`

`NSFW Load Model (HF)` có dropdown chọn model:
- `viddexa/nsfw-detection-2-nano`
- `viddexa/nsfw-detection-2-mini`

Workflow gợi ý:

```text
                 +-> [NSFW Check (HF, Shared Model)] -> (nhánh input check)
[NSFW Load Model]
                 +-> [NSFW Check (HF, Shared Model)] -> (nhánh output check)
```

Cách này chỉ load model 1 lần, rồi dùng lại cho cả đầu input và output.

Nếu phát hiện NSFW, node sẽ:
- gửi event `nsfw_guard.content_blocked`
- `interrupt_processing(True)`
- raise error có type `nsfw_content_detected`

## Rule chặn hiện tại

- Chặn: `porn`, `hentai`, `sexy`
- Cho pass: `drawing`

## Model cache

Model được tự động tải về:

- `ComfyUI/models/nsfw/viddexa_nsfw_detection_2_nano/`
