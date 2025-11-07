import os
import math
import typing as tp
import cv2
import numpy as np
import torch
from torch import nn
import gradio as gr
from huggingface_hub import InferenceClient

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 🧠 PART 1: FSRCNN Image Upscaling
# ============================================================

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9,
            stride=scale_factor, padding=9 // 2,
            output_padding=scale_factor - 1
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE: dict[int, tuple[FSRCNN, bool]] = {}
WEIGHTS_PATHS = {2: "models/fsrcnn_x2.pth", 3: "models/fsrcnn_x3.pth", 4: "models/fsrcnn_x4.pth"}


def try_load_weights(model, weights_path):
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[FSRCNN] No valid weights at {weights_path}. Falling back to Bicubic.")
        return False
    try:
        checkpoint = torch.load(weights_path, map_location=Device, weights_only=False)
        model.load_state_dict(checkpoint, strict=True)
        print(f"[FSRCNN] Loaded weights from {weights_path}")
        return True
    except Exception as e:
        print(f"[FSRCNN] Failed to load weights: {e}")
        return False


def get_model(scale, weights_path=None):
    if scale not in MODEL_CACHE:
        model = FSRCNN(scale_factor=scale).to(Device).eval()
        has_weights = try_load_weights(model, weights_path)
        MODEL_CACHE[scale] = (model, has_weights)
    return MODEL_CACHE[scale]


def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)


def ycbcr_to_rgb(img_ycrcb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)


def run_fsrcnn_on_y(y: np.ndarray, model: FSRCNN) -> np.ndarray:
    y_f = y.astype(np.float32) / 255.0
    tens = torch.from_numpy(y_f).unsqueeze(0).unsqueeze(0).to(Device)
    with torch.inference_mode():
        out = model(tens)
    out_np = out.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    return (out_np * 255.0 + 0.5).astype(np.uint8)


def fsrcnn_upscale_rgb(img_rgb: np.ndarray, scale: int, weights: tp.Optional[str] = None) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    model, has_weights = get_model(scale, weights)
    if not has_weights:
        return cv2.resize(img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    ycrcb = rgb_to_ycbcr(img_rgb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    y_sr = run_fsrcnn_on_y(y, model)
    new_w, new_h = w * scale, h * scale
    cr_up = cv2.resize(cr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    ycrcb_up = np.stack([y_sr, cr_up, cb_up], axis=-1)
    return ycbcr_to_rgb(ycrcb_up)


def maybe_downscale_for_memory(img_rgb: np.ndarray, max_pixels: int = 8_000_000) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h * w <= max_pixels:
        return img_rgb
    scale = (max_pixels / (h * w)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def upscale_ui(image: np.ndarray, scale_factor: int, method: str):
    if image is None:
        return None, "Please upload an image."
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 4:
        image = image[..., :3]

    image = maybe_downscale_for_memory(image)
    weights_path = WEIGHTS_PATHS.get(scale_factor)
    if method == "FSRCNN (Y channel)":
        out = fsrcnn_upscale_rgb(image, scale_factor, weights_path)
        status = f"Used FSRCNN x{scale_factor} (bundled weights)."
    else:
        out = cv2.resize(image, (image.shape[1]*scale_factor, image.shape[0]*scale_factor), interpolation=cv2.INTER_CUBIC)
        status = f"Used Bicubic x{scale_factor}."
    return out, status


# ============================================================
# 🌍 PART 2: Multilingual Translator
# ============================================================

HF_TOKEN = os.environ.get("HF_TOKEN", None)
client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

lang_map = {
    "English": "en_XX", "French": "fr_XX", "Spanish": "es_XX", "German": "de_DE",
    "Hindi": "hi_IN", "Chinese": "zh_CN", "Japanese": "ja_XX", "Korean": "ko_KR",
    "Tamil": "ta_IN", "Telugu": "te_IN", "Arabic": "ar_AR", "Russian": "ru_RU"
    # You can add full map from your existing file if needed
}

def translate_text(text, src_lang, tgt_lang):
    if not text.strip():
        return "Please enter any text to translate 😃"
    try:
        src_code, tgt_code = lang_map[src_lang], lang_map[tgt_lang]
        result = client.translation(
            text,
            model="facebook/mbart-large-50-many-to-many-mmt",
            src_lang=src_code,
            tgt_lang=tgt_code
        )
        return result.translation_text
    except Exception as e:
        return f"Error in translation: {str(e)}"


# ============================================================
# 🎨 Combine into One Interface with Tabs
# ============================================================

custom_theme = gr.themes.Default().set(
    button_primary_background_fill="#1769aa",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e0e0e0",
    button_secondary_text_color="#222222"
)

with gr.Blocks(theme=custom_theme, title="AI Multi-Tool: FSRCNN & Translator") as demo:
    gr.Markdown("# 🚀 AI Multi-Tool Suite\nChoose an application below 👇")

    with gr.Tabs():
        # Tab 1: FSRCNN Upscaler
        with gr.Tab("🖼️ Image Upscaling"):
            with gr.Row():
                with gr.Column():
                    inp_img = gr.Image(type="numpy", label="Input Image")
                    scale = gr.Dropdown([2, 3, 4], value=2, label="Upscale Factor")
                    method = gr.Radio(["FSRCNN (Y channel)", "Bicubic"], value="FSRCNN (Y channel)")
                    run_btn = gr.Button("Upscale", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                    status_box = gr.Textbox(label="Status")
                with gr.Column():
                    out_img = gr.Image(type="numpy", label="Upscaled Output")

            run_btn.click(fn=upscale_ui, inputs=[inp_img, scale, method], outputs=[out_img, status_box])
            clear_btn.click(fn=lambda: (None, 2, "FSRCNN (Y channel)", None, ""), 
                            outputs=[inp_img, scale, method, out_img, status_box])

        # Tab 2: Translator
        with gr.Tab("🌍 Text Translator"):
            with gr.Row():
                with gr.Column():
                    src_lang = gr.Dropdown(choices=list(lang_map.keys()), value="English", label="Source Language")
                    input_text = gr.Textbox(lines=4, label="Enter Text")
                with gr.Column():
                    tgt_lang = gr.Dropdown(choices=list(lang_map.keys()), value="French", label="Target Language")
                    output_text = gr.Textbox(lines=4, label="Translation", interactive=False)

            translate_btn = gr.Button("Translate ✨", variant="primary")
            clear_btn2 = gr.Button("Clear", variant="secondary")

            translate_btn.click(fn=translate_text, inputs=[input_text, src_lang, tgt_lang], outputs=output_text)
            clear_btn2.click(fn=lambda: ("", "English", "French", ""), outputs=[input_text, src_lang, tgt_lang, output_text])

if __name__ == "__main__":
    demo.launch()
