"""
سرویس حذف پس‌زمینه — نسخه‌ی اجرا روی Hugging Face Spaces (ZeroGPU، رایگان).

معماری نهایی (با آزمون و خطای واقعی روی همین Space):

  * سخت‌افزار ZeroGPU **خودش** یک سرور Gradio روی پورت پیش‌فرض بالا می‌آورد؛ پس
    اجرای uvicorn خودمان ممکن نیست (`address already in use`) و مسیرهای سفارشی
    هم زیر Gradio گم می‌شوند (۴۰۴).
  * بنابراین ارتباط افزونه از طریق **API خود Gradio** انجام می‌شود و برای اینکه
    هیچ فایل/مسیر موقتی لازم نباشد، ورودی و خروجی **base64** است:

        POST /gradio_api/call/api_cutout   {"data": ["<image b64>", "silueta", "<token>"]}
        → {"event_id": "..."}
        GET  /gradio_api/call/api_cutout/<event_id>
        → {"data": ["<png b64>", "0.5068"]}

  * سخت‌افزار ZeroGPU بدون وجود یک تابع `@spaces.GPU` بالا نمی‌آید
    («No @spaces.GPU function detected during startup»)؛ یک تابع تشریفاتی داریم
    که هرگز صدا زده نمی‌شود و سهمیه‌ی GPU مصرف نمی‌کند.

تله‌های مدل که در کد لحاظ شده:
  * پیش‌پردازش **فقط /255** (بدون نرمال‌سازی ImageNet؛ وگرنه خروجی صفر و ماسک
    خالی می‌شود و کل تصویر پاک می‌گردد).
  * نرمال‌سازی خروجی کمینه-بیشینه (سیگموید ماسک بد می‌دهد).
  * `intra_op_num_threads` محدود، تا روی کانتینر مشترک پایدار بماند.
"""

import base64
import io
import os
import threading
import urllib.request

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

try:  # pragma: no cover
    import spaces  # noqa: F401
except Exception:
    spaces = None


if spaces is not None:

    @spaces.GPU(duration=1)
    def _gpu_probe() -> bool:
        """فقط برای الزام ZeroGPU؛ هرگز اجرا نمی‌شود (سهمیه‌ی GPU مصرف نمی‌کند)."""
        return True

else:

    def _gpu_probe() -> bool:
        return True


# کلید مشترک با وردپرس؛ در Space → Settings → Variables and secrets ست می‌شود.
TOKEN = os.environ.get("PCMO_TOKEN", "")

MODELS = {
    "silueta": ("models/silueta.onnx", 320, "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"),
    "u2netp": ("models/u2netp.onnx", 320, "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"),
}
DEFAULT_MODEL = os.environ.get("PCMO_MODEL", "silueta")

MAX_BYTES = int(os.environ.get("PCMO_MAX_MB", "15")) * 1024 * 1024

_lock = threading.Lock()
_sessions: dict[str, tuple[ort.InferenceSession, int]] = {}


def ensure_model(name: str) -> str:
    """فایل مدل را (یک‌بار در عمر کانتینر) دانلود می‌کند."""
    path, _, url = MODELS[name]
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path

    os.makedirs("models", exist_ok=True)
    partial = path + ".part"
    print(f"[pcmo] downloading model {name} …", flush=True)
    with urllib.request.urlopen(url, timeout=600) as response, open(partial, "wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(partial, path)
    print(f"[pcmo] model ready: {os.path.getsize(path) // 1048576} MB", flush=True)
    return path


def session_for(name: str) -> tuple[ort.InferenceSession, int]:
    with _lock:
        if name not in _sessions:
            path = ensure_model(name)
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(os.environ.get("PCMO_THREADS", "2"))
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _sessions[name] = (
                ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"]),
                MODELS[name][1],
            )
        return _sessions[name]


def build_alpha(image: Image.Image, name: str) -> Image.Image:
    session, size = session_for(name)

    rgb = image.convert("RGB").resize((size, size), Image.LANCZOS)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = np.transpose(pixels, (2, 0, 1))[None].astype(np.float32)

    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    mask = raw[0, 0] if raw.ndim == 4 else raw[0]

    span = float(mask.max() - mask.min())
    mask = (mask - mask.min()) / (span if span > 1e-6 else 1.0)

    return Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(image.size, Image.LANCZOS)


def cut_out(image: Image.Image, name: str, feather: int = 0) -> tuple[Image.Image, float]:
    alpha = build_alpha(image, name)
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    output = image.convert("RGBA")
    output.putalpha(alpha)
    coverage = float((np.asarray(alpha) > 128).mean())
    return output, coverage


def ui_cutout(image, model_name):
    """رابط کاربری مرورگر (برای تست دستی)."""
    if image is None:
        return None
    output, _ = cut_out(image, model_name or DEFAULT_MODEL, 0)
    return output


def api_cutout(image_b64: str, model_name: str, token: str):
    """
    اندپوینت افزونه‌ی وردپرس: تصویر base64 می‌گیرد و PNG شفاف base64 + پوشش
    ماسک برمی‌گرداند. توکن برای جلوگیری از استفاده‌ی ناخواسته بررسی می‌شود.
    """
    if TOKEN and (token or "").strip() != TOKEN:
        return "", "ERR:token"

    name = (model_name or DEFAULT_MODEL).strip()
    if name not in MODELS:
        return "", "ERR:model"

    payload = (image_b64 or "").strip()
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[-1]
    if not payload:
        return "", "ERR:empty"

    try:
        raw_bytes = base64.b64decode(payload, validate=False)
    except Exception:
        return "", "ERR:decode"

    if len(raw_bytes) > MAX_BYTES:
        return "", "ERR:big"

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except Exception:
        return "", "ERR:image"

    output, coverage = cut_out(image, name, 0)
    buffer = io.BytesIO()
    output.save(buffer, format="PNG", optimize=True)

    return base64.b64encode(buffer.getvalue()).decode("ascii"), f"{coverage:.4f}"


with gr.Blocks(title="حذف پس‌زمینه") as demo:
    gr.Markdown("# حذف پس‌زمینه\nسرویس افزونه‌ی چاپ سفارشی وردپرس.")

    with gr.Row():
        ui_image = gr.Image(type="pil", label="تصویر")
        ui_out = gr.Image(type="pil", label="بدون پس‌زمینه")

    ui_model = gr.Dropdown(choices=list(MODELS), value=DEFAULT_MODEL, label="مدل")
    ui_btn = gr.Button("حذف پس‌زمینه", variant="primary")
    ui_btn.click(ui_cutout, inputs=[ui_image, ui_model], outputs=[ui_out], api_name="ui_cutout")

    gr.Markdown("---\n### اندپوینت API افزونه (base64)")
    api_in = gr.Textbox(label="ورودی base64")
    api_model = gr.Dropdown(choices=list(MODELS), value=DEFAULT_MODEL, label="مدل")
    api_token = gr.Textbox(label="توکن")
    api_btn = gr.Button("اجرا")
    api_out = gr.Textbox(label="خروجی base64")
    api_cov = gr.Textbox(label="پوشش ماسک")
    api_btn.click(api_cutout, inputs=[api_in, api_model, api_token], outputs=[api_out, api_cov], api_name="api_cutout")


# گرم‌کردن مدل در پس‌زمینه تا اولین درخواست، معطل دانلود نماند.
threading.Thread(target=lambda: session_for(DEFAULT_MODEL), daemon=True).start()


if __name__ == "__main__":
    demo.launch()
