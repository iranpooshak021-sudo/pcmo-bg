"""
سرویس حذف پس‌زمینه — نسخه‌ی سبک برای میزبان رایگان (Render free · 512MB RAM).

نکات مهم این نسخه:
  * مدل پیش‌فرض `silueta` (۴۲ مگابایت) است تا در ۵۱۲ مگ رم جا شود؛ `isnet`
    (کیفیت بالاتر، ۱۷۰ مگابایت) هم موجود است ولی روی پلن رایگان ممکن است به
    سقف حافظه بخورد.
  * پیش‌پردازش **فقط تقسیم بر ۲۵۵** است (بدون نرمال‌سازی ImageNet) — با
    نرمال‌سازی ImageNet خروجی مدل صفر می‌شود و ماسک خالی می‌دهد (این را روی
    تصاویر واقعی تست کردیم).
  * خروجی با کمینه-بیشینه نرمال و مستقیماً به کانال آلفا تبدیل می‌شود
    (پیاده‌سازی‌های مرجع هم همین کار را می‌کنند).
  * نسبت پوشش ماسک در هدر `X-Coverage` برمی‌گردد تا افزونه جلوی پاک‌شدن کل
    تصویر را بگیرد (اگر مدل سوژه را پیدا نکرد).
"""

import io
import os
import threading

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageFilter

# کلید مشترک با وردپرس؛ از متغیر محیطی خوانده می‌شود و در کد نیست.
TOKEN = os.environ.get("PCMO_TOKEN", "")

# نام مدل → (مسیر فایل، ابعاد ورودی)
MODELS = {
    "silueta": ("models/silueta.onnx", 320),
    "u2netp": ("models/u2netp.onnx", 320),
    "isnet": ("models/isnet-general-use.onnx", 1024),
}
DEFAULT_MODEL = os.environ.get("PCMO_MODEL", "silueta")

MAX_UPLOAD = int(os.environ.get("PCMO_MAX_MB", "15")) * 1024 * 1024

_lock = threading.Lock()
_sessions: dict[str, tuple[ort.InferenceSession, int]] = {}


def session_for(name: str) -> tuple[ort.InferenceSession, int]:
    """مدل را یک‌بار بارگذاری و در حافظه نگه می‌دارد (هم‌زمانی امن)."""
    with _lock:
        if name not in _sessions:
            path, size = MODELS[name]
            if not os.path.exists(path):
                raise HTTPException(500, f"فایل مدل «{name}» پیدا نشد.")
            options = ort.SessionOptions()
            # تعداد تردها محدود می‌شود تا روی سرور رایگان (کم‌منبع) پایدار بماند.
            options.intra_op_num_threads = int(os.environ.get("PCMO_THREADS", "2"))
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _sessions[name] = (
                ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"]),
                size,
            )
        return _sessions[name]


def build_alpha(image: Image.Image, name: str) -> Image.Image:
    """ماسک سوژه را می‌سازد و به ابعاد تصویر اصلی برمی‌گرداند."""
    session, size = session_for(name)

    rgb = image.convert("RGB").resize((size, size), Image.LANCZOS)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0  # بدون ImageNet
    tensor = np.transpose(pixels, (2, 0, 1))[None].astype(np.float32)

    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    mask = raw[0, 0] if raw.ndim == 4 else raw[0]

    span = float(mask.max() - mask.min())
    mask = (mask - mask.min()) / (span if span > 1e-6 else 1.0)

    return Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(image.size, Image.LANCZOS)


app = FastAPI(title="PCmo Background Removal", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "models": list(MODELS),
        "default": DEFAULT_MODEL,
        "auth_required": bool(TOKEN),
    }


async def _process(data: bytes, token: str, name: str, feather: int) -> Response:
    """منطق مشترک پردازش (ورودی بایت خام تصویر)."""
    if TOKEN and token != TOKEN:
        raise HTTPException(403, "کلید نامعتبر است.")

    if not name:
        name = DEFAULT_MODEL
    if name not in MODELS:
        raise HTTPException(400, "مدل ناشناخته است.")

    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "فایل بزرگ‌تر از حد مجاز است.")
    if not data:
        raise HTTPException(400, "فایلی ارسال نشد.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise HTTPException(400, "فایل تصویری معتبر نیست.")

    alpha = build_alpha(image, name)
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))

    output = image.convert("RGBA")
    output.putalpha(alpha)

    buffer = io.BytesIO()
    output.save(buffer, format="PNG", optimize=True)

    coverage = float((np.asarray(alpha) > 128).mean())

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"X-Coverage": f"{coverage:.4f}", "X-Model": name},
    )


@app.post("/remove-bg/raw")
async def remove_bg_raw(
    request: Request,
    x_token: str = Header(default=""),
    x_model: str = Header(default=""),
    feather: int = Query(default=0, ge=0, le=6),
) -> Response:
    """
    مخصوص PHP: **بدنه‌ی درخواست خودِ فایل تصویر است** (بدون multipart).

    چرا جدا: وردپرس به‌سادگی نمی‌تواند multipart بفرستد و FastAPI برای فیلد فایل
    انتظار `multipart/form-data` دارد؛ با بدنه‌ی خام پاسخ ۴۲۲ می‌دهد. این مسیر
    مستقل از آن است و ساده‌ترین راه برای سرویس‌به‌سرویس است.
    """
    return await _process(await request.body(), x_token, x_model, feather)


@app.post("/remove-bg")
async def remove_bg(
    request: Request,
    file: UploadFile | None = File(default=None),
    x_token: str = Header(default=""),
    x_model: str = Header(default=""),
    model: str = Query(default=""),
    feather: int = Query(default=0, ge=0, le=6),
) -> Response:
    """
    دو حالت ورودی پشتیبانی می‌شود:

      * `multipart/form-data` با فیلد `file` (برای مرورگر و curl)
      * **بدنه‌ی خام** با `Content-Type: image/...` (برای PHP)
    """
    data = await file.read() if file is not None else await request.body()
    return await _process(data, x_token, model or x_model, feather)
