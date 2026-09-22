# سرویس حذف پس‌زمینه — برای Hugging Face Spaces (تیر رایگان CPU)
#
# چرا مدل‌ها در زمان ساخت ایمیج دانلود می‌شوند: تا شروع به‌کار سرویس سریع باشد و
# هر بار از اینترنت دانلود نشود (فایل‌ها از مخزن رسمی rembg گرفته می‌شوند).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# مدل‌ها: isnet (کیفیت بالا، ۱۷۰ مگابایت) · silueta (سبک، ۴۲ مگابایت) · u2netp (خیلی سبک، ۴ مگابایت)
RUN mkdir -p models \
    && curl -L --retry 3 -o models/isnet-general-use.onnx \
       https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx \
    && curl -L --retry 3 -o models/silueta.onnx \
       https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx \
    && curl -L --retry 3 -o models/u2netp.onnx \
       https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx \
    && ls -la models/

COPY app.py .

EXPOSE 7860

# یک کارگر کافی است: هر درخواست شامل استنتاج CPU-محور است و حافظه محدود است.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
