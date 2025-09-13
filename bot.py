import os
import io
import json
import time
import base64
import logging
import re
from uuid import uuid4
from typing import Tuple

import aiohttp
from dotenv import load_dotenv
from PIL import Image

from telegram import Update, InputFile, WebAppInfo, Message, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

# ───────── конфиг ─────────
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://funfishinggame.store").rstrip("/")
UPLOAD_DIR = "/var/www/funfishinggame.store/html/uploads"

# переводчики (любой из них опционален)
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "").strip()
DEEPL_API_URL = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate").strip()
LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com").rstrip("/")
LIBRETRANSLATE_API_KEY = os.getenv("LIBRETRANSLATE_API_KEY", "").strip()

if not BOT_TOKEN or not STABILITY_API_KEY:
    raise SystemExit("BOT_TOKEN и/или STABILITY_API_KEY не заданы в .env")

os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inpaintbot")

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=180, connect=30)
LAST_IMAGE_URL: dict[int, str] = {}
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# ───────── утилиты ─────────
async def download_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
        async with sess.get(url) as r:
            r.raise_for_status()
            return await r.read()

async def save_tg_file_to_uploads(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    f = await context.bot.get_file(file_id)
    tg_url = f.file_path if f.file_path.startswith("http") else f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"

    ext = os.path.splitext(f.file_path)[1].lower() or ".jpg"
    name = f"{int(time.time())}_{uuid4().hex}{ext}"
    local_path = os.path.join(UPLOAD_DIR, name)

    data = await download_bytes(tg_url)

    # webp -> png
    if ext == ".webp":
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            data = buf.getvalue()
            name = name.rsplit(".", 1)[0] + ".png"
            local_path = os.path.join(UPLOAD_DIR, name)
        except Exception:
            pass

    with open(local_path, "wb") as out:
        out.write(data)

    return local_path, f"{PUBLIC_BASE_URL}/uploads/{name}"

def _unpack_bitmask_to_L(b64: str, w: int, h: int) -> Image.Image:
    raw = base64.b64decode(b64)
    total = w * h
    out = bytearray(total)
    idx = 0
    for b in raw:
        for bit in range(7, -1, -1):
            if idx >= total:
                break
            out[idx] = 255 if ((b >> bit) & 1) else 0
            idx += 1
        if idx >= total:
            break
    return Image.frombytes("L", (w, h), bytes(out))

def _build_fullsize_mask(nat_w: int, nat_h: int, bbox: Tuple[int, int, int, int], small_mask: Image.Image) -> Image.Image:
    x, y, w, h = map(int, bbox)
    mask_full = Image.new("L", (nat_w, nat_h), 0)
    if w > 0 and h > 0 and x < nat_w and y < nat_h:
        w = min(w, nat_w - x)
        h = min(h, nat_h - y)
        if w > 0 and h > 0:
            m = small_mask.resize((w, h), Image.NEAREST)
            mask_full.paste(m, (x, y))
    return mask_full

async def call_stability_inpaint(image_bytes: bytes, mask_png: bytes, prompt: str) -> bytes:
    url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "image/*"}

    form = aiohttp.FormData()
    form.add_field("image", image_bytes, filename="image.png", content_type="image/png")
    form.add_field("mask",  mask_png,   filename="mask.png",  content_type="image/png")
    if prompt.strip():
        form.add_field("prompt", prompt.strip())
    form.add_field("output_format", "png")

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
        async with sess.post(url, headers=headers, data=form) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"Stability API error: {resp.status} {txt}")
            ctype = resp.headers.get("Content-Type", "")
            if "image/" not in ctype:
                txt = await resp.text()
                raise RuntimeError(f"Unexpected content-type: {ctype} | body: {txt[:400]}")
            return await resp.read()

def reply_wa(url: str) -> ReplyKeyboardMarkup:
    kb = [[KeyboardButton(text="🖌 Открыть кисть", web_app=WebAppInfo(url=url))]]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)

# ───────── перевод RU→EN ─────────
async def translate_ru_to_en(text: str) -> tuple[str, str | None]:
    """
    Возвращает (перевод, провайдер) либо (исходный, None) если перевода не требуется/не вышло.
    """
    if not text or not CYRILLIC_RE.search(text):
        return text, None

    # 1) DeepL (если есть ключ)
    if DEEPL_API_KEY:
        try:
            headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
            data = aiohttp.FormData()
            data.add_field("text", text)
            data.add_field("target_lang", "EN")
            data.add_field("source_lang", "RU")
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
                async with sess.post(DEEPL_API_URL, headers=headers, data=data) as r:
                    if r.status == 200:
                        j = await r.json()
                        tr = j.get("translations", [{}])[0].get("text")
                        if tr:
                            return tr, "deepl"
                    else:
                        log.warning("DeepL error %s: %s", r.status, await r.text())
        except Exception as e:
            log.warning("DeepL exception: %s", e)

    # 2) LibreTranslate (по умолчанию публичный инстанс)
    try:
        url = f"{LIBRETRANSLATE_URL}/translate"
        payload = {"q": text, "source": "ru", "target": "en", "format": "text"}
        if LIBRETRANSLATE_API_KEY:
            payload["api_key"] = LIBRETRANSLATE_API_KEY
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
            async with sess.post(url, json=payload) as r:
                if r.status == 200:
                    j = await r.json()
                    tr = j.get("translatedText")
                    if tr:
                        return tr, "libre"
                else:
                    log.warning("LibreTranslate error %s: %s", r.status, await r.text())
    except Exception as e:
        log.warning("LibreTranslate exception: %s", e)

    # если всё мимо — возвращаем исходник
    return text, None

# ───────── хендлеры ─────────
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Пришли фото (как фото или документ), затем нажми «🖌 Открыть кисть».")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅ Бот онлайн.")

async def brush_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = LAST_IMAGE_URL.get(update.effective_user.id)
    ver = int(time.time())
    wa_url = f"{PUBLIC_BASE_URL}/index.html?v={ver}" + (f"&img={last}" if last else "")
    await update.message.reply_text("Открой WebApp:", reply_markup=reply_wa(wa_url))

async def got_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg.photo:
        file_id = msg.photo[-1].file_id
    else:
        if not (msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/")):
            await msg.reply_text("Пришли изображение в виде фото или документа.")
            return
        file_id = msg.document.file_id

    local_path, public_url = await save_tg_file_to_uploads(file_id, context)
    LAST_IMAGE_URL[update.effective_user.id] = public_url
    log.info("Saved: %s -> %s", local_path, public_url)

    ver = int(time.time())
    wa_url = f"{PUBLIC_BASE_URL}/index.html?v={ver}&img={public_url}"
    await msg.reply_text("✅ Изображение загружено. Нажми «🖌 Открыть кисть».", reply_markup=reply_wa(wa_url))

async def webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg: Message = update.message
    wad = getattr(msg, "web_app_data", None)
    if not (wad and wad.data):
        return

    raw = wad.data.encode("utf-8")
    log.info("web_app_data: %d bytes | chat=%s user=%s", len(raw), msg.chat_id, update.effective_user.id)

    try:
        data = json.loads(wad.data)
    except Exception as e:
        await msg.reply_text(f"❌ Ошибка JSON: {e}")
        return

    img_url = data.get("img") or LAST_IMAGE_URL.get(update.effective_user.id)
    prompt  = (data.get("prompt") or "").strip()
    nat_w   = int(data.get("nat_w") or 0)
    nat_h   = int(data.get("nat_h") or 0)
    bm_w    = int(data.get("bm_w") or 0)
    bm_h    = int(data.get("bm_h") or 0)
    bbox    = data.get("bbox") or [0, 0, 0, 0]
    b64mask = data.get("bitmask")

    if not img_url or not b64mask or not nat_w or not nat_h or not bm_w or not bm_h:
        await msg.reply_text("❌ Нет обязательных полей (img/mask/size). Откройте кисть и попробуйте снова.")
        return

    # Маска
    try:
        small = _unpack_bitmask_to_L(b64mask, bm_w, bm_h)
        full  = _build_fullsize_mask(nat_w, nat_h, tuple(map(int, bbox)), small)
        buff = io.BytesIO()
        full.save(buff, format="PNG")
        mask_bytes = buff.getvalue()
    except Exception as e:
        await msg.reply_text(f"❌ Ошибка сборки маски: {e}")
        return

    # Исходник
    try:
        prefix = f"{PUBLIC_BASE_URL}/uploads/"
        if img_url.startswith(prefix):
            name = img_url.split("/uploads/")[-1].split("?")[0]
            local_path = os.path.join(UPLOAD_DIR, name)
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    image_bytes = f.read()
            else:
                image_bytes = await download_bytes(img_url)
        else:
            image_bytes = await download_bytes(img_url)
    except Exception as e:
        await msg.reply_text(f"❌ Ошибка загрузки исходного изображения: {e}")
        return

    # Перевод промпта (если нужен)
    caption_prompt = ""
    prompt_to_send = prompt
    if prompt:
        translated, provider = await translate_ru_to_en(prompt)
        if provider and translated:
            log.info("Prompt translated via %s: '%s' -> '%s'", provider, prompt, translated)
            prompt_to_send = translated
            caption_prompt = f"\nPrompt (ru→en): {prompt} → {translated}"
        else:
            caption_prompt = f"\nPrompt: {prompt}"

    # Stability
    try:
        result_png = await call_stability_inpaint(image_bytes, mask_bytes, prompt_to_send or "inpaint")
    except Exception as e:
        await msg.reply_text(f"❌ Ошибка Stability: {e}")
        return

    # Ответ
    try:
        await msg.reply_photo(
            photo=InputFile(io.BytesIO(result_png), filename="result.png"),
            caption=f"Готово ✨{caption_prompt}"
        )
    except Exception as e:
        await msg.reply_text(f"❌ Ошибка отправки результата: {e}")

# подстраховка и лог
async def route_webapp_if_any(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = getattr(update, "message", None)
    if msg and getattr(msg, "web_app_data", None) and msg.web_app_data.data:
        await webapp_data(update, context)

async def log_every_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = getattr(update, "message", None)
    if not msg:
        return
    marker = "WEB_APP" if (getattr(msg, "web_app_data", None) and msg.web_app_data.data) else "MSG"
    t = (msg.text or "")[:80]
    log.info("[rx:%s] chat=%s text=%s", marker, msg.chat_id, t)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error", exc_info=context.error)

# ───────── запуск ─────────
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping",  ping_cmd))
    app.add_handler(CommandHandler("brush", brush_cmd))

    app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, webapp_data))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, got_image))
    app.add_handler(MessageHandler(filters.ALL, route_webapp_if_any))
    app.add_handler(MessageHandler(filters.ALL, log_every_message))

    app.add_error_handler(error_handler)

    log.info("Bot started")
    app.run_polling(allowed_updates=None, drop_pending_updates=False)

if __name__ == "__main__":
    main()
