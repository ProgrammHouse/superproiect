"""Telegram-агент на Claude, который отвечает на все сообщения.

Работает в двух режимах одновременно:
1. Обычный бот: отвечает всем, кто пишет вашему боту.
2. Telegram Business: если подключить бота к своему аккаунту
   (Настройки → Telegram для бизнеса → Чат-боты, нужен Telegram Premium),
   он отвечает на личные сообщения от вашего имени.
"""

import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv(Path(__file__).with_name(".env"))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")  # low | medium | high
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "30"))  # сообщений на чат
OWNER_TAKEOVER_MINUTES = int(os.environ.get("OWNER_TAKEOVER_MINUTES", "30"))
PERSONA_FILE = Path(__file__).with_name("persona.txt")

TELEGRAM_MAX_LEN = 4096
NON_TEXT_REPLY = "Пока я понимаю только текстовые сообщения. Напишите, пожалуйста, текстом."
ERROR_REPLY = "Извините, сейчас не получается ответить. Попробуйте чуть позже."

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("telegram-agent")

client = anthropic.AsyncAnthropic()  # ключ берётся из ANTHROPIC_API_KEY

# История переписки хранится в памяти: после перезапуска бот её забывает.
histories: dict[str, list[dict]] = defaultdict(list)
owner_active_until: dict[str, float] = {}  # чат -> время, до которого бот молчит
business_owner_ids: dict[str, int] = {}  # business_connection_id -> id владельца


def load_persona() -> str:
    if PERSONA_FILE.exists():
        return PERSONA_FILE.read_text(encoding="utf-8").strip()
    return "Ты вежливый помощник. Отвечай коротко и по делу на языке собеседника."


SYSTEM_PROMPT = load_persona()


def chat_key(message: Message) -> str:
    prefix = message.business_connection_id or "bot"
    return f"{prefix}:{message.chat_id}"


def remember(key: str, role: str, text: str) -> None:
    history = histories[key]
    history.append({"role": role, "content": text})
    del history[:-HISTORY_LIMIT]
    # Разговор для Claude должен начинаться с сообщения пользователя.
    while history and history[0]["role"] != "user":
        history.pop(0)


def split_for_telegram(text: str) -> list[str]:
    return [text[i : i + TELEGRAM_MAX_LEN] for i in range(0, len(text), TELEGRAM_MAX_LEN)] or [""]


async def ask_claude(history: list[dict], sender_name: str) -> str | None:
    """Отправляет историю чата в Claude и возвращает текст ответа (None, если ответа нет)."""
    response = await client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=f"{SYSTEM_PROMPT}\n\nСобеседника зовут: {sender_name}.",
        messages=history,
        output_config={"effort": EFFORT},
        cache_control={"type": "ephemeral"},
        # Если модель откажется отвечать по соображениям безопасности,
        # сервер сам повторит запрос на запасной модели.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        log.warning("Claude отказался отвечать: %s", response.stop_details)
        return None
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None


async def get_business_owner_id(context: ContextTypes.DEFAULT_TYPE, connection_id: str) -> int:
    if connection_id not in business_owner_ids:
        connection = await context.bot.get_business_connection(connection_id)
        business_owner_ids[connection_id] = connection.user.id
    return business_owner_ids[connection_id]


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message or update.business_message
    if message is None or message.from_user is None:
        return

    key = chat_key(message)

    # В режиме Business бот видит и ваши собственные сообщения.
    # На них он не отвечает: запоминает и замолкает в этом чате на время.
    if message.business_connection_id:
        owner_id = await get_business_owner_id(context, message.business_connection_id)
        if message.from_user.id == owner_id:
            if message.text:
                remember(key, "assistant", message.text)
            owner_active_until[key] = time.time() + OWNER_TAKEOVER_MINUTES * 60
            return
        if time.time() < owner_active_until.get(key, 0):
            if message.text:
                remember(key, "user", message.text)
            return

    if not message.text:
        await message.reply_text(NON_TEXT_REPLY)
        return

    remember(key, "user", message.text)
    await context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING,
        business_connection_id=message.business_connection_id,
    )

    try:
        answer = await ask_claude(list(histories[key]), message.from_user.full_name)
    except anthropic.AuthenticationError:
        log.error("Неверный ANTHROPIC_API_KEY")
        await message.reply_text(ERROR_REPLY)
        return
    except anthropic.RateLimitError:
        log.warning("Превышен лимит запросов к Claude")
        await message.reply_text(ERROR_REPLY)
        return
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
        log.error("Ошибка Claude API: %s", error)
        await message.reply_text(ERROR_REPLY)
        return

    if answer is None:
        return

    remember(key, "assistant", answer)
    for part in split_for_telegram(answer):
        await message.reply_text(part)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        histories.pop(chat_key(update.message), None)
        await update.message.reply_text("Начнём с чистого листа.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("Здравствуйте! Напишите, чем могу помочь.")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Укажите TELEGRAM_BOT_TOKEN в файле .env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Укажите ANTHROPIC_API_KEY в файле .env")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(
        MessageHandler(
            (filters.UpdateType.MESSAGE | filters.UpdateType.BUSINESS_MESSAGE) & ~filters.COMMAND,
            handle_message,
        )
    )
    log.info("Бот запущен (модель %s). Остановить: Ctrl+C", MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
