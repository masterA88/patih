"""Chainlit entrypoint: chat UI with citation cards, confidence badge, and feedback.

Run with:
    chainlit run app/ui/chainlit_app.py -w --port 8000

build-spec Section 5.6.

Design notes (Chainlit 1.2.0):
  - Generator.answer() is synchronous; wrapped in asyncio.to_thread() so the
    event loop is not blocked during the ~8 s LLM call.
  - Streaming (yield tokens) is NOT implemented in Phase 1.  Generator returns
    a complete response.  Future: refactor Generator to yield, then use
    msg.stream_token(token).
  - cl.user_session is per-WebSocket, in-memory; lost on server restart.
    Persistent data goes to ConversationStore (SQLite).
  - Phase 1 single-tenant; no auth.  Phase 2: add cl.password_auth_callback.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Chainlit loads this file as a standalone script (not as a package member),
# so `from app.* import ...` requires the project root on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl

from app.ui.api_sidecar import register_api_routes
from app.ui.components import build_citation_cards, build_confidence_badge
from app.ui.history import ConversationStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton — shared across all sessions on this process.
# Re-initialised once per cold start; safe because Generator holds no per-user
# state.
# ---------------------------------------------------------------------------
_generator = None


def _get_generator():
    """Lazy-init Generator on first call; reuse on subsequent calls."""
    global _generator
    if _generator is None:
        from app.llm.generator import Generator
        logger.info("Chainlit: initialising Generator (cold start)…")
        _generator = Generator()
        logger.info("Chainlit: Generator ready.")
    return _generator


# ---------------------------------------------------------------------------
# REST sidecar (/health + /api/query) — mounted onto Chainlit's ASGI app.
# Chainlit exposes `chainlit.server.app` (a FastAPI instance); attaching here
# at import time means routes are live as soon as `chainlit run …` starts.
# ---------------------------------------------------------------------------
try:
    from chainlit.server import app as _chainlit_fastapi_app
    register_api_routes(_chainlit_fastapi_app, _get_generator)
    logger.info("Chainlit: /health and /api/query REST routes registered.")
except Exception as _exc:  # pragma: no cover — defensive: don't break UI if API mount fails
    logger.warning("Chainlit: failed to register REST sidecar (%s); UI still works.", _exc)


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Initialise per-session state and send welcome message."""
    # Trigger Generator init in a background thread so the welcome message
    # renders before the heavy model/index loading begins.
    asyncio.create_task(_background_init())

    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("history_store", ConversationStore())

    welcome = (
        "**Selamat datang di Asisten Regulasi Permensos 8/2023**\n"
        "(Tindak Pidana Perdagangan Orang & PMI Bermasalah)\n\n"
        "Saya dapat menjawab pertanyaan berdasarkan teks **Peraturan Menteri Sosial "
        "Nomor 8 Tahun 2023**.\n\n"
        "> ⚠️ **Disclaimer**: Asisten ini adalah alat bantu riset regulasi, "
        "**bukan nasihat hukum**. Semua jawaban perlu diverifikasi dengan sumber "
        "resmi sebelum digunakan untuk keputusan formal.\n\n"
        "---\n\n"
        "Coba tanya:\n"
        "- *Apa saja bentuk eksploitasi menurut Permensos 8/2023?*\n"
        "- *Siapa yang berwenang menangani korban TPPO?*\n"
        "- *What are the responsibilities of the social welfare ministry?*"
    )
    await cl.Message(content=welcome, author="Asisten").send()


async def _background_init():
    """Pre-warm Generator in background so first query is faster."""
    await asyncio.to_thread(_get_generator)


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    """Main QA handler.

    Flow:
      1. Show an empty "thinking" message immediately.
      2. Call Generator.answer() in a thread (sync → async bridge).
      3. Update the message with confidence badge + response text.
      4. Send citation cards as sidebar elements.
      5. Send thumbs-up / thumbs-down feedback buttons.
      6. Persist turn to SQLite.
    """
    session_id = cl.user_session.get("session_id")
    user_query = message.content.strip()

    if not user_query:
        await cl.Message(content="Mohon ketikkan pertanyaan Anda.", author="Asisten").send()
        return

    # ------------------------------------------------------------------
    # 1. Placeholder message — gives user immediate visual feedback
    # ------------------------------------------------------------------
    thinking_msg = cl.Message(content="_Sedang memproses pertanyaan Anda…_", author="Asisten")
    await thinking_msg.send()

    # ------------------------------------------------------------------
    # 2. Run Generator (sync, wrapped in thread)
    # ------------------------------------------------------------------
    generator = _get_generator()
    try:
        result = await asyncio.to_thread(generator.answer, user_query, validate=True)
    except Exception as exc:
        logger.error("Generator error for query '%s…': %s", user_query[:50], exc, exc_info=True)
        thinking_msg.content = (
            "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. "
            "Silakan coba lagi. Jika masalah berlanjut, periksa log server."
        )
        await thinking_msg.update()
        return

    # result is a GenerationResult Pydantic model; .model_dump() gives a plain dict
    result_dict = result.model_dump()

    # ------------------------------------------------------------------
    # 3. Build and update response message
    # ------------------------------------------------------------------
    badge = build_confidence_badge(result_dict.get("validation"))
    response_text = result_dict.get("response", "")

    if badge:
        full_content = f"{badge}\n\n---\n\n{response_text}"
    else:
        full_content = response_text

    thinking_msg.content = full_content
    await thinking_msg.update()

    # ------------------------------------------------------------------
    # 4. Citation cards (sidebar elements on a separate message)
    # ------------------------------------------------------------------
    cards = build_citation_cards(
        result_dict.get("retrieved_pasals", []),
        result_dict.get("validation"),
        parent_chunks=result_dict.get("parent_chunks"),
    )
    if cards:
        # Chainlit "side" display elements only become clickable when their
        # `name` is referenced verbatim in the message content. Without this,
        # the element is registered but no entry point is rendered.
        links = ", ".join(card.name for card in cards)
        await cl.Message(
            content=(
                f"**Sumber yang Dirujuk** — klik nama Pasal untuk membaca teks lengkap:\n\n"
                f"{links}"
            ),
            elements=cards,
            author="Asisten",
        ).send()

    # ------------------------------------------------------------------
    # 5. Persist conversation turn BEFORE sending feedback buttons so
    #    the row exists when add_feedback() is called.
    # ------------------------------------------------------------------
    store: ConversationStore = cl.user_session.get("history_store")
    store.append(session_id, user_query, result_dict)

    # ------------------------------------------------------------------
    # 6. Feedback buttons
    # ------------------------------------------------------------------
    actions = [
        cl.Action(name="thumbs_up",   value=session_id, label="👍 Membantu"),
        cl.Action(name="thumbs_down", value=session_id, label="👎 Perlu perbaikan"),
    ]
    await cl.Message(
        content="Apakah jawaban ini membantu?",
        actions=actions,
        author="Asisten",
    ).send()


# ---------------------------------------------------------------------------
# Feedback callbacks
# ---------------------------------------------------------------------------

@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    store: ConversationStore = cl.user_session.get("history_store")
    store.add_feedback(action.value, feedback="thumbs_up")
    await cl.Message(content="Terima kasih atas feedback-nya!", author="Asisten").send()


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    res = await cl.AskUserMessage(
        content="Boleh ceritakan apa yang perlu diperbaiki? (ketik jawaban atau tekan Enter untuk lewati)",
        timeout=120,
    ).send()

    feedback_text = (res.get("output") or "").strip() if res else ""
    store: ConversationStore = cl.user_session.get("history_store")
    store.add_feedback(action.value, feedback="thumbs_down", feedback_text=feedback_text)
    await cl.Message(
        content="Terima kasih. Masukan Anda akan digunakan untuk meningkatkan kualitas jawaban.",
        author="Asisten",
    ).send()
