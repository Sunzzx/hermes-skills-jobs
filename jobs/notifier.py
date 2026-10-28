#!/usr/bin/env python3
"""Unified Notification Dispatcher for Hermes.

Routes job summaries and exam drills strictly to:
- WhatsApp (via local bridge at http://127.0.0.1:3000)
- Discord (via bot token directly to #ai-chat)
Suppresses all desktop/OS-level banners per operator directive.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
ENV_FILE = HOME / ".hermes" / ".env"

# Load environment
ENV = {}
if ENV_FILE.exists():
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                ENV[k.strip()] = v.strip().strip("\"'")

WHATSAPP_URL = "http://127.0.0.1:3000/send"
WHATSAPP_CHAT_ID = os.environ.get("WHATSAPP_HOME_CHANNEL") or ENV.get("WHATSAPP_HOME_CHANNEL") or "64974567268429@lid"
DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN") or ENV.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL = os.environ.get("DISCORD_HOME_CHANNEL") or ENV.get("DISCORD_HOME_CHANNEL") or "1118562963960381490"


def send_whatsapp(message: str, chat_id: str = WHATSAPP_CHAT_ID) -> bool:
    try:
        payload = json.dumps({"chatId": chat_id, "message": message}).encode("utf-8")
        req = urllib.request.Request(
            WHATSAPP_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"WhatsApp dispatch note: {e}")
        return False


def send_discord(message: str, channel_id: str = DISCORD_CHANNEL) -> bool:
    try:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        # Discord has a 2000 character limit per message
        chunks = [message[i:i + 1950] for i in range(0, len(message), 1950)]
        for chunk in chunks:
            payload = json.dumps({"content": chunk}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bot {DISCORD_TOKEN}",
                    "Content-Type": "application/json",
                    "User-Agent": "DiscordBot (Hermes, 1.0)"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass
        return True
    except Exception as e:
        print(f"Discord dispatch note: {e}")
        return False


def notify_job_summary(job_name: str, summary: str, extra_dest: str = "whatsapp"):
    """Send job summary to WhatsApp (and optionally Discord)."""
    text = f"🤖 *Hermes Job Completed: {job_name}*\n\n{summary}\n\n_— Hermes Autonomous Daemon_"
    send_whatsapp(text)
    if extra_dest == "both":
        # Format for Discord markdown
        disc_text = f"🤖 **Hermes Job Completed: {job_name}**\n\n{summary}\n\n*— Hermes Autonomous Daemon*"
        send_discord(disc_text)


def notify_exam_drill(exam: str, topic: str, fact: str, question: str, answer: str, explanation: str):
    """Deliver exam question and facts to BOTH WhatsApp and Discord."""
    # 1. WhatsApp Format
    wa_msg = (
        f"🏛️ *HERMES OFFICER EXAM DRILL (10m)*\n"
        f"📚 *Exam:* {exam}\n"
        f"🎯 *Topic:* {topic}\n\n"
        f"📌 *HIGH-YIELD FACT:*\n{fact}\n\n"
        f"🧠 *CHALLENGE QUESTION:*\n{question}\n\n"
        f"💡 *CORRECT ANSWER:*\n{answer}\n\n"
        f"📖 *EXPLANATION:*\n{explanation}\n\n"
        f"_— Focus: UPSC CSE & CAT | Next drill in 10m_"
    )
    send_whatsapp(wa_msg)

    # 2. Discord Format
    disc_msg = (
        f"🏛️ **HERMES OFFICER EXAM DRILL**\n"
        f"📚 **Exam:** {exam} | 🎯 **Topic:** `{topic}`\n\n"
        f"📌 **High-Yield Fact:**\n> {fact}\n\n"
        f"🧠 **Challenge Question:**\n```text\n{question}\n```\n\n"
        f"||💡 **Answer:** {answer}||\n\n"
        f"📖 **Explanation:**\n{explanation}\n\n"
        f"*— Next drill in 10 mins*"
    )
    send_discord(disc_msg)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "test":
        msg = sys.argv[2]
        print("Testing WhatsApp...")
        print("WA success:", send_whatsapp(msg))
        print("Testing Discord...")
        print("Discord success:", send_discord(msg))
    else:
        print("Hermes Notifier Module Ready.")
