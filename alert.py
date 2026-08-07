# alert.py — Discord + Telegram Alerter (sends to both in parallel)
import os, time, logging, threading, queue, requests
log = logging.getLogger(__name__)

class TelegramAlerter:  # keep same class name — no other file changes needed
    def __init__(self, config):
        self.webhook       = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.tg_token      = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.tg_chat_id    = os.getenv("TELEGRAM_CHAT_ID", "")

        self._discord_on = bool(self.webhook and "discord.com" in self.webhook)
        self._telegram_on = bool(self.tg_token and self.tg_chat_id)

        self._queue   = queue.Queue(maxsize=50)
        self._enabled = self._discord_on or self._telegram_on
        if self._enabled:
            threading.Thread(target=self._worker, daemon=True).start()
        print(f"[Alert] {'✅' if self._discord_on else '⚠️ '} Discord {'ready' if self._discord_on else 'not set — disabled'}.")
        print(f"[Alert] {'✅' if self._telegram_on else '⚠️ '} Telegram {'ready' if self._telegram_on else 'not set — disabled'}.")

    def send(self, screenshot_path, event):
        if not self._enabled: return
        try: self._queue.put_nowait((screenshot_path, event))
        except queue.Full: log.warning("[Alert] Queue full.")

    def stop(self):
        if self._enabled: self._queue.join()

    def _worker(self):
        while True:
            try: item = self._queue.get(timeout=1)
            except queue.Empty: continue
            try: self._send(*item)
            finally: self._queue.task_done()

    def _send(self, screenshot_path, event):
        # Sent independently — a Discord failure must not block the Telegram
        # send (or vice versa), so each gets its own try/except.
        if self._discord_on:
            try: self._send_discord(screenshot_path, event)
            except Exception as e: log.error(f"[Alert] Discord {e}")
        if self._telegram_on:
            try: self._send_telegram(screenshot_path, event)
            except Exception as e: log.error(f"[Alert] Telegram {e}")

    def _send_discord(self, screenshot_path, event):
        ts, ids, flags = self._fields(event)
        zone    = f"\n📍 Zone: {event.zone_id}" if event.zone_id else ""
        content = (f"🚨 **COLLISION DETECTED**\n"
                   f"🕐 {ts}\n"
                   f"🚗 IDs: {ids}\n"
                   f"⚠️ Triggers: {flags}\n"
                   f"📊 Severity: **{event.severity}** ({event.severity_score:.2f})"
                   f"{zone}")
        with open(screenshot_path, "rb") as f:
            resp = requests.post(
                self.webhook,
                data={"content": content},
                files={"file": ("alert.jpg", f, "image/jpeg")},
                timeout=15,
            )
        if resp.status_code in (200, 204):
            print(f"[Alert] ✅ Discord alert sent → IDs {ids}")
        else:
            log.error(f"[Alert] ❌ Discord {resp.status_code}: {resp.text[:200]}")

    def _send_telegram(self, screenshot_path, event):
        ts, ids, flags = self._fields(event)
        zone    = f"\n📍 Zone: {event.zone_id}" if event.zone_id else ""
        caption = (f"🚨 <b>COLLISION DETECTED</b>\n"
                   f"🕐 {ts}\n"
                   f"🚗 IDs: {ids}\n"
                   f"⚠️ Triggers: {flags}\n"
                   f"📊 Severity: <b>{event.severity}</b> ({event.severity_score:.2f})"
                   f"{zone}")
        url = f"https://api.telegram.org/bot{self.tg_token}/sendPhoto"
        with open(screenshot_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": self.tg_chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": ("alert.jpg", f, "image/jpeg")},
                timeout=15,
            )
        if resp.status_code == 200:
            print(f"[Alert] ✅ Telegram alert sent → IDs {ids}")
        else:
            log.error(f"[Alert] ❌ Telegram {resp.status_code}: {resp.text[:200]}")

    @staticmethod
    def _fields(event):
        ts    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))
        ids   = ", ".join(str(i) for i in event.vehicle_ids)
        flags = " | ".join(event.trigger_flags)
        return ts, ids, flags
