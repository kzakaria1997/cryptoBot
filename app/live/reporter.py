from __future__ import annotations

import json
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


TRADES_FILE = "app/live/paper_trades.json"


def load_trades() -> list[dict]:
    if not os.path.exists(TRADES_FILE):
        return []
    with open(TRADES_FILE, "r") as f:
        return json.load(f)


def save_trade(trade: dict) -> None:
    trades = load_trades()
    trades.append(trade)
    os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def build_report(trades: list[dict], period_days: int = 7) -> str:
    if not trades:
        return "Aucun trade durant cette période."

    wins   = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    total_pnl   = sum(t.get("pnl_usd", 0) for t in trades)
    gross_p     = sum(t["pnl_usd"] for t in wins)
    gross_l     = abs(sum(t["pnl_usd"] for t in losses))
    pf          = gross_p / gross_l if gross_l > 0 else 0.0
    win_rate    = len(wins) / len(trades) * 100 if trades else 0.0

    lines = [
        f"=== RAPPORT PAPER TRADING — {period_days} JOURS ===",
        f"Période   : {trades[0].get('entry_time','?')[:10]} → {trades[-1].get('exit_time','?')[:10]}",
        f"",
        f"Trades    : {len(trades)}  (W:{len(wins)} / L:{len(losses)})",
        f"Win rate  : {win_rate:.1f}%",
        f"PnL total : ${total_pnl:+.2f}",
        f"Profit F. : {pf:.3f}",
        f"",
        f"--- Détail des trades ---",
    ]

    for i, t in enumerate(trades, 1):
        pnl   = t.get("pnl_usd", 0)
        sign  = "✅" if pnl > 0 else "❌"
        lines.append(
            f"{sign} #{i:02d} {t.get('pair','?')} [{t.get('strategy','?')}]"
            f"  Entry ${t.get('entry_price',0):,.2f} → Exit ${t.get('exit_price',0):,.2f}"
            f"  PnL ${pnl:+.2f}  ({t.get('exit_reason','?')})"
            f"  {t.get('entry_time','?')[:16]}"
        )

    lines += [
        f"",
        f"--- Verdict ---",
    ]
    if pf >= 1.5 and len(trades) >= 3:
        lines.append("🟢 Système PROFITABLE sur la semaine — envisage le passage en live.")
    elif pf >= 1.0:
        lines.append("🟡 Système légèrement profitable — attends encore 1 semaine avant le live.")
    else:
        lines.append("🔴 Système non profitable cette semaine — NE PAS passer en live encore.")

    lines.append(f"\nEnvoie ce rapport à Claude pour analyse approfondie.")

    return "\n".join(lines)


def send_email(report: str, to_email: str, smtp_email: str, smtp_password: str) -> bool:
    subject = f"[CryptoBot] Rapport hebdomadaire paper trading — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_email
    msg["To"]      = to_email
    msg.attach(MIMEText(report, "plain"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[Reporter] Erreur envoi email: {e}", flush=True)
        return False
