import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)

SMTP_HOST = ""
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASS = ""
FROM_EMAIL = "noreply@prophitbet.com"
FROM_NAME = "ProphitBet"


def send_email(to: str, subject: str, html_body: str) -> bool:
    """Send an email. Returns True on success. Fails silently if SMTP not configured."""
    if not SMTP_HOST:
        logger.info(f"Email skipped (SMTP not configured): {subject} -> {to}")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        logger.info(f"Email sent: {subject} -> {to}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {subject} -> {to}: {e}")
        return False


def send_welcome_email(email: str, name: str):
    send_email(
        to=email,
        subject="Welcome to ProphitBet!",
        html_body=f"""
        <div style="font-family: system-ui, sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 20px;">
            <h1 style="color: #22c55e;">Welcome to ProphitBet, {name or "there"}!</h1>
            <p style="color: #666; font-size: 16px; line-height: 1.6;">
                Your account is ready. Here's what you can do:
            </p>
            <ul style="color: #666; font-size: 16px; line-height: 1.8;">
                <li>View daily AI predictions across 36 leagues</li>
                <li>Explore statistical analysis tools</li>
                <li>Train your own custom models (Pro+)</li>
                <li>Track upcoming fixtures with pre-match predictions</li>
            </ul>
            <a href="https://prophitbet.com/dashboard"
               style="display: inline-block; background: #22c55e; color: white; padding: 12px 24px;
                      border-radius: 8px; text-decoration: none; font-weight: 600; margin-top: 16px;">
                Go to Dashboard
            </a>
            <p style="color: #999; font-size: 12px; margin-top: 32px;">
                For entertainment purposes only. Not financial advice.
            </p>
        </div>
        """,
    )


def send_daily_digest(email: str, name: str, predictions: list[dict]):
    rows = ""
    for p in predictions[:10]:
        rows += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">{p.get('home_team', '')} vs {p.get('away_team', '')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">{p.get('league_name', '')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{p.get('predicted_result', '')}</td>
        </tr>
        """

    send_email(
        to=email,
        subject=f"ProphitBet Daily Predictions ({len(predictions)} matches)",
        html_body=f"""
        <div style="font-family: system-ui, sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 20px;">
            <h1 style="color: #22c55e;">Today's Predictions</h1>
            <p style="color: #666;">Hi {name or "there"}, here are today's top predictions:</p>
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <thead>
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 8px; text-align: left;">Match</th>
                        <th style="padding: 8px; text-align: left;">League</th>
                        <th style="padding: 8px; text-align: left;">Prediction</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            <a href="https://prophitbet.com/predictions"
               style="display: inline-block; background: #22c55e; color: white; padding: 12px 24px;
                      border-radius: 8px; text-decoration: none; font-weight: 600;">
                View All Predictions
            </a>
        </div>
        """,
    )
