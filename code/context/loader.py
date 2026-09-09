"""
context/loader.py

Loads dataset CSV files and builds indexed lookup maps for O(1) access.
This module does not perform business logic or feature extraction; it only parses.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Optional

from .models import (
    BusinessInfo,
    DailyNotificationSummary,
    GroupInfo,
    GroupMembership,
    HistoricalMessage,
    ImageRecord,
    IncomingMessage,
    MessageEvent,
    UserBusinessRelation,
    UserProfile,
    VoiceNoteRecord,
)

logger = logging.getLogger(__name__)


def _read_csv(path: str) -> list[dict]:
    """Reads a CSV file into a list of dicts, stripping headers and values."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required dataset file: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [
            {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            for row in reader
        ]


def _parse_int(val: str, default: int = 0) -> int:
    try:
        return int(val) if val else default
    except ValueError:
        return default


def _parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except ValueError:
        return default


def _parse_bool(val: str) -> bool:
    v = val.lower()
    return v in ("true", "1", "yes", "y")


class DataContext:
    """Holds all loaded dataset tables with appropriate indexes for O(1) lookups."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

        # The core incoming messages to process
        self.messages: list[IncomingMessage] = []

        # Indexes
        self.users: dict[str, UserProfile] = {}
        self.groups: dict[str, GroupInfo] = {}
        self.group_members: dict[tuple[str, str], GroupMembership] = {}  # (group_id, user_id) -> membership
        self.businesses: dict[str, BusinessInfo] = {}
        self.user_business: dict[tuple[str, str], UserBusinessRelation] = {}  # (user_id, business_id) -> relation
        
        # Historical messages and events
        self.historical_messages: dict[str, HistoricalMessage] = {}
        self.message_events: dict[str, MessageEvent] = {}

        # History indexed by sender_user_id (to easily fetch all past messages from a sender)
        self.history_by_sender: dict[str, list[HistoricalMessage]] = {}

        # Daily summaries indexed by user_id
        self.daily_summaries: dict[str, list[DailyNotificationSummary]] = {}

        # Media maps
        self.images: dict[str, ImageRecord] = {}
        self.voice_notes: dict[str, VoiceNoteRecord] = {}

        self._load_all()

    def _load_all(self):
        """Loads all CSVs in deterministic order."""
        logger.info(f"Loading dataset from {self.dataset_dir}")

        self._load_messages()
        self._load_users()
        self._load_groups()
        self._load_group_members()
        self._load_businesses()
        self._load_user_business()
        self._load_history()
        self._load_events()
        self._load_daily_summaries()
        
        # Images and voice_notes might not exist depending on the dataset drop
        # But we should try loading them if they do
        self._load_media()
        
        logger.info(f"✓ Context loaded: {len(self.messages)} messages, {len(self.users)} users")

    def _load_messages(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "messages.csv"))
        for r in rows:
            self.messages.append(IncomingMessage(
                message_id=r["message_id"],
                user_id=r["user_id"],
                conversation_type=r["conversation_type"],
                group_id=r.get("group_id") or None,
                business_id=r.get("business_id") or None,
                sender_user_id=r.get("sender_user_id") or None,
                created_at=r["created_at"],
                message_text=r["message_text"],
                media_type=r.get("media_type", ""),
                media_id=r.get("media_id") or None,
                forwarded_count=_parse_int(r.get("forwarded_count")),
            ))

    def _load_users(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "users.csv"))
        for r in rows:
            uid = r["user_id"]
            self.users[uid] = UserProfile(
                user_id=uid,
                do_not_disturb_window=r.get("do_not_disturb_window", ""),
                messages_opened_30d=_parse_int(r.get("messages_opened_30d")),
                messages_replied_30d=_parse_int(r.get("messages_replied_30d")),
                notifications_dismissed_30d=_parse_int(r.get("notifications_dismissed_30d")),
                messages_reported_30d=_parse_int(r.get("messages_reported_30d")),
            )

    def _load_groups(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "groups.csv"))
        for r in rows:
            gid = r["group_id"]
            self.groups[gid] = GroupInfo(
                group_id=gid,
                group_name=r.get("group_name", ""),
                group_type=r.get("group_type", ""),
                member_count=_parse_int(r.get("member_count")),
                admin_count=_parse_int(r.get("admin_count")),
                created_at=r.get("created_at", ""),
                messages_30d=_parse_int(r.get("messages_30d")),
            )

    def _load_group_members(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "group_members.csv"))
        for r in rows:
            gid = r["group_id"]
            uid = r["user_id"]
            self.group_members[(gid, uid)] = GroupMembership(
                group_id=gid,
                user_id=uid,
                role=r.get("role", ""),
                joined_at=r.get("joined_at", ""),
                messages_sent_30d=_parse_int(r.get("messages_sent_30d")),
                messages_read_30d=_parse_int(r.get("messages_read_30d")),
                replies_sent_30d=_parse_int(r.get("replies_sent_30d")),
                notifications_dismissed_30d=_parse_int(r.get("notifications_dismissed_30d")),
                group_muted_by_user=_parse_bool(r.get("group_muted_by_user", "")),
            )

    def _load_businesses(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "business_accounts.csv"))
        for r in rows:
            bid = r["business_id"]
            self.businesses[bid] = BusinessInfo(
                business_id=bid,
                display_name=r.get("display_name", ""),
                brand_name=r.get("brand_name", ""),
                category=r.get("category", ""),
                verified=_parse_bool(r.get("verified", "")),
                official_domain=r.get("official_domain", ""),
                domain_used_by_sender=r.get("domain_used_by_sender", ""),
                account_age_days=_parse_int(r.get("account_age_days")),
                messages_sent_30d=_parse_int(r.get("messages_sent_30d")),
                user_reports_30d=_parse_int(r.get("user_reports_30d")),
                domain_used_by_sender_age_days=_parse_int(r.get("domain_used_by_sender_age_days")),
            )

    def _load_user_business(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "user_business_history.csv"))
        for r in rows:
            uid = r["user_id"]
            bid = r["business_id"]
            self.user_business[(uid, bid)] = UserBusinessRelation(
                user_id=uid,
                business_id=bid,
                why_user_knows_account=r.get("why_user_knows_account", ""),
                last_activity_at=r.get("last_activity_at", ""),
                allows_promotions=_parse_bool(r.get("allows_promotions", "")),
                promotions_opted_out_at=r.get("promotions_opted_out_at") or None,
                activity_count_180d=_parse_int(r.get("activity_count_180d")),
                messages_opened_30d=_parse_int(r.get("messages_opened_30d")),
                messages_dismissed_30d=_parse_int(r.get("messages_dismissed_30d")),
                messages_replied_30d=_parse_int(r.get("messages_replied_30d")),
                last_reply_at=r.get("last_reply_at") or None,
            )

    def _load_history(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "message_history.csv"))
        for r in rows:
            mid = r["message_id"]
            sender = r.get("sender_user_id") or None
            
            msg = HistoricalMessage(
                message_id=mid,
                user_id=r["user_id"],
                conversation_type=r.get("conversation_type", ""),
                group_id=r.get("group_id") or None,
                business_id=r.get("business_id") or None,
                sender_user_id=sender,
                created_at=r.get("created_at", ""),
                message_text=r.get("message_text", ""),
                media_type=r.get("media_type", ""),
                media_id=r.get("media_id") or None,
                forwarded_count=_parse_int(r.get("forwarded_count")),
            )
            self.historical_messages[mid] = msg
            if sender:
                if sender not in self.history_by_sender:
                    self.history_by_sender[sender] = []
                self.history_by_sender[sender].append(msg)

    def _load_events(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "message_events.csv"))
        for r in rows:
            mid = r["message_id"]
            rt = r.get("reaction_time_minutes")
            self.message_events[mid] = MessageEvent(
                user_id=r["user_id"],
                message_id=mid,
                message_opened=_parse_bool(r.get("message_opened", "")),
                message_replied=_parse_bool(r.get("message_replied", "")),
                reaction_time_minutes=_parse_int(rt) if rt else None,
                notification_dismissed=_parse_bool(r.get("notification_dismissed", "")),
                muted_after_message=_parse_bool(r.get("muted_after_message", "")),
                message_reported=_parse_bool(r.get("message_reported", "")),
            )

    def _load_daily_summaries(self):
        rows = _read_csv(os.path.join(self.dataset_dir, "daily_notification_summary.csv"))
        for r in rows:
            uid = r["user_id"]
            summary = DailyNotificationSummary(
                user_id=uid,
                date=r.get("date", ""),
                notifications_sent=_parse_int(r.get("notifications_sent")),
                notifications_dismissed=_parse_int(r.get("notifications_dismissed")),
            )
            if uid not in self.daily_summaries:
                self.daily_summaries[uid] = []
            self.daily_summaries[uid].append(summary)

    def _load_media(self):
        # Optional media csvs
        img_path = os.path.join(self.dataset_dir, "images.csv")
        if os.path.exists(img_path):
            rows = _read_csv(img_path)
            for r in rows:
                mid = r["image_id"]
                self.images[mid] = ImageRecord(
                    image_id=mid,
                    file_path=r.get("file_path", ""),
                )
                
        audio_path = os.path.join(self.dataset_dir, "voice_notes.csv")
        if os.path.exists(audio_path):
            rows = _read_csv(audio_path)
            for r in rows:
                mid = r["voice_note_id"]
                self.voice_notes[mid] = VoiceNoteRecord(
                    voice_note_id=mid,
                    file_path=r.get("file_path", ""),
                )
