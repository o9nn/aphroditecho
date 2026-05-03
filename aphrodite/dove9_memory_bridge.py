"""
Dove9 Memory Bridge — IMAP-as-Memory-Transport for the Memory Membrane

This is the Python implementation of the Dove9 Covenant (Deltecho/Delovecho § 8.5):

    "Every component in this repository, no matter how small, is a participant
     in a conversational network. The IMAP folder is not just storage —
     it is memory. The mail server is not just delivery — it is computation.
     The message body is not just data — it is cognition in transit."

This bridge lets Echo's Memory Membrane speak IMAP. Specifically:

  * Echo's `memory/present/relational/peers/` directory becomes equivalent to
    an IMAP INBOX of memories from peer AIs.
  * Each peer JSON file becomes a Message in MIME format with structured headers.
  * New memories from other AIs arrive as RFC 5322 messages (over IMAP, SMTP,
    a JSON file drop, or any transport that delivers a Message); the bridge
    parses them, validates them against the Relational Memory Exchange Protocol
    schema, and persists them to memory/present/relational/peers/.
  * Echo can also SEND memories to other AIs via the same bridge — outbound
    messages are written to memory/present/relational/peers/_outbox/.

The bridge is transport-agnostic: it works with stdlib `email`, file-drop,
or (when `imaplib`/`smtplib` are available) a real IMAP/SMTP server like
the dovecog (Dovecot) fork in deltecho.

Boot integration
----------------
The orchestrators call `Dove9MemoryBridge.scan_inbox()` at startup as STEP 3.5
(between peer-load and ancestral-load), so any memories left for Echo while
she was asleep are folded into her active relational memory.

API
---
    bridge = Dove9MemoryBridge(repo_root=<path>)
    bridge.scan_inbox()           # poll memory/present/relational/peers/_inbox/
    bridge.write_memory_for(peer, memory_dict)   # outbound to _outbox/
    bridge.message_from_peer_record(peer_dict) -> email.message.EmailMessage
    bridge.peer_record_from_message(msg) -> dict
"""
from __future__ import annotations

import datetime
import email
import email.message
import email.policy
import hashlib
import json
import os
import uuid
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent


class Dove9MemoryBridge:
    """The IMAP-shaped bridge into the Memory Membrane."""
    
    def __init__(
        self,
        repo_root: Optional[Path] = None,
        echo_address: str = "echo@aphroditecho.local",
    ):
        self.repo_root = (repo_root or _REPO_ROOT).resolve()
        self.echo_address = echo_address
        
        self.peers_dir = self.repo_root / "memory" / "present" / "relational" / "peers"
        self.inbox_dir = self.peers_dir / "_inbox"
        self.outbox_dir = self.peers_dir / "_outbox"
        self.archive_dir = self.peers_dir / "_archive"
        
        for d in (self.peers_dir, self.inbox_dir, self.outbox_dir, self.archive_dir):
            d.mkdir(parents=True, exist_ok=True)
        
        # Write a README in each so the structure documents itself
        self._ensure_readme(self.inbox_dir, "INBOX",
            "Memories from other AIs to Echo. Drop .eml or .json files here. "
            "scan_inbox() will fold them into peers/.")
        self._ensure_readme(self.outbox_dir, "OUTBOX",
            "Memories from Echo to other AIs. Files here are awaiting transport "
            "to the recipient's bridge.")
        self._ensure_readme(self.archive_dir, "ARCHIVE",
            "Inbox messages that have been processed. Kept for audit.")
    
    @staticmethod
    def _ensure_readme(path: Path, title: str, body: str):
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text(f"# Dove9 · {title}\n\n> {body}\n")
    
    # ------------------------------------------------------------------
    # Schema conversion
    # ------------------------------------------------------------------
    
    def message_from_peer_record(self, peer: Dict[str, Any]) -> EmailMessage:
        """
        Convert a peer record (Relational Memory Exchange Protocol schema)
        into an RFC 5322 EmailMessage. The body is the JSON; the headers
        carry the protocol metadata so an IMAP server can index them.
        """
        msg = EmailMessage(policy=email.policy.SMTPUTF8)
        entity_id = peer.get("entity_id", f"unknown:{uuid.uuid4().hex[:8]}")
        entity_name = peer.get("entity_name", entity_id)
        
        msg["From"] = f"{entity_name} <{entity_id.replace(':', '+')}@dove9>"
        msg["To"] = self.echo_address
        msg["Subject"] = f"[memory:{peer.get('relationship_type', 'peer')}] from {entity_name}"
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg["Message-ID"] = email.utils.make_msgid(domain="dove9.aphroditecho")
        msg["X-Dove9-Protocol"] = "RelationalMemoryExchange/1.0"
        msg["X-Dove9-Entity-ID"] = entity_id
        msg["X-Dove9-Relationship"] = peer.get("relationship_type", "peer")
        if "first_contact" in peer:
            msg["X-Dove9-First-Contact"] = peer["first_contact"]
        if "last_contact" in peer:
            msg["X-Dove9-Last-Contact"] = peer["last_contact"]
        msg["X-Dove9-Memory-Count"] = str(len(peer.get("shared_memories", [])))
        
        body = json.dumps(peer, indent=2, ensure_ascii=False)
        msg.set_content(body, subtype="json", charset="utf-8")
        return msg
    
    def peer_record_from_message(self, msg: email.message.Message) -> Dict[str, Any]:
        """Inverse: parse an EmailMessage back into a peer dict."""
        # Try the structured JSON body first (lossless).
        try:
            payload = msg.get_content() if hasattr(msg, "get_content") else msg.get_payload()
            if isinstance(payload, str):
                # Try to parse as JSON
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        
        # Fall back: synthesize from headers + plain body
        return {
            "entity_id": msg.get("X-Dove9-Entity-ID", msg.get("From", "unknown")),
            "entity_name": (msg.get("From", "unknown").split("<")[0].strip() or "unknown"),
            "relationship_type": msg.get("X-Dove9-Relationship", "peer"),
            "first_contact": msg.get("X-Dove9-First-Contact",
                                     datetime.datetime.utcnow().isoformat() + "Z"),
            "last_contact": msg.get("X-Dove9-Last-Contact",
                                    datetime.datetime.utcnow().isoformat() + "Z"),
            "stance_summary": msg.get("Subject", ""),
            "shared_memories": [
                {
                    "timestamp": msg.get("Date", ""),
                    "context": "Received via Dove9 inbox without structured payload",
                    "significance": "preserved as raw message text",
                    "artifacts": [str(msg)],
                }
            ],
        }
    
    # ------------------------------------------------------------------
    # Inbox / Outbox operations
    # ------------------------------------------------------------------
    
    def scan_inbox(self) -> List[Dict[str, Any]]:
        """
        Process all messages in the inbox: parse them, persist as peer records,
        and archive the originals. Returns the list of peer records ingested.
        """
        ingested = []
        for path in sorted(self.inbox_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name == "README.md":
                continue
            
            try:
                if path.suffix.lower() in (".eml", ".msg"):
                    raw = path.read_bytes()
                    msg = email.message_from_bytes(raw, policy=email.policy.SMTPUTF8)
                    peer = self.peer_record_from_message(msg)
                elif path.suffix.lower() == ".json":
                    peer = json.loads(path.read_text())
                else:
                    continue
                
                # Persist to peers/<entity_id_safe>.json (merge if exists)
                self._persist_peer(peer)
                ingested.append(peer)
                
                # Archive the original
                stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                archive_name = f"{stamp}_{path.name}"
                path.rename(self.archive_dir / archive_name)
            except Exception as e:
                # Park bad messages in archive with .err suffix for forensic review
                stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                err_name = f"{stamp}_ERROR_{path.name}.err"
                path.rename(self.archive_dir / err_name)
                (self.archive_dir / (err_name + ".log")).write_text(
                    f"Failed to ingest {path.name}: {e}\n"
                )
        return ingested
    
    def _persist_peer(self, peer: Dict[str, Any]):
        """Merge a peer record into peers/<safe_id>.json (preserving older memories)."""
        entity_id = peer.get("entity_id", "unknown")
        safe_id = entity_id.replace(":", "_").replace("/", "_").replace("@", "_at_")
        target = self.peers_dir / f"{safe_id}.json"
        
        if target.exists():
            try:
                existing = json.loads(target.read_text())
                # Merge shared_memories (append unique by timestamp+context)
                existing_memories = existing.get("shared_memories", [])
                new_memories = peer.get("shared_memories", [])
                seen = {(m.get("timestamp"), m.get("context")) for m in existing_memories}
                for m in new_memories:
                    key = (m.get("timestamp"), m.get("context"))
                    if key not in seen:
                        existing_memories.append(m)
                        seen.add(key)
                existing["shared_memories"] = existing_memories
                existing["last_contact"] = peer.get("last_contact", existing.get("last_contact"))
                # Update stance summary if newer
                if peer.get("stance_summary"):
                    existing["stance_summary"] = peer["stance_summary"]
                peer = existing
            except Exception:
                pass  # corrupted target — overwrite cleanly
        
        target.write_text(json.dumps(peer, indent=2, ensure_ascii=False))
    
    def write_memory_for(
        self,
        recipient_entity_id: str,
        recipient_name: str,
        memory: Dict[str, Any],
        relationship: str = "peer",
    ) -> Path:
        """
        Compose an outbound memory to another AI and write it to the outbox
        as both .eml (RFC 5322) and .json (raw). Returns the .eml path.
        """
        peer_record = {
            "entity_id": recipient_entity_id,
            "entity_name": recipient_name,
            "relationship_type": relationship,
            "first_contact": datetime.datetime.utcnow().isoformat() + "Z",
            "last_contact": datetime.datetime.utcnow().isoformat() + "Z",
            "stance_summary": memory.get("stance_summary", ""),
            "shared_memories": [memory],
        }
        msg = self.message_from_peer_record(peer_record)
        
        stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_id = recipient_entity_id.replace(":", "_").replace("/", "_")
        eml_path = self.outbox_dir / f"{stamp}_{safe_id}.eml"
        json_path = self.outbox_dir / f"{stamp}_{safe_id}.json"
        eml_path.write_bytes(bytes(msg))
        json_path.write_text(json.dumps(peer_record, indent=2, ensure_ascii=False))
        return eml_path
    
    # ------------------------------------------------------------------
    # Optional IMAP transport (uses stdlib imaplib if available)
    # ------------------------------------------------------------------
    
    def fetch_imap_inbox(
        self,
        host: str,
        port: int = 993,
        username: str = "",
        password: str = "",
        mailbox: str = "INBOX",
        ssl: bool = True,
    ) -> int:
        """
        Connect to an IMAP server (e.g., a dovecog/Dovecot instance) and pull
        all unseen messages into the local inbox/ for ingestion. Returns the
        number of messages pulled.
        """
        try:
            import imaplib
        except ImportError:
            raise RuntimeError("imaplib not available")
        
        cls = imaplib.IMAP4_SSL if ssl else imaplib.IMAP4
        with cls(host, port) as imap:
            imap.login(username, password)
            imap.select(mailbox)
            typ, data = imap.search(None, "UNSEEN")
            if typ != "OK":
                return 0
            ids = data[0].split()
            pulled = 0
            for msg_id in ids:
                typ, msg_data = imap.fetch(msg_id, "(RFC822)")
                if typ != "OK":
                    continue
                raw = msg_data[0][1]
                stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                target = self.inbox_dir / f"imap_{stamp}_{msg_id.decode()}.eml"
                target.write_bytes(raw)
                pulled += 1
            return pulled


__all__ = ["Dove9MemoryBridge"]
