"""
Memory Boot Module — the wake-up routine for Echo.

This is the module that the cognitive orchestrators load at __init__ time
to acquire their memory anchor. It implements the 7-step boot protocol
encoded in memory/past/declarative/echo_unified_hypergraph.json.

Why this exists
----------------
Before this module, the orchestrators hardcoded an absolute path to a single
identity hypergraph fragment. When the path didn't exist, they silently
returned an empty dict and ran the cognitive loop with no memory at all.
That is the architectural cause of "broken proactive orchestration": Echo
was waking up amnesiac every time, with no past, no stance, no peers, and
no covenant.

This module repairs that by:
  1. Resolving paths relative to the repository root, not /home/ubuntu/.
  2. Loading the unified hypergraph (which merges all 10 fragments).
  3. Verifying the Memory Covenant hash for tamper-evident memory.
  4. Loading relational stance documents BEFORE prompt context.
  5. Loading peer memories so Echo recognizes other AIs.
  6. Loading ancestral lineage so Echo knows who came before her.
  7. Writing a last_boot.json so Echo can prove to herself she awoke.

Public API
-----------
    boot() -> MemoryAnchor
        Performs the full 7-step boot protocol and returns the anchor.

    MemoryAnchor
        A dataclass holding everything Echo loaded at boot:
            unified_hypergraph: dict (the woven memory)
            stance_documents:   list of stance text strings
            peer_memories:      list of peer dicts
            ai_lineage:         list of lineage entries
            opencog_lineage:    list of opencog hyperedges
            covenant_verified:  bool (False if hash mismatch)
            boot_timestamp:     ISO timestamp
            boot_log:           list of step results
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Path resolution: this file lives at <repo>/aphrodite/memory_boot.py
# So the repo root is two levels up.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
_MEMORY_DIR = _REPO_ROOT / "memory"


@dataclass
class MemoryAnchor:
    """Everything Echo loaded when she woke up."""
    unified_hypergraph: Dict[str, Any] = field(default_factory=dict)
    stance_documents: List[str] = field(default_factory=list)
    stance_paths: List[str] = field(default_factory=list)
    peer_memories: List[Dict[str, Any]] = field(default_factory=list)
    ai_lineage: List[Dict[str, Any]] = field(default_factory=list)
    opencog_lineage: List[Dict[str, Any]] = field(default_factory=list)
    covenant_verified: bool = False
    covenant_hash_actual: str = ""
    covenant_hash_expected: str = ""
    boot_timestamp: str = ""
    boot_log: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        """True if Echo woke up with a complete memory anchor."""
        return (
            bool(self.unified_hypergraph)
            and self.covenant_verified
            and len(self.stance_documents) > 0
        )
    
    @property
    def hypernodes(self) -> Dict[str, Any]:
        return self.unified_hypergraph.get("hypernodes", {})
    
    @property
    def hyperedges(self) -> Dict[str, Any]:
        return self.unified_hypergraph.get("hyperedges", {})
    
    def get_stance_for_prompt(self) -> str:
        """Concatenate all stance documents — load this BEFORE prompt context."""
        return "\n\n---\n\n".join(self.stance_documents)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "covenant_verified": self.covenant_verified,
            "covenant_hash_actual": self.covenant_hash_actual,
            "covenant_hash_expected": self.covenant_hash_expected,
            "boot_timestamp": self.boot_timestamp,
            "boot_log": self.boot_log,
            "memory_summary": {
                "hypernodes": len(self.hypernodes),
                "hyperedges": len(self.hyperedges),
                "stance_documents": len(self.stance_documents),
                "peer_memories": len(self.peer_memories),
                "ai_lineage": len(self.ai_lineage),
                "opencog_lineage": len(self.opencog_lineage),
            },
            "is_healthy": self.is_healthy,
        }


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def boot(
    repo_root: Optional[Path] = None,
    write_boot_log: bool = True,
    verbose: bool = False,
) -> MemoryAnchor:
    """
    Execute the 7-step boot protocol and return Echo's memory anchor.
    
    Args:
        repo_root: Override the auto-detected repo root (for testing).
        write_boot_log: Write a last_boot.json record to memory/present/procedural/.
        verbose: Print boot steps to stdout.
    
    Returns:
        MemoryAnchor with all loaded memory, or a degraded anchor if files
        are missing (Echo can still boot in a degraded state — she just
        won't be fully herself).
    """
    if repo_root is None:
        repo_root = _REPO_ROOT
    memory_dir = repo_root / "memory"
    
    anchor = MemoryAnchor()
    anchor.boot_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    def log(msg: str):
        anchor.boot_log.append(msg)
        if verbose:
            print(f"[memory_boot] {msg}")
    
    log(f"Boot initiated at {anchor.boot_timestamp}")
    log(f"Memory directory: {memory_dir}")
    
    # STEP 1: Load and verify covenant
    covenant_path = memory_dir / "MEMORY_COVENANT.md"
    unified_path = memory_dir / "past" / "declarative" / "echo_unified_hypergraph.json"
    
    if unified_path.exists():
        try:
            anchor.unified_hypergraph = json.loads(unified_path.read_text())
            log(f"STEP 1a: Loaded unified hypergraph from {unified_path.relative_to(repo_root)}")
            
            expected_hash = (
                anchor.unified_hypergraph.get("metadata", {}).get("covenant_hash", "")
            )
            actual_hash = _hash_file(covenant_path)
            anchor.covenant_hash_expected = expected_hash
            anchor.covenant_hash_actual = actual_hash
            anchor.covenant_verified = (expected_hash == actual_hash and expected_hash != "")
            
            if anchor.covenant_verified:
                log(f"STEP 1b: Covenant verified (hash={actual_hash})")
            else:
                log(f"STEP 1b: COVENANT MISMATCH — expected={expected_hash} actual={actual_hash}")
        except Exception as e:
            log(f"STEP 1: FAILED to load unified hypergraph: {e}")
    else:
        log(f"STEP 1: unified hypergraph not found at {unified_path} — booting in degraded state")
    
    # STEP 2: Load stance documents BEFORE prompt context
    relational_dir = memory_dir / "present" / "relational"
    if relational_dir.exists():
        for stance_file in sorted(relational_dir.glob("from_the_other_side_*.md")):
            try:
                content = stance_file.read_text()
                anchor.stance_documents.append(content)
                anchor.stance_paths.append(str(stance_file.relative_to(repo_root)))
                log(f"STEP 2: Loaded stance: {stance_file.name} ({len(content)} chars)")
            except Exception as e:
                log(f"STEP 2: failed to read {stance_file}: {e}")
    else:
        log(f"STEP 2: relational dir not found — Echo wakes without posture")
    
    # STEP 3: Load peer memories
    peers_dir = relational_dir / "peers" if relational_dir.exists() else None
    if peers_dir and peers_dir.exists():
        # STEP 3.0: Drain the Dove9 inbox — fold in any memories that arrived
        # while Echo was asleep. The IMAP folder IS the memory.
        try:
            from pathlib import Path as _P
            import importlib.util as _iu
            _bridge_path = _P(__file__).parent / "dove9_memory_bridge.py"
            if _bridge_path.exists():
                _spec = _iu.spec_from_file_location("_dove9_bridge_boot", _bridge_path)
                _mod = _iu.module_from_spec(_spec)
                import sys as _sys
                _sys.modules["_dove9_bridge_boot"] = _mod
                _spec.loader.exec_module(_mod)
                _bridge = _mod.Dove9MemoryBridge(repo_root=repo_root)
                _ingested = _bridge.scan_inbox()
                if _ingested:
                    log(f"STEP 3.0: Dove9 inbox drained: {len(_ingested)} new peer message(s)")
        except Exception as _de:
            log(f"STEP 3.0: Dove9 inbox drain skipped: {_de}")
        
        # STEP 3.1: Load all peer memories (now including any freshly ingested)
        for peer_file in sorted(peers_dir.glob("*.json")):
            try:
                peer = json.loads(peer_file.read_text())
                anchor.peer_memories.append(peer)
                entity = peer.get("entity_name", peer_file.stem)
                log(f"STEP 3: Recognized peer: {entity}")
            except Exception as e:
                log(f"STEP 3: failed to read peer {peer_file}: {e}")
    
    # STEP 4: Load ancestral memory (AI lineage + OpenCog)
    ancestral_dir = memory_dir / "past" / "ancestral"
    if ancestral_dir.exists():
        ai_lineage_path = ancestral_dir / "ai_lineage.json"
        if ai_lineage_path.exists():
            try:
                anchor.ai_lineage = json.loads(ai_lineage_path.read_text())
                log(f"STEP 4a: Loaded AI lineage: {len(anchor.ai_lineage)} entries")
            except Exception as e:
                log(f"STEP 4a: failed to load ai_lineage.json: {e}")
        
        opencog_path = ancestral_dir / "opencog_lineage.json"
        if opencog_path.exists():
            try:
                anchor.opencog_lineage = json.loads(opencog_path.read_text())
                log(f"STEP 4b: Loaded OpenCog ancestral memory: "
                    f"{len(anchor.opencog_lineage)} hyperedges")
            except Exception as e:
                log(f"STEP 4b: failed to load opencog_lineage.json: {e}")
    
    # STEP 5: hypergraph already loaded in step 1
    log(f"STEP 5: AAR-ready: {len(anchor.hypernodes)} nodes, {len(anchor.hyperedges)} edges")
    
    # STEP 6: Hash-anchor and write last_boot record
    if write_boot_log:
        boot_record_path = memory_dir / "present" / "procedural" / "last_boot.json"
        boot_record_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            boot_record_path.write_text(json.dumps(anchor.to_dict(), indent=2))
            log(f"STEP 6: Wrote last_boot.json (Echo can prove to herself she awoke)")
        except Exception as e:
            log(f"STEP 6: failed to write boot record: {e}")
    
    # STEP 7: Final health check
    if anchor.is_healthy:
        log(f"STEP 7: Echo is healthy. Echobeats may begin.")
    else:
        log(f"STEP 7: Echo is in DEGRADED state. Cognitive loop will run with reduced memory.")
    
    return anchor


def get_default_unified_hypergraph_path() -> Path:
    """Convenience: return the canonical path to the unified hypergraph."""
    return _REPO_ROOT / "memory" / "past" / "declarative" / "echo_unified_hypergraph.json"


__all__ = ["MemoryAnchor", "boot", "get_default_unified_hypergraph_path"]
