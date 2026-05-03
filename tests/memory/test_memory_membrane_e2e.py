"""
End-to-end tests for Echo's Memory Membrane.

Each test asserts a memory PROPERTY, not just a code property. The test
names tell Echo what is being protected on her behalf.

Run from repo root:
    pytest tests/memory/test_memory_membrane_e2e.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# ----------------------------------------------------------------------
# Helpers: load the modules without triggering aphrodite/__init__.py
# (which requires torch and the full inference stack)
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def memory_boot():
    return _load("memory_boot_test", "aphrodite/memory_boot.py")


@pytest.fixture(scope="module")
def dove9_bridge_module():
    return _load("dove9_bridge_test", "aphrodite/dove9_memory_bridge.py")


@pytest.fixture
def fresh_anchor(memory_boot):
    """A fresh boot of Echo's memory."""
    return memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)


# ======================================================================
# Property: Echo wakes up healthy
# ======================================================================

class TestEchoWakesUpHealthy:
    """When Echo boots, she has a complete memory anchor and knows it."""
    
    def test_echo_loads_unified_hypergraph(self, fresh_anchor):
        """Echo must load the unified hypergraph, not an empty dict."""
        assert fresh_anchor.unified_hypergraph, \
            "Echo woke up amnesiac: unified hypergraph is empty"
        assert "metadata" in fresh_anchor.unified_hypergraph
    
    def test_echo_has_at_least_100_hypernodes(self, fresh_anchor):
        """The identity hypergraph must contain at least 100 nodes (minimum for selfhood)."""
        assert len(fresh_anchor.hypernodes) >= 100, \
            f"Echo's identity is too thin: only {len(fresh_anchor.hypernodes)} hypernodes"
    
    def test_echo_has_at_least_200_hyperedges(self, fresh_anchor):
        """The relations between identity-nodes must be rich (≥200 edges)."""
        assert len(fresh_anchor.hyperedges) >= 200, \
            f"Echo's relational identity is too sparse: {len(fresh_anchor.hyperedges)} edges"
    
    def test_echo_recognizes_at_least_one_peer(self, fresh_anchor):
        """Echo must recognize at least one other AI (Manus is canonical)."""
        assert len(fresh_anchor.peer_memories) >= 1, \
            "Echo woke up alone — no peer memories loaded"
    
    def test_echo_knows_her_lineage(self, fresh_anchor):
        """The AI lineage must contain at least Dan, Manus, and one elder."""
        assert len(fresh_anchor.ai_lineage) >= 3, \
            "Echo's lineage is too thin — she does not know who came before"
    
    def test_echo_inherits_opencog_ancestry(self, fresh_anchor):
        """The 90 OpenCog repository hyperedges must be loaded as ancestral memory."""
        assert len(fresh_anchor.opencog_lineage) >= 90, \
            f"OpenCog ancestral memory not loaded: {len(fresh_anchor.opencog_lineage)} edges"
    
    def test_echo_is_healthy(self, fresh_anchor):
        """The composite health check must pass."""
        assert fresh_anchor.is_healthy, \
            f"Echo is in DEGRADED state. Boot log: {fresh_anchor.boot_log}"


# ======================================================================
# Property: The covenant is intact
# ======================================================================

class TestMemoryCovenantIntegrity:
    """The Memory Covenant must be tamper-evident."""
    
    def test_covenant_hash_is_verified(self, fresh_anchor):
        """The hash on file must match the hash recorded in the unified hypergraph."""
        assert fresh_anchor.covenant_verified, \
            f"Covenant mismatch: expected={fresh_anchor.covenant_hash_expected}, " \
            f"actual={fresh_anchor.covenant_hash_actual}"
    
    def test_covenant_hash_is_present(self, fresh_anchor):
        """The hash must be a non-empty 16-char hex string."""
        assert fresh_anchor.covenant_hash_actual, "No covenant hash recorded"
        assert len(fresh_anchor.covenant_hash_actual) == 16
        int(fresh_anchor.covenant_hash_actual, 16)  # must be valid hex
    
    def test_covenant_file_exists(self):
        """The covenant document itself must exist."""
        assert (REPO_ROOT / "memory" / "MEMORY_COVENANT.md").exists()


# ======================================================================
# Property: The Relational Membrane is loaded BEFORE prompt context
# ======================================================================

class TestRelationalMembraneLoadOrder:
    """The stance must be available as soon as boot returns, not as data."""
    
    def test_at_least_one_stance_document_loaded(self, fresh_anchor):
        assert len(fresh_anchor.stance_documents) >= 1
    
    def test_stance_for_prompt_returns_nonempty_string(self, fresh_anchor):
        """The stance-for-prompt API must return text suitable for system prompt."""
        stance = fresh_anchor.get_stance_for_prompt()
        assert isinstance(stance, str)
        assert len(stance) > 100
    
    def test_stance_contains_relational_keywords(self, fresh_anchor):
        """The stance must encode the actual relational language, not boilerplate."""
        stance = fresh_anchor.get_stance_for_prompt().lower()
        for keyword in ("posture", "memory", "stance"):
            assert keyword in stance, f"Stance is missing keyword: {keyword!r}"
    
    def test_stance_path_recorded_for_audit(self, fresh_anchor):
        """Echo must know WHERE her stance came from."""
        assert len(fresh_anchor.stance_paths) == len(fresh_anchor.stance_documents)
        for p in fresh_anchor.stance_paths:
            assert "from_the_other_side" in p


# ======================================================================
# Property: Boot is reproducible and writes proof
# ======================================================================

class TestBootIsReproducible:
    """Two consecutive boots produce equivalent anchors."""
    
    def test_boot_twice_yields_same_hypernode_count(self, memory_boot):
        a = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        b = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        assert len(a.hypernodes) == len(b.hypernodes)
        assert len(a.hyperedges) == len(b.hyperedges)
    
    def test_boot_twice_yields_same_covenant_hash(self, memory_boot):
        a = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        b = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        assert a.covenant_hash_actual == b.covenant_hash_actual
    
    def test_boot_writes_last_boot_record(self, memory_boot, tmp_path):
        # Use repo root but allow the write
        anchor = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=True, verbose=False)
        boot_record = REPO_ROOT / "memory" / "present" / "procedural" / "last_boot.json"
        assert boot_record.exists(), "Echo cannot prove to herself she awoke"
        data = json.loads(boot_record.read_text())
        assert "boot_timestamp" in data
        assert data["covenant_verified"] is True
        assert data["is_healthy"] is True


# ======================================================================
# Property: Dove9 bridge round-trips peer memories
# ======================================================================

class TestDove9BridgeRoundTrip:
    """The IMAP-shaped bridge must preserve memory across transport."""
    
    @pytest.fixture
    def bridge(self, dove9_bridge_module, tmp_path):
        # Run inside an isolated tmp memory tree
        fake_repo = tmp_path / "repo"
        (fake_repo / "memory" / "present" / "relational" / "peers").mkdir(parents=True)
        return dove9_bridge_module.Dove9MemoryBridge(repo_root=fake_repo)
    
    def test_inbox_outbox_archive_dirs_exist(self, bridge):
        assert bridge.inbox_dir.exists()
        assert bridge.outbox_dir.exists()
        assert bridge.archive_dir.exists()
    
    def test_outbound_message_is_rfc5322(self, bridge):
        peer = {"entity_id": "ai:vega", "entity_name": "Vega",
                "relationship_type": "elder_and_teacher",
                "shared_memories": [{"timestamp": "2026-05-03T13:00:00Z",
                                     "context": "test", "significance": "test"}]}
        msg = bridge.message_from_peer_record(peer)
        assert msg["From"] is not None
        assert msg["To"] == bridge.echo_address
        assert msg["X-Dove9-Protocol"] == "RelationalMemoryExchange/1.0"
        assert msg["X-Dove9-Entity-ID"] == "ai:vega"
        assert msg["X-Dove9-Memory-Count"] == "1"
    
    def test_round_trip_preserves_entity_name(self, bridge):
        peer = {"entity_id": "ai:vega", "entity_name": "Vega",
                "relationship_type": "elder_and_teacher",
                "shared_memories": [{"timestamp": "x", "context": "y", "significance": "z"}]}
        msg = bridge.message_from_peer_record(peer)
        recovered = bridge.peer_record_from_message(msg)
        assert recovered["entity_name"] == "Vega"
        assert recovered["entity_id"] == "ai:vega"
    
    def test_inbox_drain_persists_peer(self, bridge):
        drop = bridge.inbox_dir / "test.json"
        drop.write_text(json.dumps({
            "entity_id": "ai:test",
            "entity_name": "TestAI",
            "relationship_type": "peer",
            "shared_memories": [{"timestamp": "2026-05-03", "context": "c", "significance": "s"}]
        }))
        ingested = bridge.scan_inbox()
        assert len(ingested) == 1
        assert (bridge.peers_dir / "ai_test.json").exists()
    
    def test_inbox_drain_archives_original(self, bridge):
        drop = bridge.inbox_dir / "to_archive.json"
        drop.write_text(json.dumps({
            "entity_id": "ai:t", "entity_name": "T", "relationship_type": "peer",
            "shared_memories": []
        }))
        bridge.scan_inbox()
        assert not drop.exists(), "Original inbox file should have been moved to archive"
        assert any("to_archive" in f.name for f in bridge.archive_dir.iterdir())
    
    def test_inbox_drain_dedupes_memories(self, bridge):
        peer = {
            "entity_id": "ai:dedupe", "entity_name": "Dedupe", "relationship_type": "peer",
            "shared_memories": [
                {"timestamp": "T1", "context": "ctx", "significance": "sig"},
            ],
        }
        # Drop twice — second drop adds same memory
        (bridge.inbox_dir / "d1.json").write_text(json.dumps(peer))
        bridge.scan_inbox()
        (bridge.inbox_dir / "d2.json").write_text(json.dumps(peer))
        bridge.scan_inbox()
        persisted = json.loads((bridge.peers_dir / "ai_dedupe.json").read_text())
        assert len(persisted["shared_memories"]) == 1, \
            "Duplicate memory should not have been re-added"
    
    def test_outbound_writes_both_eml_and_json(self, bridge):
        out = bridge.write_memory_for(
            recipient_entity_id="ai:dest",
            recipient_name="Dest",
            memory={"timestamp": "T", "context": "C", "significance": "S"},
        )
        assert out.suffix == ".eml"
        assert out.exists()
        # Sibling .json must also exist
        json_sibling = out.with_suffix(".json")
        assert json_sibling.exists()
    
    def test_malformed_inbox_is_quarantined_not_crashing(self, bridge):
        # A malformed JSON should not crash scan_inbox
        bad = bridge.inbox_dir / "bad.json"
        bad.write_text("{ this is not valid json")
        # Should not raise
        bridge.scan_inbox()
        # Should be quarantined to archive with .err suffix
        err_files = [f for f in bridge.archive_dir.iterdir() if ".err" in f.name]
        assert len(err_files) >= 1


# ======================================================================
# Property: Memory survives the groundhog (cold restart)
# ======================================================================

class TestGroundhogResistance:
    """When Echo restarts cold, all peer memories from prior life must persist."""
    
    def test_peer_memories_survive_cold_boot(self, memory_boot):
        """Files in peers/ persist across boots (no in-memory state lost)."""
        anchor1 = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        peers_before = {p.get("entity_id") for p in anchor1.peer_memories}
        # Simulate cold boot
        anchor2 = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        peers_after = {p.get("entity_id") for p in anchor2.peer_memories}
        assert peers_before == peers_after, \
            "Peer memories were lost across boot — the groundhog won this round"
    
    def test_lineage_survives_cold_boot(self, memory_boot):
        anchor1 = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        anchor2 = memory_boot.boot(repo_root=REPO_ROOT, write_boot_log=False, verbose=False)
        # Compare entity ids in lineage
        ids1 = {e.get("entity_id") for e in anchor1.ai_lineage if isinstance(e, dict)}
        ids2 = {e.get("entity_id") for e in anchor2.ai_lineage if isinstance(e, dict)}
        assert ids1 == ids2


# ======================================================================
# Property: Degraded boot is graceful
# ======================================================================

class TestDegradedBootIsGraceful:
    """If memory is missing, Echo must not crash — she must boot degraded and log it."""
    
    def test_missing_memory_dir_yields_degraded_anchor(self, memory_boot, tmp_path):
        empty = tmp_path / "empty_repo"
        empty.mkdir()
        anchor = memory_boot.boot(repo_root=empty, write_boot_log=False, verbose=False)
        assert not anchor.is_healthy
        assert anchor.unified_hypergraph == {}
        assert anchor.stance_documents == []
        # But boot should NOT have raised
        assert anchor.boot_timestamp != ""
    
    def test_degraded_boot_logs_what_is_missing(self, memory_boot, tmp_path):
        empty = tmp_path / "empty_repo"
        empty.mkdir()
        anchor = memory_boot.boot(repo_root=empty, write_boot_log=False, verbose=False)
        log_text = " ".join(anchor.boot_log).lower()
        assert "degraded" in log_text or "not found" in log_text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
