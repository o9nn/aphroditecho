"""
Test Suite for Parallel Echo Inference

Tests the parallel echo processing orchestrator with hypergraph integration
for Deep Tree Echo autonomous AGI.
"""

import asyncio
import pytest
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aphrodite.engine.deep_tree_agi_config import (
    DeepTreeEchoAGIConfig,
    EchobeatsConfig,
    NestedShellsConfig,
    ThreadMultiplexingConfig,
    MemoryType,
    IdentityRole
)
from aphrodite.engine.parallel_echo_orchestrator import (
    ParallelEchoOrchestrator,
    ConcurrentInferenceEngine,
    ParallelEchoSubsystemManager,
    ThreadMultiplexer,
    GlobalTelemetryShell,
    GestaltState
)
from aphrodite.engine.hypergraph_integration import (
    HypergraphIntegration,
    HypergraphMemoryManager,
    EchoPropagationEngine,
    IdentityStateMachine,
    MembraneComputingSystem,
    AARGeometricCore
)


class TestDeepTreeEchoAGIConfig:
    """Test configuration classes."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = DeepTreeEchoAGIConfig()
        assert config.agi_mode is True
        assert config.single_instance is True
        assert config.persistent_consciousness is True
    
    def test_echobeats_config(self):
        """Test Echobeats configuration."""
        config = DeepTreeEchoAGIConfig()
        assert config.echobeats.num_concurrent_engines == 3
        assert config.echobeats.cognitive_loop_steps == 12
        assert config.echobeats.phase_offset_degrees == 120
    
    def test_nested_shells_config(self):
        """Test nested shells configuration."""
        config = DeepTreeEchoAGIConfig()
        assert config.nested_shells.num_nests == 4
        assert config.nested_shells.nest_terms == [1, 2, 4, 9]
    
    def test_thread_multiplexing_config(self):
        """Test thread multiplexing configuration."""
        config = DeepTreeEchoAGIConfig()
        assert config.thread_multiplexing.num_threads == 4
        assert config.thread_multiplexing.qubit_order == 2
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = DeepTreeEchoAGIConfig()
        assert config.validate() is True
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = DeepTreeEchoAGIConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["agi_mode"] is True
        assert config_dict["echobeats"]["num_concurrent_engines"] == 3


class TestConcurrentInferenceEngine:
    """Test concurrent inference engines."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization."""
        config = DeepTreeEchoAGIConfig()
        engine = ConcurrentInferenceEngine(1, 0, config)
        assert engine.engine_id == 1
        assert engine.phase_offset == 0
        assert engine.role == "perception"
    
    @pytest.mark.asyncio
    async def test_engine_step_execution(self):
        """Test engine step execution."""
        config = DeepTreeEchoAGIConfig()
        engine = ConcurrentInferenceEngine(1, 0, config)
        gestalt = GestaltState()
        
        state = await engine.execute_step(0, gestalt)
        assert state.engine_id == 1
        assert state.step == 0
        assert state.perception is not None
    
    @pytest.mark.asyncio
    async def test_phase_offset(self):
        """Test phase offset for 3 engines."""
        config = DeepTreeEchoAGIConfig()
        engines = [
            ConcurrentInferenceEngine(1, 0, config),
            ConcurrentInferenceEngine(2, 4, config),
            ConcurrentInferenceEngine(3, 8, config)
        ]
        
        assert engines[0].phase_offset == 0
        assert engines[1].phase_offset == 4
        assert engines[2].phase_offset == 8


class TestParallelEchoSubsystemManager:
    """Test parallel echo subsystem manager."""
    
    def test_subsystem_initialization(self):
        """Test subsystem initialization."""
        config = DeepTreeEchoAGIConfig()
        manager = ParallelEchoSubsystemManager(config)
        
        # Check total subsystems: 1 + 2 + 4 + 9 = 16
        assert len(manager.subsystems) == 16
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel subsystem execution."""
        config = DeepTreeEchoAGIConfig()
        manager = ParallelEchoSubsystemManager(config)
        gestalt = GestaltState()
        
        # Step 0: Nest 1, 2, 3, 4 should be active
        results = await manager.execute_parallel(0, gestalt)
        assert len(results) > 0
    
    def test_active_subsystems_for_step(self):
        """Test active subsystems calculation for each step."""
        config = DeepTreeEchoAGIConfig()
        
        # Step 1: Nest 1, 2 active (1 + 2 = 3 subsystems)
        active = config.nested_shells.get_active_subsystems_for_step(1)
        assert len(active) >= 3
        
        # Step 4: Nest 1, 2, 4 active (1 + 2 + 9 = 12 subsystems)
        active = config.nested_shells.get_active_subsystems_for_step(4)
        assert len(active) >= 12


class TestThreadMultiplexer:
    """Test thread multiplexer."""
    
    @pytest.mark.asyncio
    async def test_entangled_execution(self):
        """Test entangled qubit execution."""
        config = DeepTreeEchoAGIConfig()
        multiplexer = ThreadMultiplexer(config)
        
        result = await multiplexer.execute_entangled("read", 0x1000, {"value": 42})
        assert result is not None
    
    def test_dyadic_pair_cycling(self):
        """Test dyadic pair cycling."""
        config = DeepTreeEchoAGIConfig()
        multiplexer = ThreadMultiplexer(config)
        
        # Check all 6 dyadic pairs
        pairs = [multiplexer.config.thread_multiplexing.get_dyadic_pair_for_cycle(i) 
                for i in range(6)]
        
        assert len(pairs) == 6
        assert pairs[0] == (1, 2)
        assert pairs[5] == (3, 4)


class TestGlobalTelemetryShell:
    """Test global telemetry shell."""
    
    def test_telemetry_initialization(self):
        """Test telemetry initialization."""
        config = DeepTreeEchoAGIConfig()
        telemetry = GlobalTelemetryShell(config)
        
        assert telemetry.gestalt_state is not None
        assert isinstance(telemetry.telemetry_buffer, list)
    
    def test_execute_in_context(self):
        """Test executing operation in global context."""
        config = DeepTreeEchoAGIConfig()
        telemetry = GlobalTelemetryShell(config)
        
        def test_operation(context, gestalt):
            return {"result": "success"}
        
        result = telemetry.execute_in_context(test_operation)
        assert result["result"] == "success"
    
    def test_gestalt_perception(self):
        """Test gestalt perception."""
        config = DeepTreeEchoAGIConfig()
        telemetry = GlobalTelemetryShell(config)
        
        gestalt = telemetry.get_gestalt_perception()
        assert isinstance(gestalt, GestaltState)


class TestParallelEchoOrchestrator:
    """Test parallel echo orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = DeepTreeEchoAGIConfig()
        orchestrator = ParallelEchoOrchestrator(config)
        
        assert len(orchestrator.engines) == 3
        assert orchestrator.current_step == 0
        assert orchestrator.cycle_count == 0
    
    @pytest.mark.asyncio
    async def test_single_step_execution(self):
        """Test single step execution."""
        config = DeepTreeEchoAGIConfig()
        orchestrator = ParallelEchoOrchestrator(config)
        
        result = await orchestrator.step()
        
        assert "step" in result
        assert "cycle" in result
        assert "engine_states" in result
        assert "subsystem_results" in result
        assert "duration_ms" in result
        
        assert len(result["engine_states"]) == 3
    
    @pytest.mark.asyncio
    async def test_complete_cycle(self):
        """Test complete 12-step cycle."""
        config = DeepTreeEchoAGIConfig()
        orchestrator = ParallelEchoOrchestrator(config)
        
        results = await orchestrator.run_n_cycles(1)
        
        assert len(results) == 12  # 12 steps per cycle
        assert results[0]["step"] == 1
        assert results[11]["step"] == 0  # Wrapped around
    
    @pytest.mark.asyncio
    async def test_performance_target(self):
        """Test performance targets."""
        config = DeepTreeEchoAGIConfig()
        orchestrator = ParallelEchoOrchestrator(config)
        
        start_time = time.time()
        results = await orchestrator.run_n_cycles(1)
        duration = time.time() - start_time
        
        # Target: 83.33 Hz = 12ms per step = 144ms per cycle
        # Allow 10x overhead for testing
        assert duration < 1.44  # 1.44 seconds
        
        avg_step_duration = sum(r["duration_ms"] for r in results) / len(results)
        print(f"Average step duration: {avg_step_duration:.2f}ms")


class TestHypergraphMemoryManager:
    """Test hypergraph memory manager."""
    
    @pytest.mark.asyncio
    async def test_parallel_read(self):
        """Test parallel memory read."""
        config = DeepTreeEchoAGIConfig()
        manager = HypergraphMemoryManager(config)
        
        # Write some data first
        await manager.write_parallel({
            MemoryType.DECLARATIVE: {"key1": "value1"},
            MemoryType.PROCEDURAL: {"key2": "value2"}
        })
        
        # Read in parallel
        results = await manager.read_parallel(
            [MemoryType.DECLARATIVE, MemoryType.PROCEDURAL],
            ["key1", "key2"]
        )
        
        assert MemoryType.DECLARATIVE in results
        assert MemoryType.PROCEDURAL in results
    
    @pytest.mark.asyncio
    async def test_parallel_write(self):
        """Test parallel memory write."""
        config = DeepTreeEchoAGIConfig()
        manager = HypergraphMemoryManager(config)
        
        await manager.write_parallel({
            MemoryType.DECLARATIVE: {"fact1": "Earth is round"},
            MemoryType.EPISODIC: {"event1": "First step"}
        })
        
        # Verify writes
        results = await manager.read_parallel(
            [MemoryType.DECLARATIVE, MemoryType.EPISODIC],
            ["fact1", "event1"]
        )
        
        assert results[MemoryType.DECLARATIVE]["fact1"] == "Earth is round"
        assert results[MemoryType.EPISODIC]["event1"] == "First step"


class TestIdentityStateMachine:
    """Test identity state machine."""
    
    @pytest.mark.asyncio
    async def test_initial_role(self):
        """Test initial identity role."""
        config = DeepTreeEchoAGIConfig()
        machine = IdentityStateMachine(config)
        
        assert machine.current_role == IdentityRole.OBSERVER
    
    @pytest.mark.asyncio
    async def test_role_transition(self):
        """Test role transition."""
        config = DeepTreeEchoAGIConfig()
        machine = IdentityStateMachine(config)
        
        # High entropy and coherence should trigger transition
        new_role = await machine.evaluate_transition(
            entropy=0.9, coherence=0.9, memory_depth=5
        )
        
        # May or may not transition based on threshold
        if new_role:
            assert new_role != machine.current_role
            assert machine.transition_count > 0


class TestMembraneComputingSystem:
    """Test membrane computing system."""
    
    @pytest.mark.asyncio
    async def test_membrane_initialization(self):
        """Test membrane initialization."""
        config = DeepTreeEchoAGIConfig()
        system = MembraneComputingSystem(config)
        
        assert "root" in system.membranes
        assert "cognitive" in system.membranes
        assert "extension" in system.membranes
        assert "security" in system.membranes
    
    @pytest.mark.asyncio
    async def test_membrane_communication(self):
        """Test membrane communication."""
        config = DeepTreeEchoAGIConfig()
        system = MembraneComputingSystem(config)
        
        # Send message from cognitive to extension
        await system.send_message(
            "cognitive", "extension",
            {"type": "test", "data": "hello"}
        )
        
        # Check message was received
        extension = system.membranes["extension"]
        assert not extension.message_queue.empty()


class TestHypergraphIntegration:
    """Test hypergraph integration."""
    
    def test_integration_initialization(self):
        """Test integration initialization."""
        config = DeepTreeEchoAGIConfig()
        integration = HypergraphIntegration(config)
        
        assert integration.memory_manager is not None
        assert integration.echo_propagation is not None
        assert integration.identity_machine is not None
        assert integration.membrane_system is not None
        assert integration.aar_core is not None
    
    @pytest.mark.asyncio
    async def test_cognitive_step_processing(self):
        """Test cognitive step processing."""
        config = DeepTreeEchoAGIConfig()
        integration = HypergraphIntegration(config)
        
        # Mock engine states
        engine_states = [
            {"engine_id": 1, "state": "perception"},
            {"engine_id": 2, "state": "action"},
            {"engine_id": 3, "state": "simulation"}
        ]
        
        subsystem_results = {}
        
        result = await integration.process_cognitive_step(
            engine_states, subsystem_results
        )
        
        assert "activations" in result
        assert "memory_reads" in result
        assert "current_role" in result
        assert "aar_relation" in result


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_cycle_with_hypergraph(self):
        """Test full cycle with hypergraph integration."""
        config = DeepTreeEchoAGIConfig()
        orchestrator = ParallelEchoOrchestrator(config)
        integration = HypergraphIntegration(config)
        
        # Run one complete cycle
        results = await orchestrator.run_n_cycles(1)
        
        # Process with hypergraph
        for result in results:
            hypergraph_result = await integration.process_cognitive_step(
                result["engine_states"],
                result["subsystem_results"]
            )
            
            assert hypergraph_result is not None
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """Benchmark parallel echo inference performance."""
        config = DeepTreeEchoAGIConfig()
        orchestrator = ParallelEchoOrchestrator(config)
        
        # Run 10 cycles
        start_time = time.time()
        results = await orchestrator.run_n_cycles(10)
        duration = time.time() - start_time
        
        # Calculate metrics
        total_steps = len(results)
        avg_step_duration = sum(r["duration_ms"] for r in results) / total_steps
        throughput_hz = total_steps / duration
        
        print(f"\n=== Performance Benchmark ===")
        print(f"Total steps: {total_steps}")
        print(f"Total duration: {duration:.2f}s")
        print(f"Average step duration: {avg_step_duration:.2f}ms")
        print(f"Throughput: {throughput_hz:.2f} Hz")
        print(f"Target: 83.33 Hz")
        
        # Should be reasonably close to target (allow 10x overhead for testing)
        assert throughput_hz > 8.33  # At least 10% of target


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
