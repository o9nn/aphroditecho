"""
AAR Core Orchestrator

Central orchestration system for agent-arena-relation management.
Integrates with Aphrodite Engine and Echo-Self Evolution Engine.
"""

from typing import Dict, List, Any, Optional
import asyncio
import logging
from dataclasses import dataclass

from ..agents.agent_manager import AgentManager, AgentCapabilities, AgentStatus
from ..arena.simulation_engine import SimulationEngine, ArenaType
from ..relations.relation_graph import RelationGraph, RelationType

logger = logging.getLogger(__name__)


@dataclass
class AARConfig:
    """Configuration for AAR orchestration system."""
    max_concurrent_agents: int = 1000
    arena_simulation_enabled: bool = True
    relation_graph_depth: int = 3
    resource_allocation_strategy: str = "adaptive"
    agent_lifecycle_timeout: int = 300  # seconds
    performance_monitoring_interval: int = 10  # seconds


class AARCoreOrchestrator:
    """Central orchestration system for agent-arena-relation management."""
    
    def __init__(self, config: AARConfig):
        self.config = config
        
        # Initialize core components
        self.agent_manager = AgentManager(config.max_concurrent_agents)
        self.simulation_engine = SimulationEngine() if config.arena_simulation_enabled else None
        self.relation_graph = RelationGraph(config.relation_graph_depth)
        
        # Integration points
        self.aphrodite_engine = None
        self.dtesn_kernel = None
        self.echo_self_engine = None
        
        # Performance metrics
        self.performance_stats = {
            'total_requests': 0,
            'active_agents_count': 0,
            'arena_utilization': 0.0,
            'avg_response_time': 0.0,
            'error_rate': 0.0
        }
        
        logger.info(f"AAR Core Orchestrator initialized with config: {config}")
    
    def set_aphrodite_integration(self, aphrodite_engine):
        """Set Aphrodite Engine integration."""
        self.aphrodite_engine = aphrodite_engine
        logger.info("Aphrodite Engine integration enabled")
    
    def set_dtesn_integration(self, dtesn_kernel):
        """Set DTESN kernel integration."""
        self.dtesn_kernel = dtesn_kernel
        logger.info("DTESN kernel integration enabled")
    
    def set_echo_self_integration(self, echo_self_engine):
        """Set Echo-Self evolution engine integration."""
        self.echo_self_engine = echo_self_engine
        logger.info("Echo-Self evolution engine integration enabled")
    
    def enable_echo_self_integration(self, echo_self_engine):
        """Enable integration with Echo-Self evolution engine (alias)."""
        self.set_echo_self_integration(echo_self_engine)
    
    async def orchestrate_inference(self, 
                                      request: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate inference through agent-arena system."""
        try:
            self.performance_stats['total_requests'] += 1
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: Allocate appropriate agents for request
            allocated_agents = await self._allocate_agents(request)
            
            if not allocated_agents:
                return {'error': 'No agents available for request'}
            
            # Step 2: Create or select virtual arena if simulation enabled
            arena_id = None
            if self.simulation_engine:
                arena_id = await self._get_arena(request.get('context', {}))
            
            # Step 3: Execute distributed inference
            results = []
            for agent_id in allocated_agents:
                # Get agent data
                agent_data = self.agent_manager.get_agent_status(agent_id)
                if not agent_data:
                    continue
                
                # Update agent with current membrane states
                await self._sync_agent_membranes(agent_data)
                
                # Execute in virtual arena or directly
                if arena_id and self.simulation_engine:
                    agent_result = await self.simulation_engine.execute_agent_in_arena(
                        arena_id, agent_data, request
                    )
                else:
                    # Execute directly through agent manager
                    agent_result = await self.agent_manager.process_agent_request(
                        agent_id, request
                    )
                
                results.append(agent_result)
            
            # Step 4: Aggregate results through relation graph
            final_result = await self.relation_graph.aggregate_results(results)
            
            # Step 5: Update relationships based on performance
            await self._update_relations(allocated_agents, final_result)
            
            # Update performance metrics
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            await self._update_performance_stats(response_time, success=True)
            
            # Add orchestration metadata
            final_result.update({
                'orchestration_meta': {
                    'agents_used': len(allocated_agents),
                    'arena_id': arena_id,
                    'processing_time': response_time,
                    'request_id': request.get('request_id', 'unknown')
                }
            })
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in orchestrate_inference: {e}")
            await self._update_performance_stats(0.0, success=False)
            return {'error': str(e)}
    
    async def _allocate_agents(self, request: Dict[str, Any]) -> List[str]:
        """Allocate appropriate agents for request."""
        # Determine required agent count and capabilities
        agent_count = self._calculate_required_agents(request)
        
        # Use agent manager to allocate agents
        allocated_agents = await self.agent_manager.allocate_agents(request, agent_count)
        
        logger.debug(f"Allocated {len(allocated_agents)} agents for request")
        return allocated_agents
    
    def _calculate_required_agents(self, request: Dict[str, Any]) -> int:
        """Calculate number of agents required for request."""
        # Basic heuristic - in practice this would be more sophisticated
        base_agents = 1
        
        # Adjust based on request complexity
        if 'complex_reasoning' in request.get('features', []):
            base_agents += 2
        
        if 'collaboration' in request.get('features', []):
            base_agents += 2  # Collaboration requires multiple agents
            
        if 'multi_modal' in request.get('features', []):
            base_agents += 1
        
        if request.get('priority', 'normal') == 'high':
            base_agents += 1
        
        # Check for explicit minimum agent requirement
        min_agents = request.get('required_capabilities', {}).get('min_agents', 1)
        base_agents = max(base_agents, min_agents)
        
        # Check context requirements
        context = request.get('context', {})
        if context.get('interaction_type') == 'complex_collaboration':
            base_agents = max(base_agents, 3)
        
        return min(base_agents, 10)  # Cap at reasonable limit
    
    async def _get_arena(self, context: Dict[str, Any]) -> str:
        """Create or select virtual arena."""
        if not self.simulation_engine:
            return None
        
        # Determine arena type from context
        arena_type_name = context.get('arena_type', 'GENERAL')
        arena_type = ArenaType.GENERAL
        
        try:
            arena_type = ArenaType(arena_type_name.lower())
        except ValueError:
            logger.warning(f"Unknown arena type {arena_type_name}, using GENERAL")
        
        # Get or create arena
        arena_id = await self.simulation_engine.get_or_create_arena(context, arena_type)
        
        return arena_id
    
    async def _sync_agent_membranes(self, agent_data: Dict[str, Any]) -> None:
        """Update agent with current membrane states."""
        if self.dtesn_kernel:
            # Get current membrane states from DTESN kernel
            membrane_states = await self._get_dtesn_membrane_states()
            agent_data['membrane_states'] = membrane_states
    
    async def _get_dtesn_membrane_states(self) -> Dict[str, Any]:
        """Get current DTESN membrane states."""
        # Placeholder for DTESN integration
        return {
            'hierarchy_depth': 4,
            'active_membranes': [],
            'reservoir_dynamics': {},
            'evolution_state': {}
        }
    
    async def _update_relations(self, agent_ids: List[str], result: Dict[str, Any]) -> None:
        """Update relationships based on performance."""
        if len(agent_ids) < 2:
            return
        
        # Create agent data for relation graph
        agents = [{'id': agent_id} for agent_id in agent_ids]
        
        # Extract performance score from result
        performance_score = result.get('consensus_confidence', 0.5)
        
        # Update relationships through relation graph
        await self.relation_graph.update_relationships(agents, performance_score)
    
    async def _update_performance_stats(self, response_time: float, success: bool) -> None:
        """Update system performance statistics."""
        # Update average response time
        total_requests = self.performance_stats['total_requests']
        current_avg = self.performance_stats['avg_response_time']
        
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.performance_stats['avg_response_time'] = new_avg
        
        # Update error rate
        if not success:
            errors = self.performance_stats.get('total_errors', 0) + 1
            self.performance_stats['total_errors'] = errors
            self.performance_stats['error_rate'] = errors / total_requests
        
        # Update active agents count
        if self.agent_manager:
            agent_stats = self.agent_manager.get_system_stats()
            self.performance_stats['active_agents_count'] = agent_stats['agent_counts']['active']
        
        # Update arena utilization
        if self.simulation_engine:
            arenas = self.simulation_engine.list_arenas()
            active_arenas = len([a for a in arenas if a['active_sessions'] > 0])
            total_arenas = len(arenas)
            self.performance_stats['arena_utilization'] = active_arenas / max(total_arenas, 1)
    
    async def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get current orchestration statistics."""
        agent_stats = self.agent_manager.get_system_stats() if self.agent_manager else {}
        simulation_stats = self.simulation_engine.get_system_stats() if self.simulation_engine else {}
        relation_stats = self.relation_graph.get_graph_stats() if self.relation_graph else {}
        
        # Extract active agents count for top-level access
        active_agents_count = 0
        if agent_stats and 'agent_counts' in agent_stats:
            active_agents_count = agent_stats['agent_counts'].get('active', 0)
        
        return {
            'performance_stats': self.performance_stats,
            'config': self.config,
            'active_agents_count': active_agents_count,  # Add for test compatibility
            'integration_status': {
                'aphrodite_engine': self.aphrodite_engine is not None,
                'dtesn_kernel': self.dtesn_kernel is not None,
                'echo_self_engine': self.echo_self_engine is not None
            },
            'component_stats': {
                'agents': agent_stats,
                'simulation': simulation_stats,
                'relations': relation_stats
            },
            'system_health': await self._calculate_system_health()
        }
    
    async def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        health_score = 1.0
        
        # Adjust based on error rate
        error_rate = self.performance_stats.get('error_rate', 0.0)
        health_score -= error_rate
        
        # Adjust based on agent manager health
        if self.agent_manager:
            agent_health = self.agent_manager.get_system_stats()['health_status']
            health_score *= agent_health['overall_score']
        
        return {
            'overall_score': max(0.0, min(1.0, health_score)),
            'components_healthy': {
                'agent_manager': self.agent_manager is not None,
                'simulation_engine': self.simulation_engine is not None,
                'relation_graph': self.relation_graph is not None
            },
            'error_rate': error_rate,
            'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'critical'
        }
    
    async def run_agent_evaluation(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run agent evaluation in virtual arena for evolution engine."""
        try:
            # Spawn agent for evaluation if not already active
            agent_id = agent_data.get('id')
            if not agent_id or agent_id not in self.agent_manager.agents:
                # Create temporary agent for evaluation
                capabilities = AgentCapabilities(
                    reasoning=agent_data.get('reasoning', True),
                    multimodal=agent_data.get('multimodal', False),
                    memory_enabled=agent_data.get('memory_enabled', True),
                    learning_enabled=agent_data.get('learning_enabled', True),
                    collaboration=agent_data.get('collaboration', True)
                )
                agent_id = await self.agent_manager.spawn_agent(capabilities, context={'evaluation': True})
            
            # Create evaluation arena
            arena_id = None
            if self.simulation_engine:
                arena_context = {
                    'evaluation_mode': True,
                    'agent_count': 1,
                    'arena_type': 'evaluation'
                }
                arena_id = await self._get_arena(arena_context)
            
            # Run evaluation tasks
            evaluation_tasks = [
                {'type': 'reasoning', 'complexity': 'medium'},
                {'type': 'problem_solving', 'complexity': 'high'},
                {'type': 'adaptation', 'complexity': 'variable'}
            ]
            
            results = []
            for task in evaluation_tasks:
                task_result = await self._evaluate_agent_task(agent_id, task, arena_id)
                results.append(task_result)
            
            # Calculate overall fitness score
            fitness_score = self._calculate_agent_fitness(results)
            
            # Clean up temporary agent if created for evaluation
            if agent_data.get('temporary_agent', False):
                await self.agent_manager.terminate_agent(agent_id)
            
            return {
                'agent_id': agent_id,
                'arena_id': arena_id,
                'fitness_score': fitness_score,
                'task_results': results,
                'evaluation_timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error during agent evaluation: {e}")
            return {
                'agent_id': agent_data.get('id'),
                'fitness_score': 0.0,
                'error': str(e),
                'evaluation_timestamp': asyncio.get_event_loop().time()
            }
    
    async def _evaluate_agent_task(self, agent_id: str, task: Dict[str, Any], arena_id: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate agent performance on a specific task."""
        start_time = asyncio.get_event_loop().time()
        
        # Simulate task execution in arena
        if arena_id and self.simulation_engine:
            arena_result = await self._run_arena_task(agent_id, task, arena_id)
        else:
            # Basic task execution without arena
            arena_result = await self._run_basic_task(agent_id, task)
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        return {
            'task_type': task['type'],
            'complexity': task['complexity'],
            'execution_time': execution_time,
            'success_rate': arena_result.get('success_rate', 0.5),
            'performance_score': arena_result.get('performance_score', 0.5),
            'arena_interactions': arena_result.get('interactions', [])
        }
    
    async def _run_arena_task(self, agent_id: str, task: Dict[str, Any], arena_id: str) -> Dict[str, Any]:
        """Run task in virtual arena environment."""
        # Simulate arena-based task execution
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Mock arena interaction results based on task complexity
        base_performance = 0.6
        complexity_factor = {'low': 1.2, 'medium': 1.0, 'high': 0.8, 'variable': 0.9}.get(task['complexity'], 1.0)
        
        performance_score = min(1.0, base_performance * complexity_factor)
        success_rate = max(0.1, performance_score - 0.1)
        
        return {
            'success_rate': success_rate,
            'performance_score': performance_score,
            'interactions': [
                {'type': 'environment_analysis', 'score': performance_score},
                {'type': 'task_execution', 'score': success_rate},
                {'type': 'adaptation', 'score': min(1.0, performance_score + 0.1)}
            ]
        }
    
    async def _run_basic_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run basic task without arena simulation."""
        await asyncio.sleep(0.005)  # Simulate processing time
        
        # Mock basic task results
        base_performance = 0.5
        complexity_factor = {'low': 1.1, 'medium': 1.0, 'high': 0.9, 'variable': 0.95}.get(task['complexity'], 1.0)
        
        performance_score = min(1.0, base_performance * complexity_factor)
        
        return {
            'success_rate': performance_score,
            'performance_score': performance_score,
            'interactions': []
        }
    
    def _calculate_agent_fitness(self, task_results: List[Dict[str, Any]]) -> float:
        """Calculate overall agent fitness from task results."""
        if not task_results:
            return 0.0
        
        # Weighted fitness calculation
        total_score = 0.0
        total_weight = 0.0
        
        task_weights = {
            'reasoning': 0.4,
            'problem_solving': 0.4,
            'adaptation': 0.2
        }
        
        for result in task_results:
            task_type = result['task_type']
            weight = task_weights.get(task_type, 0.3)
            performance = result['performance_score']
            
            # Adjust for execution efficiency
            time_penalty = min(0.1, result['execution_time'] * 0.01)
            adjusted_performance = max(0.0, performance - time_penalty)
            
            total_score += adjusted_performance * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def update_agent_configurations(self, agent_configs: List[Dict[str, Any]]) -> None:
        """Update agent configurations after evolution."""
        # First, clean up old agents to make room for evolved ones
        active_count = len([a for a in self.agent_manager.agents.values() 
                           if a.status not in [AgentStatus.TERMINATED, AgentStatus.ERROR]])
        
        # Calculate available capacity
        available_capacity = self.config.max_concurrent_agents - active_count
        
        # Limit the number of new agents to available capacity
        max_new_agents = min(len(agent_configs), available_capacity)
        
        if max_new_agents < len(agent_configs):
            logger.warning(f"Limited new agents to {max_new_agents} due to capacity constraints")
            agent_configs = agent_configs[:max_new_agents]
        
        for config in agent_configs:
            agent_id = config.get('id')
            if agent_id and agent_id in self.agent_manager.agents:
                # Update agent with evolved parameters
                await self._apply_evolved_config(agent_id, config)
            elif not config.get('temporary_agent', False):  # Only create permanent agents
                # Spawn new evolved agent
                capabilities = self._config_to_capabilities(config)
                context = {'evolved': True, 'generation': config.get('generation', 0)}
                try:
                    new_agent_id = await self.agent_manager.spawn_agent(capabilities, context)
                    logger.info(f"Spawned evolved agent {new_agent_id} from generation {config.get('generation', 0)}")
                except Exception as e:
                    logger.warning(f"Failed to spawn evolved agent: {e}")
                    break  # Stop spawning if we hit capacity
    
    async def _apply_evolved_config(self, agent_id: str, config: Dict[str, Any]) -> None:
        """Apply evolved configuration to existing agent."""
        agent = self.agent_manager.agents.get(agent_id)
        if agent:
            # Update agent parameters
            evolution_data = {
                'new_capabilities': config.get('capabilities', {}),
                'learning_parameters': config.get('learning_params', {}),
                'performance_targets': config.get('targets', {})
            }
            await self.agent_manager.evolve_agent(agent_id, evolution_data)
    
    def _config_to_capabilities(self, config: Dict[str, Any]) -> AgentCapabilities:
        """Convert evolution config to agent capabilities."""
        cap_data = config.get('capabilities', {})
        return AgentCapabilities(
            reasoning=cap_data.get('reasoning', True),
            multimodal=cap_data.get('multimodal', False),
            memory_enabled=cap_data.get('memory_enabled', True),
            learning_enabled=cap_data.get('learning_enabled', True),
            collaboration=cap_data.get('collaboration', True),
            specialized_domains=cap_data.get('domains', []),
            max_context_length=cap_data.get('context_length', 4096),
            processing_power=cap_data.get('processing_power', 1.0)
        )
    
    async def shutdown(self) -> None:
        """Graceful shutdown of orchestration system."""
        logger.info("Shutting down AAR Core Orchestrator...")
        
        # Shutdown components in order
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        if self.simulation_engine:
            await self.simulation_engine.shutdown()
        
        if self.relation_graph:
            await self.relation_graph.shutdown()
        
        logger.info("AAR Core Orchestrator shutdown complete")