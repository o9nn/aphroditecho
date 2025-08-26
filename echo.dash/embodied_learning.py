#!/usr/bin/env python3
"""
Embodied Learning Algorithms for Deep Tree Echo

Implements sensorimotor learning for body awareness, spatial reasoning based on
body constraints, and motor skill acquisition through embodied practice.

This module integrates with the DTESN cognitive architecture and provides
the foundation for 4E Embodied AI Framework implementation.
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Import existing cognitive architecture if available
try:
    from cognitive_architecture import Memory, MemoryType, CognitiveArchitecture
    COGNITIVE_ARCH_AVAILABLE = True
except ImportError:
    COGNITIVE_ARCH_AVAILABLE = False
    
# Import sensory motor system for integration
try:
    from sensory_motor import SensoryMotor
    SENSORY_MOTOR_AVAILABLE = True
except ImportError:
    SENSORY_MOTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BodyState:
    """Represents the current state of the embodied agent's body"""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    joint_angles: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    muscle_activations: Dict[str, float] = field(default_factory=dict)
    sensory_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_vector(self) -> np.ndarray:
        """Convert body state to numerical vector for learning algorithms"""
        vector = list(self.position) + list(self.orientation)
        vector.extend(self.joint_angles.values())
        vector.extend(self.joint_velocities.values())
        vector.extend(self.muscle_activations.values())
        return np.array(vector)
    
    def update_from_sensory_input(self, sensory_data: Dict[str, Any]):
        """Update body state from sensory input"""
        self.sensory_state.update(sensory_data)
        self.timestamp = time.time()


@dataclass 
class MotorAction:
    """Represents a motor action to be executed"""
    joint_targets: Dict[str, float] = field(default_factory=dict)
    muscle_commands: Dict[str, float] = field(default_factory=dict)
    duration: float = 1.0
    force: float = 1.0
    precision: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_vector(self) -> np.ndarray:
        """Convert motor action to numerical vector"""
        vector = list(self.joint_targets.values()) + list(self.muscle_commands.values())
        vector.extend([self.duration, self.force, self.precision])
        return np.array(vector)


@dataclass
class SensorimotorExperience:
    """Represents a complete sensorimotor experience for learning"""
    initial_body_state: BodyState
    motor_action: MotorAction
    resulting_body_state: BodyState
    sensory_feedback: Dict[str, Any]
    reward: float = 0.0
    success: bool = False
    timestamp: float = field(default_factory=time.time)


class BodySchemaLearner:
    """Learns and maintains body schema through embodied experience"""
    
    def __init__(self, body_dimensions: Dict[str, float]):
        self.body_dimensions = body_dimensions
        self.joint_limits = {}
        self.muscle_strength = {}
        self.proprioceptive_model = {}
        self.body_schema_weights = np.random.randn(100, 100) * 0.1  # Neural network weights
        self.learning_rate = 0.01
        self.experiences = []
        
    def update_body_schema(self, experience: SensorimotorExperience):
        """Update body schema based on new sensorimotor experience"""
        try:
            # Extract features from experience
            initial_state = experience.initial_body_state.to_vector()
            motor_action = experience.motor_action.to_vector()
            resulting_state = experience.resulting_body_state.to_vector()
            
            # Pad vectors to consistent size
            max_size = max(len(initial_state), len(motor_action), len(resulting_state))
            initial_state = np.pad(initial_state, (0, max_size - len(initial_state)))
            motor_action = np.pad(motor_action, (0, max_size - len(motor_action)))
            resulting_state = np.pad(resulting_state, (0, max_size - len(resulting_state)))
            
            # Compute prediction error
            predicted_state = self._predict_outcome(initial_state, motor_action)
            error = resulting_state - predicted_state
            
            # Update proprioceptive model using Hebbian learning
            self._update_proprioceptive_model(initial_state, motor_action, error)
            
            # Store experience for future learning
            self.experiences.append(experience)
            if len(self.experiences) > 1000:  # Limit memory usage
                self.experiences = self.experiences[-1000:]
                
            logger.debug(f"Updated body schema with prediction error: {np.mean(np.abs(error))}")
            
        except Exception as e:
            logger.error(f"Error updating body schema: {e}")
    
    def _predict_outcome(self, initial_state: np.ndarray, motor_action: np.ndarray) -> np.ndarray:
        """Predict the outcome of a motor action given initial body state"""
        # Simple forward model - can be enhanced with neural networks
        combined_input = np.concatenate([initial_state[:50], motor_action[:50]])
        if len(combined_input) < 100:
            combined_input = np.pad(combined_input, (0, 100 - len(combined_input)))
        
        # Forward pass through simple neural network
        hidden = np.tanh(np.dot(self.body_schema_weights, combined_input[:100]))
        prediction = np.dot(self.body_schema_weights.T[:len(initial_state)], hidden)[:len(initial_state)]
        
        return prediction
    
    def _update_proprioceptive_model(self, initial_state: np.ndarray, 
                                   motor_action: np.ndarray, error: np.ndarray):
        """Update proprioceptive model based on prediction error"""
        try:
            # Gradient descent update with proper shape handling
            combined_input = np.concatenate([initial_state[:50], motor_action[:50]])
            if len(combined_input) < 100:
                combined_input = np.pad(combined_input, (0, 100 - len(combined_input)))
            
            # Ensure error vector has correct shape
            if len(error) < 100:
                error = np.pad(error, (0, 100 - len(error)))
            error = error[:100]  # Truncate to fit
            
            # Update weights based on error with proper broadcasting
            gradient = np.outer(error, combined_input[:100]) * self.learning_rate
            self.body_schema_weights += gradient[:100, :100]
            
        except Exception as e:
            # Fallback to simple weight adjustment
            adjustment = self.learning_rate * 0.01 * np.random.randn(*self.body_schema_weights.shape)
            self.body_schema_weights += adjustment
        
    def get_body_awareness(self) -> Dict[str, float]:
        """Get current body awareness metrics"""
        return {
            'schema_confidence': self._compute_schema_confidence(),
            'proprioceptive_accuracy': self._compute_proprioceptive_accuracy(),
            'joint_flexibility': self._compute_joint_flexibility(),
            'muscle_coordination': self._compute_muscle_coordination()
        }
    
    def _compute_schema_confidence(self) -> float:
        """Compute confidence in current body schema"""
        if len(self.experiences) < 10:
            return 0.0
            
        recent_errors = []
        for exp in self.experiences[-10:]:
            try:
                initial = exp.initial_body_state.to_vector()
                action = exp.motor_action.to_vector()
                result = exp.resulting_body_state.to_vector()
                
                max_size = max(len(initial), len(action), len(result))
                initial = np.pad(initial, (0, max_size - len(initial)))
                action = np.pad(action, (0, max_size - len(action)))
                result = np.pad(result, (0, max_size - len(result)))
                
                predicted = self._predict_outcome(initial, action)
                error = np.mean(np.abs(result - predicted))
                recent_errors.append(error)
            except Exception:
                continue
                
        if not recent_errors:
            return 0.0
            
        avg_error = np.mean(recent_errors)
        confidence = 1.0 / (1.0 + avg_error)  # Higher confidence with lower error
        return float(confidence)
    
    def _compute_proprioceptive_accuracy(self) -> float:
        """Compute proprioceptive sensing accuracy"""
        # Simplified metric based on recent prediction accuracy
        return min(1.0, self._compute_schema_confidence() * 1.2)
    
    def _compute_joint_flexibility(self) -> float:
        """Compute joint flexibility metric"""
        if not self.experiences:
            return 0.5
            
        # Compute range of motion from recent experiences
        joint_ranges = {}
        for exp in self.experiences[-50:]:
            for joint, angle in exp.initial_body_state.joint_angles.items():
                if joint not in joint_ranges:
                    joint_ranges[joint] = [angle, angle]
                else:
                    joint_ranges[joint][0] = min(joint_ranges[joint][0], angle)
                    joint_ranges[joint][1] = max(joint_ranges[joint][1], angle)
        
        if not joint_ranges:
            return 0.5
            
        avg_range = np.mean([r[1] - r[0] for r in joint_ranges.values()])
        return min(1.0, avg_range / np.pi)  # Normalize to [0,1]
    
    def _compute_muscle_coordination(self) -> float:
        """Compute muscle coordination metric"""
        if len(self.experiences) < 5:
            return 0.5
            
        # Measure consistency of muscle activations for similar actions
        coordination_scores = []
        for i, exp in enumerate(self.experiences[-10:]):
            if exp.success:
                coordination_scores.append(1.0 - np.var(list(exp.motor_action.muscle_commands.values())))
        
        if not coordination_scores:
            return 0.5
            
        return min(1.0, max(0.0, np.mean(coordination_scores)))


class SpatialReasoningEngine:
    """Spatial reasoning engine based on body constraints"""
    
    def __init__(self, body_schema_learner: BodySchemaLearner):
        self.body_schema = body_schema_learner
        self.spatial_memory = {}
        self.reachability_map = {}
        self.obstacle_map = {}
        
    def update_spatial_understanding(self, experience: SensorimotorExperience):
        """Update spatial understanding from embodied experience"""
        try:
            # Extract spatial information
            position = experience.resulting_body_state.position
            reachable = experience.success
            
            # Update reachability map
            pos_key = self._discretize_position(position)
            if pos_key not in self.reachability_map:
                self.reachability_map[pos_key] = {'attempts': 0, 'successes': 0}
            
            self.reachability_map[pos_key]['attempts'] += 1
            if reachable:
                self.reachability_map[pos_key]['successes'] += 1
            
            # Update spatial memory with contextual information
            self.spatial_memory[pos_key] = {
                'last_body_state': experience.resulting_body_state,
                'last_action': experience.motor_action,
                'timestamp': time.time(),
                'sensory_context': experience.sensory_feedback
            }
            
        except Exception as e:
            logger.error(f"Error updating spatial understanding: {e}")
    
    def _discretize_position(self, position: Tuple[float, float, float], 
                           resolution: float = 0.1) -> Tuple[int, int, int]:
        """Discretize continuous position for spatial mapping"""
        return (
            int(position[0] / resolution),
            int(position[1] / resolution), 
            int(position[2] / resolution)
        )
    
    def plan_spatial_action(self, target_position: Tuple[float, float, float],
                          current_body_state: BodyState) -> Optional[MotorAction]:
        """Plan motor action to reach target position using spatial reasoning"""
        try:
            # Check if target is reachable based on body constraints
            if not self._is_position_reachable(target_position, current_body_state):
                logger.debug(f"Target position {target_position} not reachable")
                return None
            
            # Plan trajectory based on spatial memory
            trajectory = self._plan_trajectory(current_body_state.position, target_position)
            
            # Convert trajectory to motor commands
            motor_action = self._trajectory_to_motor_action(trajectory, current_body_state)
            
            return motor_action
            
        except Exception as e:
            logger.error(f"Error planning spatial action: {e}")
            return None
    
    def _is_position_reachable(self, target: Tuple[float, float, float],
                             current_state: BodyState) -> bool:
        """Check if target position is reachable given body constraints"""
        # Simple distance-based reachability check
        current_pos = current_state.position
        distance = np.sqrt(sum((t - c)**2 for t, c in zip(target, current_pos)))
        
        # Get body dimensions for reach calculation
        max_reach = sum(self.body_schema.body_dimensions.get(limb, 1.0) 
                       for limb in ['arm_length', 'torso_height']) * 1.5
        
        return distance <= max_reach
    
    def _plan_trajectory(self, start: Tuple[float, float, float],
                        end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Plan trajectory from start to end position"""
        # Simple linear interpolation trajectory
        steps = 10
        trajectory = []
        for i in range(steps + 1):
            alpha = i / steps
            point = tuple(s + alpha * (e - s) for s, e in zip(start, end))
            trajectory.append(point)
        
        return trajectory
    
    def _trajectory_to_motor_action(self, trajectory: List[Tuple[float, float, float]],
                                  current_state: BodyState) -> MotorAction:
        """Convert trajectory to motor action commands"""
        if len(trajectory) < 2:
            return MotorAction()
        
        # Calculate required joint movements for trajectory
        target_pos = trajectory[1]  # Next position in trajectory
        current_pos = current_state.position
        
        # Simple inverse kinematics approximation
        joint_targets = {}
        direction = np.array([t - c for t, c in zip(target_pos, current_pos)])
        
        # Map to joint space (simplified)
        joint_targets['shoulder'] = float(np.arctan2(direction[1], direction[0]))
        joint_targets['elbow'] = float(np.linalg.norm(direction) * 0.1)
        
        return MotorAction(
            joint_targets=joint_targets,
            duration=0.5,
            force=0.8,
            precision=0.9
        )
    
    def get_spatial_metrics(self) -> Dict[str, Any]:
        """Get current spatial reasoning metrics"""
        total_positions = len(self.reachability_map)
        if total_positions == 0:
            return {
                'spatial_coverage': 0.0,
                'reachability_accuracy': 0.0,
                'spatial_memory_size': 0
            }
        
        reachable_positions = sum(1 for pos_data in self.reachability_map.values()
                                if pos_data['successes'] > 0)
        
        total_attempts = sum(pos_data['attempts'] for pos_data in self.reachability_map.values())
        total_successes = sum(pos_data['successes'] for pos_data in self.reachability_map.values())
        
        return {
            'spatial_coverage': reachable_positions / total_positions if total_positions > 0 else 0.0,
            'reachability_accuracy': total_successes / total_attempts if total_attempts > 0 else 0.0,
            'spatial_memory_size': len(self.spatial_memory)
        }


class MotorSkillLearner:
    """Motor skill acquisition through embodied practice"""
    
    def __init__(self, body_schema_learner: BodySchemaLearner):
        self.body_schema = body_schema_learner
        self.skill_library = {}
        self.practice_history = {}
        self.skill_performance = {}
        
    def practice_skill(self, skill_name: str, target_outcome: Dict[str, Any],
                      current_state: BodyState) -> MotorAction:
        """Practice a motor skill and learn from experience"""
        try:
            # Get or initialize skill
            if skill_name not in self.skill_library:
                self.skill_library[skill_name] = self._initialize_skill(skill_name)
            
            skill = self.skill_library[skill_name]
            
            # Generate motor action based on current skill level
            motor_action = self._generate_skill_action(skill, current_state, target_outcome)
            
            # Record practice attempt
            if skill_name not in self.practice_history:
                self.practice_history[skill_name] = []
            
            practice_record = {
                'timestamp': time.time(),
                'initial_state': current_state,
                'action': motor_action,
                'target': target_outcome
            }
            
            self.practice_history[skill_name].append(practice_record)
            
            return motor_action
            
        except Exception as e:
            logger.error(f"Error practicing skill {skill_name}: {e}")
            return MotorAction()
    
    def learn_from_outcome(self, skill_name: str, outcome: SensorimotorExperience):
        """Learn from the outcome of skill practice"""
        try:
            if skill_name not in self.skill_library:
                return
            
            skill = self.skill_library[skill_name]
            
            # Update skill parameters based on outcome
            if outcome.success:
                # Reinforce successful actions
                self._reinforce_skill_action(skill, outcome)
            else:
                # Adjust skill to avoid unsuccessful actions
                self._adjust_skill_action(skill, outcome)
            
            # Update performance metrics
            if skill_name not in self.skill_performance:
                self.skill_performance[skill_name] = {
                    'attempts': 0,
                    'successes': 0,
                    'avg_reward': 0.0
                }
            
            perf = self.skill_performance[skill_name]
            perf['attempts'] += 1
            if outcome.success:
                perf['successes'] += 1
            
            # Update average reward using exponential moving average
            alpha = 0.1
            perf['avg_reward'] = (1 - alpha) * perf['avg_reward'] + alpha * outcome.reward
            
        except Exception as e:
            logger.error(f"Error learning from outcome for skill {skill_name}: {e}")
    
    def _initialize_skill(self, skill_name: str) -> Dict[str, Any]:
        """Initialize a new motor skill"""
        return {
            'name': skill_name,
            'parameters': np.random.randn(20) * 0.1,  # Skill parameters
            'variance': np.ones(20) * 0.2,  # Exploration variance
            'created_time': time.time(),
            'updates': 0
        }
    
    def _generate_skill_action(self, skill: Dict[str, Any], current_state: BodyState,
                             target_outcome: Dict[str, Any]) -> MotorAction:
        """Generate motor action for skill execution"""
        # Simple policy: use skill parameters with exploration
        params = skill['parameters']
        variance = skill['variance']
        
        # Add exploration noise
        exploration = np.random.normal(0, variance)
        action_params = params + exploration
        
        # Convert to motor action
        joint_targets = {
            'shoulder': float(action_params[0]),
            'elbow': float(action_params[1]),
            'wrist': float(action_params[2]) if len(action_params) > 2 else 0.0
        }
        
        muscle_commands = {
            'bicep': max(0, min(1, action_params[3])) if len(action_params) > 3 else 0.5,
            'tricep': max(0, min(1, action_params[4])) if len(action_params) > 4 else 0.5
        }
        
        return MotorAction(
            joint_targets=joint_targets,
            muscle_commands=muscle_commands,
            duration=max(0.1, min(2.0, action_params[5] if len(action_params) > 5 else 1.0)),
            force=max(0.1, min(1.0, action_params[6] if len(action_params) > 6 else 0.8)),
            precision=max(0.1, min(1.0, action_params[7] if len(action_params) > 7 else 0.7))
        )
    
    def _reinforce_skill_action(self, skill: Dict[str, Any], outcome: SensorimotorExperience):
        """Reinforce successful skill actions"""
        # Simple policy gradient update
        learning_rate = 0.01
        
        # Get action parameters that led to success
        action_vector = outcome.motor_action.to_vector()
        
        # Update skill parameters toward successful action
        if len(action_vector) <= len(skill['parameters']):
            skill['parameters'][:len(action_vector)] += learning_rate * action_vector * outcome.reward
        
        # Reduce exploration variance for successful actions
        skill['variance'] *= 0.99
        skill['updates'] += 1
    
    def _adjust_skill_action(self, skill: Dict[str, Any], outcome: SensorimotorExperience):
        """Adjust skill to avoid unsuccessful actions"""
        # Increase exploration variance after failures
        skill['variance'] *= 1.01
        skill['variance'] = np.minimum(skill['variance'], 1.0)  # Cap variance
        skill['updates'] += 1
    
    def get_skill_metrics(self) -> Dict[str, Any]:
        """Get motor skill learning metrics"""
        if not self.skill_performance:
            return {
                'total_skills': 0,
                'avg_success_rate': 0.0,
                'total_practice_time': 0
            }
        
        total_skills = len(self.skill_performance)
        success_rates = []
        total_attempts = 0
        
        for skill_name, perf in self.skill_performance.items():
            if perf['attempts'] > 0:
                success_rate = perf['successes'] / perf['attempts']
                success_rates.append(success_rate)
                total_attempts += perf['attempts']
        
        avg_success_rate = np.mean(success_rates) if success_rates else 0.0
        
        return {
            'total_skills': total_skills,
            'avg_success_rate': float(avg_success_rate),
            'total_practice_attempts': total_attempts,
            'skill_names': list(self.skill_performance.keys())
        }


class EmbodiedLearningSystem:
    """Main embodied learning system integrating all components"""
    
    def __init__(self, body_dimensions: Optional[Dict[str, float]] = None):
        # Default body dimensions
        if body_dimensions is None:
            body_dimensions = {
                'arm_length': 0.7,
                'leg_length': 0.9,
                'torso_height': 0.6,
                'head_size': 0.2
            }
        
        self.body_schema_learner = BodySchemaLearner(body_dimensions)
        self.spatial_reasoning = SpatialReasoningEngine(self.body_schema_learner)
        self.motor_skill_learner = MotorSkillLearner(self.body_schema_learner)
        
        # Integration with existing systems
        self.sensory_motor = None
        if SENSORY_MOTOR_AVAILABLE:
            try:
                self.sensory_motor = SensoryMotor()
            except Exception as e:
                logger.warning(f"Could not initialize SensoryMotor: {e}")
        
        self.cognitive_architecture = None
        if COGNITIVE_ARCH_AVAILABLE:
            try:
                self.cognitive_architecture = CognitiveArchitecture()
            except Exception as e:
                logger.warning(f"Could not initialize CognitiveArchitecture: {e}")
        
        self.learning_active = True
        self.experience_count = 0
    
    def process_embodied_experience(self, experience: SensorimotorExperience) -> Dict[str, Any]:
        """Process a complete embodied experience for learning"""
        try:
            results = {}
            
            if not self.learning_active:
                return results
            
            # Update body schema
            self.body_schema_learner.update_body_schema(experience)
            results['body_schema_updated'] = True
            
            # Update spatial understanding
            self.spatial_reasoning.update_spatial_understanding(experience)
            results['spatial_understanding_updated'] = True
            
            # Create embodied memory if cognitive architecture available
            if self.cognitive_architecture and COGNITIVE_ARCH_AVAILABLE:
                memory = Memory(
                    content=f"Embodied experience at position {experience.resulting_body_state.position}",
                    memory_type=MemoryType.EPISODIC,
                    timestamp=experience.timestamp,
                    emotional_valence=1.0 if experience.success else -0.5,
                    importance=0.8 if experience.success else 0.3
                )
                memory.context.update({
                    'body_position': experience.resulting_body_state.position,
                    'motor_action': experience.motor_action.joint_targets,
                    'success': experience.success,
                    'reward': experience.reward
                })
                results['embodied_memory_created'] = True
            
            self.experience_count += 1
            results['total_experiences'] = self.experience_count
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing embodied experience: {e}")
            return {'error': str(e)}
    
    def learn_motor_skill(self, skill_name: str, target_outcome: Dict[str, Any],
                         current_body_state: BodyState) -> Tuple[MotorAction, Dict[str, Any]]:
        """Learn a motor skill through embodied practice"""
        try:
            # Practice the skill
            motor_action = self.motor_skill_learner.practice_skill(
                skill_name, target_outcome, current_body_state
            )
            
            # Get current skill metrics
            skill_metrics = self.motor_skill_learner.get_skill_metrics()
            
            return motor_action, skill_metrics
            
        except Exception as e:
            logger.error(f"Error learning motor skill {skill_name}: {e}")
            return MotorAction(), {'error': str(e)}
    
    def update_from_skill_outcome(self, skill_name: str, outcome: SensorimotorExperience):
        """Update learning from motor skill practice outcome"""
        try:
            # Update motor skill learning
            self.motor_skill_learner.learn_from_outcome(skill_name, outcome)
            
            # Process as general embodied experience
            self.process_embodied_experience(outcome)
            
        except Exception as e:
            logger.error(f"Error updating from skill outcome: {e}")
    
    def plan_spatial_movement(self, target_position: Tuple[float, float, float],
                            current_body_state: BodyState) -> Optional[MotorAction]:
        """Plan movement to target position using spatial reasoning"""
        return self.spatial_reasoning.plan_spatial_action(target_position, current_body_state)
    
    def get_embodiment_metrics(self) -> Dict[str, Any]:
        """Get comprehensive embodied learning metrics"""
        try:
            metrics = {
                'body_awareness': self.body_schema_learner.get_body_awareness(),
                'spatial_metrics': self.spatial_reasoning.get_spatial_metrics(),
                'motor_skill_metrics': self.motor_skill_learner.get_skill_metrics(),
                'system_metrics': {
                    'total_experiences': self.experience_count,
                    'learning_active': self.learning_active,
                    'sensory_motor_available': self.sensory_motor is not None,
                    'cognitive_arch_available': self.cognitive_architecture is not None
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting embodiment metrics: {e}")
            return {'error': str(e)}
    
    def set_learning_active(self, active: bool):
        """Enable or disable active learning"""
        self.learning_active = active
        logger.info(f"Embodied learning {'activated' if active else 'deactivated'}")


# Factory function for easy instantiation
def create_embodied_learning_system(body_dimensions: Optional[Dict[str, float]] = None) -> EmbodiedLearningSystem:
    """Create and initialize an embodied learning system"""
    return EmbodiedLearningSystem(body_dimensions)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create embodied learning system
    embodied_system = create_embodied_learning_system()
    
    # Example body state
    body_state = BodyState(
        position=(0.0, 0.0, 1.0),
        joint_angles={'shoulder': 0.5, 'elbow': 1.0},
        joint_velocities={'shoulder': 0.0, 'elbow': 0.0}
    )
    
    # Example motor action
    motor_action = MotorAction(
        joint_targets={'shoulder': 1.0, 'elbow': 0.5},
        duration=1.0
    )
    
    # Create and process embodied experience
    experience = SensorimotorExperience(
        initial_body_state=body_state,
        motor_action=motor_action,
        resulting_body_state=body_state,
        sensory_feedback={'touch': 'object_contact'},
        reward=1.0,
        success=True
    )
    
    result = embodied_system.process_embodied_experience(experience)
    print(f"Processing result: {result}")
    
    # Get embodiment metrics
    metrics = embodied_system.get_embodiment_metrics()
    print(f"Embodiment metrics: {metrics}")