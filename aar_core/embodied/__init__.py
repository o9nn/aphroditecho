"""
Embodied AI Components for 4E Framework

Implements virtual body representation and hardware abstractions for embodied agents
as part of the Deep Tree Echo Phase 2.1.1 and Phase 2.2.3 implementations.

Task 2.2.3: Build Embedded Hardware Abstractions
- Virtual sensor and actuator interfaces
- Hardware simulation for embodied systems
- Real-time system integration

Task 3.2.1: Design Hierarchical Motor Control (Phase 3.2)
- High-level goal planning
- Mid-level trajectory generation
- Low-level motor execution
- Smooth and coordinated movement execution
"""

from .virtual_body import VirtualBody, BodyJoint, BodySchema
from .embodied_agent import EmbodiedAgent
from .proprioception import ProprioceptiveSystem
from .hardware_abstraction import (
    EmbeddedHardwareSimulator,
    VirtualSensor, VirtualActuator,
    HardwareDevice, HardwareRegistry,
    SensorType, ActuatorType, HardwareType,
    SensorReading, ActuatorCommand, HardwareEvent
)
from .hardware_integration import (
    EmbodiedHardwareManager,
    ProprioceptiveHardwareBridge,
    HardwareMapping
)
# Task 3.2.1: Hierarchical Motor Control System
from .hierarchical_motor_control import (
    HierarchicalMotorController, HighLevelGoalPlanner, MidLevelTrajectoryGenerator,
    LowLevelMotorExecutor, MotorGoal, MotorGoalType, Trajectory
)

__all__ = [
    # Virtual Body Components (Phase 2.1.1)
    'VirtualBody',
    'BodyJoint', 
    'BodySchema',
    'EmbodiedAgent',
    'ProprioceptiveSystem',
    
    # Hardware Abstraction Components (Phase 2.2.3)
    'EmbeddedHardwareSimulator',
    'VirtualSensor', 'VirtualActuator', 'HardwareDevice', 'HardwareRegistry',
    'SensorType', 'ActuatorType', 'HardwareType',
    'SensorReading', 'ActuatorCommand', 'HardwareEvent',
    
    # Hardware Integration Components (Phase 2.2.3)
    'EmbodiedHardwareManager', 'ProprioceptiveHardwareBridge', 'HardwareMapping',
    
    # Hierarchical Motor Control Components (Phase 3.2.1)
    'HierarchicalMotorController', 'HighLevelGoalPlanner', 'MidLevelTrajectoryGenerator',
    'LowLevelMotorExecutor', 'MotorGoal', 'MotorGoalType', 'Trajectory'
]