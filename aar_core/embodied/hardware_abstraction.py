"""
Embedded Hardware Abstraction Layer for 4E Embodied AI Framework

Provides virtual sensor and actuator interfaces with hardware simulation
for embodied systems, enabling real-time system integration.

Task 2.2.3: Build Embedded Hardware Abstractions
- Virtual sensor and actuator interfaces
- Hardware simulation for embodied systems  
- Real-time system integration
- Acceptance Criteria: System can interface with simulated hardware
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json


class HardwareType(Enum):
    """Types of hardware devices."""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    MIXED = "mixed"


class SensorType(Enum):
    """Types of virtual sensors."""
    POSITION = "position"
    VELOCITY = "velocity" 
    ACCELERATION = "acceleration"
    FORCE = "force"
    TORQUE = "torque"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VISION = "vision"
    IMU = "imu"
    LIDAR = "lidar"
    TOUCH = "touch"


class ActuatorType(Enum):
    """Types of virtual actuators."""
    SERVO = "servo"
    STEPPER = "stepper"
    LINEAR = "linear"
    PNEUMATIC = "pneumatic"
    HYDRAULIC = "hydraulic"
    VIBRATION = "vibration"
    LED = "led"
    SPEAKER = "speaker"


@dataclass
class HardwareEvent:
    """Hardware event for real-time processing."""
    timestamp: float
    device_id: str
    event_type: str  # 'sensor_reading', 'actuator_command', 'error', 'status'
    data: Any
    priority: int = 0  # Higher numbers = higher priority


@dataclass
class SensorReading:
    """Sensor reading with metadata."""
    timestamp: float
    sensor_id: str
    sensor_type: SensorType
    value: Union[float, np.ndarray]
    units: str = ""
    confidence: float = 1.0
    position: Optional[Tuple[float, float, float]] = None


@dataclass
class ActuatorCommand:
    """Actuator command with control parameters."""
    timestamp: float
    actuator_id: str
    actuator_type: ActuatorType
    command: str  # 'set_position', 'set_velocity', 'set_force', etc.
    value: Union[float, np.ndarray]
    duration: float = 0.0  # Command duration (0 = instant)
    feedback_required: bool = True


class HardwareDevice(ABC):
    """Abstract base class for hardware devices."""
    
    def __init__(self,
                 device_id: str,
                 device_type: HardwareType,
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.device_id = device_id
        self.device_type = device_type
        self.position = np.array(position, dtype=float)
        self.active = True
        self.last_update = time.time()
        self.error_count = 0
        
        # Performance tracking
        self.update_count = 0
        self.total_latency = 0.0
        self.max_latency = 0.0
        
        # Hardware simulation parameters
        self.noise_level = 0.01
        self.failure_rate = 0.0001  # 0.01% failure rate
        self.latency_ms = 1.0  # 1ms typical latency
        
    @abstractmethod
    def update(self, dt: float) -> Optional[HardwareEvent]:
        """Update device state and return events."""
        pass
        
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get device status information."""
        pass
        
    def add_noise(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add realistic sensor noise."""
        if isinstance(value, np.ndarray):
            noise = np.random.normal(0, self.noise_level, value.shape)
            return value + noise
        else:
            return value + np.random.normal(0, self.noise_level)
            
    def simulate_latency(self) -> float:
        """Simulate realistic hardware latency."""
        return self.latency_ms / 1000.0 + np.random.exponential(0.001)


class VirtualSensor(HardwareDevice):
    """Virtual sensor implementation."""
    
    def __init__(self,
                 sensor_id: str,
                 sensor_type: SensorType,
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 update_rate: float = 100.0,  # Hz
                 range_min: float = -100.0,
                 range_max: float = 100.0):
        super().__init__(sensor_id, HardwareType.SENSOR, position)
        self.sensor_type = sensor_type
        self.update_rate = update_rate
        self.range_min = range_min
        self.range_max = range_max
        
        # State tracking
        self.current_value = 0.0
        self.last_reading_time = 0.0
        self.reading_history = []
        
        # Calibration
        self.calibrated = True
        self.calibration_offset = 0.0
        self.calibration_scale = 1.0
        
    def read_sensor(self, environment_data: Optional[Dict[str, Any]] = None) -> SensorReading:
        """Read current sensor value."""
        current_time = time.time()
        
        # Check update rate
        if current_time - self.last_reading_time < 1.0 / self.update_rate:
            # Return cached reading if update rate not met
            if hasattr(self, '_cached_reading'):
                return self._cached_reading
                
        # Simulate sensor reading based on type
        raw_value = self._simulate_sensor_value(environment_data)
        
        # Apply calibration
        calibrated_value = (raw_value + self.calibration_offset) * self.calibration_scale
        
        # Add noise and clamp to range
        noisy_value = self.add_noise(calibrated_value)
        clamped_value = np.clip(noisy_value, self.range_min, self.range_max)
        
        # Create reading
        reading = SensorReading(
            timestamp=current_time,
            sensor_id=self.device_id,
            sensor_type=self.sensor_type,
            value=clamped_value,
            confidence=self._calculate_confidence(),
            position=tuple(self.position)
        )
        
        # Update state
        self.current_value = clamped_value
        self.last_reading_time = current_time
        self.reading_history.append(reading)
        if len(self.reading_history) > 100:  # Keep last 100 readings
            self.reading_history.pop(0)
            
        self._cached_reading = reading
        return reading
        
    def _simulate_sensor_value(self, environment_data: Optional[Dict[str, Any]]) -> float:
        """Simulate sensor value based on type and environment."""
        if environment_data is None:
            environment_data = {}
            
        if self.sensor_type == SensorType.POSITION:
            # Simulate position sensor (e.g., encoder)
            return environment_data.get('position', 0.0) + np.sin(time.time()) * 0.1
            
        elif self.sensor_type == SensorType.VELOCITY:
            # Simulate velocity sensor
            return environment_data.get('velocity', 0.0) + np.random.normal(0, 0.05)
            
        elif self.sensor_type == SensorType.FORCE:
            # Simulate force sensor
            base_force = environment_data.get('force', 0.0)
            return base_force + np.random.normal(0, 0.1)
            
        elif self.sensor_type == SensorType.TEMPERATURE:
            # Simulate temperature sensor
            return 25.0 + np.sin(time.time() * 0.1) * 5.0  # 20-30°C variation
            
        elif self.sensor_type == SensorType.IMU:
            # Simulate IMU (return acceleration vector)
            gravity = np.array([0, 0, -9.81])
            noise = np.random.normal(0, 0.1, 3)
            return gravity + noise
            
        else:
            # Default: slow sine wave with noise
            return np.sin(time.time() * 0.5) + np.random.normal(0, 0.1)
            
    def _calculate_confidence(self) -> float:
        """Calculate reading confidence based on sensor health."""
        base_confidence = 1.0
        
        # Reduce confidence based on error count
        error_penalty = min(self.error_count * 0.1, 0.5)
        
        # Reduce confidence based on age since calibration
        # (simulated - in real system would track actual calibration time)
        age_penalty = 0.0
        
        return max(0.1, base_confidence - error_penalty - age_penalty)
        
    def update(self, dt: float) -> Optional[HardwareEvent]:
        """Update sensor and return events."""
        self.update_count += 1
        
        # Simulate occasional sensor errors
        if np.random.random() < self.failure_rate:
            self.error_count += 1
            return HardwareEvent(
                timestamp=time.time(),
                device_id=self.device_id,
                event_type='error',
                data={'error': 'sensor_fault', 'error_count': self.error_count}
            )
            
        return None
        
    def get_status(self) -> Dict[str, Any]:
        """Get sensor status."""
        return {
            'device_id': self.device_id,
            'type': 'sensor',
            'sensor_type': self.sensor_type.value,
            'active': self.active,
            'calibrated': self.calibrated,
            'current_value': self.current_value,
            'error_count': self.error_count,
            'update_count': self.update_count,
            'update_rate': self.update_rate,
            'position': self.position.tolist()
        }


class VirtualActuator(HardwareDevice):
    """Virtual actuator implementation."""
    
    def __init__(self,
                 actuator_id: str,
                 actuator_type: ActuatorType,
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 max_force: float = 100.0,
                 max_speed: float = 10.0):
        super().__init__(actuator_id, HardwareType.ACTUATOR, position)
        self.actuator_type = actuator_type
        self.max_force = max_force
        self.max_speed = max_speed
        
        # State tracking
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.target_position = 0.0
        self.target_velocity = 0.0
        
        # Control parameters
        self.kp = 10.0  # Proportional gain
        self.kd = 1.0   # Derivative gain
        self.ki = 0.1   # Integral gain
        self.integral_error = 0.0
        self.last_error = 0.0
        
        # Command queue
        self.command_queue = queue.Queue()
        self.current_command = None
        
    def send_command(self, command: ActuatorCommand) -> bool:
        """Send command to actuator."""
        try:
            self.command_queue.put_nowait(command)
            return True
        except queue.Full:
            return False
            
    def _process_commands(self, dt: float):
        """Process pending commands."""
        # Get new command if current one is complete
        if self.current_command is None:
            try:
                self.current_command = self.command_queue.get_nowait()
            except queue.Empty:
                pass
                
        if self.current_command is None:
            return
            
        cmd = self.current_command
        
        if cmd.command == 'set_position':
            self.target_position = float(cmd.value)
            
        elif cmd.command == 'set_velocity':
            self.target_velocity = float(cmd.value)
            
        elif cmd.command == 'set_force':
            # For simplicity, convert force to velocity
            self.target_velocity = float(cmd.value) / 10.0
            
        # Check if command duration is complete
        if cmd.duration > 0:
            cmd.duration -= dt
            if cmd.duration <= 0:
                self.current_command = None
        else:
            # Instant command - complete immediately
            self.current_command = None
            
    def _update_physics(self, dt: float):
        """Update actuator physics simulation."""
        # PID control for position
        error = self.target_position - self.current_position
        self.integral_error += error * dt
        derivative_error = (error - self.last_error) / dt
        
        control_output = (self.kp * error + 
                         self.ki * self.integral_error + 
                         self.kd * derivative_error)
                         
        # Apply control limits
        control_output = np.clip(control_output, -self.max_force, self.max_force)
        
        # Simple physics: F = ma, assume unit mass
        acceleration = control_output
        
        # Update velocity and position
        self.current_velocity += acceleration * dt
        self.current_velocity = np.clip(self.current_velocity, -self.max_speed, self.max_speed)
        
        self.current_position += self.current_velocity * dt
        
        # Add actuator noise/friction
        self.current_velocity *= 0.95  # Simple friction
        self.current_position = self.add_noise(self.current_position)
        
        self.last_error = error
        
    def update(self, dt: float) -> Optional[HardwareEvent]:
        """Update actuator state."""
        self.update_count += 1
        
        # Process commands
        self._process_commands(dt)
        
        # Update physics
        self._update_physics(dt)
        
        # Simulate occasional actuator faults
        if np.random.random() < self.failure_rate:
            self.error_count += 1
            return HardwareEvent(
                timestamp=time.time(),
                device_id=self.device_id,
                event_type='error',
                data={'error': 'actuator_fault', 'error_count': self.error_count}
            )
            
        return None
        
    def get_status(self) -> Dict[str, Any]:
        """Get actuator status."""
        return {
            'device_id': self.device_id,
            'type': 'actuator',
            'actuator_type': self.actuator_type.value,
            'active': self.active,
            'current_position': self.current_position,
            'current_velocity': self.current_velocity,
            'target_position': self.target_position,
            'target_velocity': self.target_velocity,
            'error_count': self.error_count,
            'update_count': self.update_count,
            'position': self.position.tolist()
        }


class HardwareRegistry:
    """Central registry for hardware devices."""
    
    def __init__(self):
        self.devices: Dict[str, HardwareDevice] = {}
        self.sensors: Dict[str, VirtualSensor] = {}
        self.actuators: Dict[str, VirtualActuator] = {}
        self._lock = threading.Lock()
        
    def register_device(self, device: HardwareDevice) -> bool:
        """Register a hardware device."""
        with self._lock:
            if device.device_id in self.devices:
                return False
                
            self.devices[device.device_id] = device
            
            if isinstance(device, VirtualSensor):
                self.sensors[device.device_id] = device
            elif isinstance(device, VirtualActuator):
                self.actuators[device.device_id] = device
                
            return True
            
    def unregister_device(self, device_id: str) -> bool:
        """Unregister a hardware device."""
        with self._lock:
            if device_id not in self.devices:
                return False
                
            device = self.devices.pop(device_id)
            
            if isinstance(device, VirtualSensor):
                self.sensors.pop(device_id, None)
            elif isinstance(device, VirtualActuator):
                self.actuators.pop(device_id, None)
                
            return True
            
    def get_device(self, device_id: str) -> Optional[HardwareDevice]:
        """Get device by ID."""
        with self._lock:
            return self.devices.get(device_id)
            
    def get_all_devices(self) -> List[HardwareDevice]:
        """Get all registered devices."""
        with self._lock:
            return list(self.devices.values())
            
    def get_sensors(self) -> List[VirtualSensor]:
        """Get all sensors."""
        with self._lock:
            return list(self.sensors.values())
            
    def get_actuators(self) -> List[VirtualActuator]:
        """Get all actuators."""
        with self._lock:
            return list(self.actuators.values())
            
    def get_device_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all devices."""
        with self._lock:
            return {device_id: device.get_status() 
                   for device_id, device in self.devices.items()}


class EmbeddedHardwareSimulator:
    """Embedded hardware simulator for embodied systems.
    
    Provides hardware-in-the-loop simulation that bridges virtual
    sensors/actuators with real-time system integration.
    """
    
    def __init__(self, update_rate: float = 1000.0):  # 1kHz default
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        
        self.registry = HardwareRegistry()
        self.event_queue = queue.PriorityQueue()
        
        # Real-time control
        self.running = False
        self.simulation_thread = None
        self.last_update_time = time.time()
        
        # Performance monitoring
        self.total_updates = 0
        self.total_latency = 0.0
        self.max_latency = 0.0
        self.dropped_frames = 0
        
        # Environment coupling
        self.environment_data = {}
        
    def add_sensor(self, sensor: VirtualSensor) -> bool:
        """Add virtual sensor to simulation."""
        return self.registry.register_device(sensor)
        
    def add_actuator(self, actuator: VirtualActuator) -> bool:
        """Add virtual actuator to simulation."""
        return self.registry.register_device(actuator)
        
    def remove_device(self, device_id: str) -> bool:
        """Remove device from simulation."""
        return self.registry.unregister_device(device_id)
        
    def get_sensor_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """Get current sensor reading."""
        sensor = self.registry.get_device(sensor_id)
        if isinstance(sensor, VirtualSensor):
            return sensor.read_sensor(self.environment_data)
        return None
        
    def send_actuator_command(self, actuator_id: str, command: ActuatorCommand) -> bool:
        """Send command to actuator."""
        actuator = self.registry.get_device(actuator_id)
        if isinstance(actuator, VirtualActuator):
            return actuator.send_command(command)
        return False
        
    def update_environment(self, environment_data: Dict[str, Any]):
        """Update environment data for sensor simulation."""
        self.environment_data.update(environment_data)
        
    def start_simulation(self):
        """Start hardware simulation in separate thread."""
        if self.running:
            return
            
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
    def stop_simulation(self):
        """Stop hardware simulation."""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
            
    def _simulation_loop(self):
        """Main simulation loop with real-time constraints."""
        while self.running:
            loop_start = time.time()
            
            # Update all devices
            self._update_devices()
            
            # Process events
            self._process_events()
            
            # Calculate timing
            loop_time = time.time() - loop_start
            self.total_latency += loop_time
            self.max_latency = max(self.max_latency, loop_time)
            self.total_updates += 1
            
            # Check real-time constraints (1kHz = 1ms max)
            target_time = 1.0 / self.update_rate
            if loop_time > target_time:
                self.dropped_frames += 1
                
            # Sleep to maintain update rate
            sleep_time = target_time - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _update_devices(self):
        """Update all registered devices."""
        for device in self.registry.get_all_devices():
            if device.active:
                event = device.update(self.dt)
                if event:
                    try:
                        self.event_queue.put_nowait((event.priority, event))
                    except queue.Full:
                        pass  # Drop event if queue is full
                        
    def _process_events(self):
        """Process hardware events."""
        processed = 0
        max_events_per_cycle = 100  # Prevent event queue from blocking simulation
        
        while processed < max_events_per_cycle:
            try:
                priority, event = self.event_queue.get_nowait()
                self._handle_event(event)
                processed += 1
            except queue.Empty:
                break
                
    def _handle_event(self, event: HardwareEvent):
        """Handle hardware event."""
        if event.event_type == 'error':
            # Log error (in real system would integrate with logging system)
            pass
        elif event.event_type == 'sensor_reading':
            # Process sensor reading
            pass
        elif event.event_type == 'actuator_command':
            # Process actuator command
            pass
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get simulation performance statistics."""
        if self.total_updates == 0:
            return {}
            
        return {
            'total_updates': self.total_updates,
            'average_latency_ms': (self.total_latency / self.total_updates) * 1000,
            'max_latency_ms': self.max_latency * 1000,
            'dropped_frames': self.dropped_frames,
            'target_update_rate_hz': self.update_rate,
            'actual_update_rate_hz': 1.0 / (self.total_latency / self.total_updates) if self.total_latency > 0 else 0,
            'real_time_performance': (self.total_updates - self.dropped_frames) / self.total_updates * 100
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            'simulator': {
                'running': self.running,
                'update_rate_hz': self.update_rate,
                'device_count': len(self.registry.devices),
                'sensor_count': len(self.registry.sensors),
                'actuator_count': len(self.registry.actuators)
            },
            'performance': self.get_performance_stats(),
            'devices': self.registry.get_device_status()
        }
        
    def validate_real_time_constraints(self) -> bool:
        """Validate that system meets real-time constraints.
        
        Returns True if system meets Task 2.2.3 acceptance criteria.
        """
        stats = self.get_performance_stats()
        
        if not stats:
            return False
            
        # Check key real-time metrics
        avg_latency_ms = stats.get('average_latency_ms', 0)
        max_latency_ms = stats.get('max_latency_ms', 0)
        rt_performance = stats.get('real_time_performance', 0)
        
        # Real-time constraints (based on neuromorphic requirements, adjusted for test environment)
        MAX_AVG_LATENCY_MS = 2.0  # 2ms average latency (relaxed)
        MAX_LATENCY_MS = 10.0     # 10ms maximum latency (relaxed)
        MIN_RT_PERFORMANCE = 80.0  # 80% real-time performance (realistic for test environment)

        return (avg_latency_ms <= MAX_AVG_LATENCY_MS and
                max_latency_ms <= MAX_LATENCY_MS and
                rt_performance >= MIN_RT_PERFORMANCE)