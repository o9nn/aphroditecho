"""
Embodied AI Components for 4E Framework

This module implements the virtual body representation for embodied agents
as part of the Deep Tree Echo Phase 2.1.1 implementation.
"""

from .virtual_body import VirtualBody, BodyJoint, BodySchema
from .embodied_agent import EmbodiedAgent
from .proprioception import ProprioceptiveSystem

__all__ = [
    'VirtualBody',
    'BodyJoint', 
    'BodySchema',
    'EmbodiedAgent',
    'ProprioceptiveSystem'
]