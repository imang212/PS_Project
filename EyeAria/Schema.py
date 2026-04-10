import json
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class ModuleData:
    """Holds the state and data for a single hardware input."""
    name: str
    is_new: bool
    data: Dict

@dataclass
class PipelinePayload:
    """The Namespaced Envelope passed through the pipeline."""
    timestamp: float
    modules: Dict[str, ModuleData] = field(default_factory=dict)

    def copy(self) -> 'PipelinePayload':
        """Deep copies the metadata while preserving memory by passing frames by reference."""
        frame_refs = {}
        
        # 1. Temporarily extract frames from all nested modules
        for mod_name, mod in self.modules.items():
            if isinstance(mod.data, dict) and 'frame' in mod.data:
                frame_refs[mod_name] = mod.data['frame']
                mod.data['frame'] = None
                
        # 2. Safely deepcopy the lightweight metadata
        cloned = copy.deepcopy(self)
        
        # 3. Restore the frame references to both the original and the clone
        for mod_name, frame in frame_refs.items():
            self.modules[mod_name].data['frame'] = frame
            cloned.modules[mod_name].data['frame'] = frame
            
        return cloned

    def to_json(self) -> str:
        d = asdict(self)
        # Recursively scrub frames from the dict before JSON serialization
        if 'modules' in d:
            for mod in d['modules'].values():
                if isinstance(mod.get('data'), dict):
                    mod['data'].pop('frame', None)
        return json.dumps(d)