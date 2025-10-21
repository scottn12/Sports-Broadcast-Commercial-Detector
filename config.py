"""
NFL Detector Configuration
Configuration parameters for the NFL Transition Detector
"""

# Model Configuration
MODEL_PATH = 'nfl_classifier_best.pth'

# Detection Parameters
DETECTION_CONFIG = {
    'threshold': 0.6,              # Classification threshold (game vs commercial)
    'cooldown_seconds': 0,         # Minimum time between transitions  
    'check_interval': 1.0,         # Check screen every N seconds
    'smoothing_window': 5,         # Average last N predictions
    'auto_mute': True,             # Automatically mute browser during commercials
    'min_confidence': 0.90,        # Minimum confidence to allow transition
    'control_media': True          # Play/pause system media (Spotify, etc.) during transitions
}

# Runtime Configuration  
RUNTIME_CONFIG = {
    'duration': None,              # Runtime limit in seconds (None = run forever)
    'save_transitions': False      # Save screenshots of transitions
}