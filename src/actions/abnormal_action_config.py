# Placeholder list of actions considered abnormal for public space surveillance
# These are examples and should be customized based on the specific use case
# and the available labels in KINETICS_400_LABELS.

ABNORMAL_ACTION_LABELS = [
    "fighting",             # Example: physical altercation
    "vandalism",            # Example: property damage (if your model can detect this accurately)
    "stealing",             # Example: theft (if detectable)
    "pointing gun",         # Example: clear threat
    "explosion",            # Example: if applicable to model outputs
    "road accident",        # Example: if a relevant action class exists
    "pickpocketing",
    "robbery",
    "arguing",              # Less severe, but could be a precursor
    "falling",              # Could indicate an accident or health issue
    # Add more KINETICS_400_LABELS that are relevant to your definition of abnormal
    # For example, if Kinetics has labels like "shouting", "running fast in crowd", etc.
    # you would add them here if they are considered abnormal.
]

# Configuration for abnormal actions, their base severities, and duration thresholds for warnings.

# Define base severity for actions. Higher is more severe.
# 0: Not abnormal
# 1: Low severity abnormal action
# 2: Medium severity abnormal action
# 3: High severity abnormal action
# Customize these based on KINETICS_400_LABELS and your specific use case.
ABNORMAL_ACTION_SEVERITIES = {
    "fighting": 3,          # High severity
    "pointing gun": 3,      # High severity
    "robbery": 3,           # High severity
    "explosion": 3,         # High severity
    "vandalism": 2,         # Medium severity
    "stealing": 2,          # Medium severity
    "pickpocketing": 2,     # Medium severity
    "road accident": 2,     # Medium severity
    "falling": 2,           # Medium severity (could be health issue or minor incident)
    "arguing": 1,           # Low severity (precursor, needs observation)
    # Add more actions and their base severities here
    # e.g., "shouting": 1,
    #       "running fast in crowd": 1, (if considered abnormal and detectable)
}

# Duration thresholds (in seconds) for escalating warning levels.
# A warning is triggered if an abnormal action's duration exceeds these thresholds.
# The final warning level can also be influenced by the action's base severity.

# Example: Action with base severity 1
# - Duration > DURATION_THRESHOLD_LOW_SEVERITY_WARN1: Warning Level 1
# Action with base severity 2
# - Duration > DURATION_THRESHOLD_MEDIUM_SEVERITY_WARN1: Warning Level 1 (or 2 if desired)
# - Duration > DURATION_THRESHOLD_MEDIUM_SEVERITY_WARN2: Warning Level 2 (or 3 if desired)
# Action with base severity 3
# - Duration > DURATION_THRESHOLD_HIGH_SEVERITY_WARN1: Warning Level 2 (or 3 if desired)
# - Duration > DURATION_THRESHOLD_HIGH_SEVERITY_WARN2: Warning Level 3

# For simplicity, let's define thresholds that lead to specific warning levels (1, 2, 3)
# The AnomalyScorer will combine base severity with these.

# Threshold for any abnormal action to trigger a Level 1 Warning (basic alert)
DURATION_THRESHOLD_WARN_LEVEL_1 = 3.0  # seconds

# Threshold for a medium or high severity abnormal action to escalate to Level 2 Warning
DURATION_THRESHOLD_WARN_LEVEL_2 = 7.0  # seconds

# Threshold for a high severity abnormal action to escalate to Level 3 Warning (Critical)
DURATION_THRESHOLD_WARN_LEVEL_3 = 10.0 # seconds

# Alternatively, a simpler model: one threshold, and warning level is base_severity if duration > threshold
# ABNORMAL_DURATION_THRESHOLD_SECONDS = 5.0 (old way)

# You might also want to define thresholds or other parameters here in the future
ABNORMAL_DURATION_THRESHOLD_SECONDS = 5.0 