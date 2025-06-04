import json
import os

def load_kinetics_400_classes(file_path="src/kinetics_400_classes.json"):
    try:
        # Construct absolute path to ensure it's found correctly
        # Assuming this script (kinetics_classes.py) is in src/utils
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        absolute_file_path = os.path.join(base_dir, file_path)

        if not os.path.exists(absolute_file_path):
            # Fallback for common case where src is the root for pathing
            absolute_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path.split('/', 1)[-1])
            if not os.path.exists(absolute_file_path):
                 # Try path relative to project root if CWD is project root
                absolute_file_path = file_path 
                if not os.path.exists(absolute_file_path):
                    print(f"Warning: Kinetics classes JSON file not found at {file_path} or derived paths.")
                    # Attempt to search upwards from current file's directory for 'src'
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root_found = False
                    for _ in range(3): # Search up to 3 levels
                        if os.path.basename(current_dir) == 'src':
                            project_root_found = True
                            break
                        parent_dir = os.path.dirname(current_dir)
                        if parent_dir == current_dir: # Reached filesystem root
                            break
                        current_dir = parent_dir
                    
                    if project_root_found:
                         alt_file_path = os.path.join(os.path.dirname(current_dir), file_path) # Path from parent of src
                         if os.path.exists(alt_file_path):
                             absolute_file_path = alt_file_path
                         else:
                            print(f"Also not found at {alt_file_path}")
                            return [f"kinetics_class_{i}" for i in range(400)] # Fallback
                    else:
                        print("Could not reliably determine project root to find kinetics_400_classes.json")
                        return [f"kinetics_class_{i}" for i in range(400)] # Fallback
        
        with open(absolute_file_path, 'r') as f:
            kinetics_classes = json.load(f)
        
        # Create a list where index maps to action name
        class_list = [""] * 400  # Initialize with empty strings
        for action_name, index in kinetics_classes.items():
            if 0 <= index < 400:
                # Remove quotes from action names if present
                clean_name = action_name.strip('"')
                class_list[index] = clean_name
        
        # Fill any empty slots with fallback names
        for i in range(400):
            if not class_list[i]:
                class_list[i] = f"kinetics_class_{i}"
        
        return class_list
    except FileNotFoundError:
        print(f"Error: Kinetics classes JSON file not found at {absolute_file_path}. Using fallback.")
        return [f"kinetics_class_{i}" for i in range(400)] # Fallback
    except json.JSONDecodeError:
        print(f"Error: Could not decode Kinetics classes JSON file at {absolute_file_path}. Using fallback.")
        return [f"kinetics_class_{i}" for i in range(400)] # Fallback
    except Exception as e:
        print(f"An unexpected error occurred while loading Kinetics classes: {e}. Using fallback.")
        return [f"kinetics_class_{i}" for i in range(400)] # Fallback

def get_anomaly_relevant_classes():
    """
    Returns a mapping of Kinetics action IDs to anomaly types.
    
    Anomaly Types:
    1 = Violence (Critical - Level 3)
    2 = Falling (High - Level 2) 
    3 = Suspicious Activity (Medium - Level 1)
    
    Returns:
        dict: {kinetics_action_id: anomaly_type_index}
    """
    return {
        # Violence Actions (Anomaly Type 1 - Critical Level 3)
        259: 1,  # "punching person (boxing)"
        314: 1,  # "slapping"
        395: 1,  # "wrestling"
        345: 1,  # "sword fighting"
        150: 1,  # "headbutting"
        6: 1,    # "arm wrestling" (can escalate)
        258: 1,  # "punching bag" (training, but can indicate aggression)
        302: 1,  # "side kick"
        
        # Falling Actions (Anomaly Type 2 - High Level 2)
        147: 2,  # "gymnastics tumbling"
        122: 2,  # "faceplanting"
        325: 2,  # "somersaulting"
        93: 2,   # "diving cliff"
        172: 2,  # "jumping into pool"
        105: 2,  # "drop kicking"
        
        # Suspicious Activities (Anomaly Type 3 - Medium Level 1)
        13: 3,   # "balloon blowing" (could be suspicious in some contexts)
        153: 3,  # "hitting baseball" (object as weapon)
        174: 3,  # "kicking field goal" (aggressive kicking motion)
        175: 3,  # "kicking soccer ball" (aggressive kicking)
        8: 3,    # "assembling computer" (tampering with equipment)
        23: 3,   # "blasting sand" (destructive activity)
        39: 3,   # "building shed" (unauthorized construction)
        40: 3,   # "bungee jumping" (dangerous activity)
        356: 3,  # "throwing axe" (weapon throwing)
        357: 3,  # "throwing ball" (projectile throwing)
        358: 3,  # "throwing discus" (projectile throwing)
        371: 3,  # "unboxing" (suspicious package handling)
        203: 3,  # "opening bottle" (potential weapon/substance)
        215: 3,  # "planting trees" (unauthorized digging/planting)
        90: 3,   # "digging" (unauthorized excavation)
    }

if __name__ == '__main__':
    # Test loading classes
    classes = load_kinetics_400_classes()
    print(f"Loaded {len(classes)} Kinetics classes.")
    if classes and len(classes) == 400:
        print("First 5 classes:", classes[:5])
        print("Last 5 classes:", classes[-5:])
    else:
        print("Problem loading classes or fallback was used.")

    anomalies = get_anomaly_relevant_classes()
    print(f"Anomaly relevant classes: {anomalies}") 