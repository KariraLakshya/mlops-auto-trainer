import json
import os
import sys
import jsonschema

# 1. Define the Rules (Schema)
# This acts as the "Contract" that the user must follow.
SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {"type": "string"},
        "problem_type": {"type": "string", "enum": ["tabular", "nlp", "vision"]},
        "train_script": {"type": "string", "minLength": 1},
        "eval_script": {"type": "string", "minLength": 1},
        "data_path": {"type": "string", "minLength": 1},
        "model_output_path": {"type": "string", "minLength": 1},
        "requirements": {"type": "string"}
    },
    # These fields MUST be present
    "required": ["train_script", "eval_script", "data_path", "model_output_path"],
    "additionalProperties": True # Allow extra fields (like 'registry_path')
}

def validate():
    manifest_path = "mlmanifest.json"
    
    # Check 1: Does the file exist?
    if not os.path.exists(manifest_path):
        print(" CRITICAL ERROR: 'mlmanifest.json' is missing from the root directory.")
        sys.exit(1)

    try:
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            
        # Check 2: Does it follow the Schema?
        jsonschema.validate(instance=data, schema=SCHEMA)
        
        # Check 3: Do the referenced files actually exist?
        # This prevents the "File Not Found" crash later in the pipeline
        files_to_check = [
            ("train_script", data.get("train_script")),
            ("eval_script", data.get("eval_script")),
            ("data_path", data.get("data_path")),
            ("requirements", data.get("requirements"))
        ]
        
        for name, path in files_to_check:
            if path: # If field is defined
                if not os.path.exists(path):
                    print(f" PATH ERROR: Field '{name}' points to '{path}', but that file does not exist.")
                    sys.exit(1)

        print(" Manifest Validation Passed. Configuration is healthy.")
        sys.exit(0)

    except json.JSONDecodeError:
        print(" JSON ERROR: 'mlmanifest.json' is not valid JSON. Check for missing commas or quotes.")
        sys.exit(1)
    except jsonschema.ValidationError as e:
        print(f" CONFIG ERROR: {e.message}")
        print(f"   -> Location: {e.json_path}")
        sys.exit(1)

if __name__ == "__main__":
    validate()