import json
import pandas as pd
import os

def convert_config_to_csv(config_path, output_path):
    # Load the JSON configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Extract the technical indicators
    indicators = config_data.get('technical_indicators', {})
    
    # Prepare data for DataFrame
    data_list = []
    for name, details in indicators.items():
        # Get params if they exist, else None
        params = details.get('params')
        complexity = details.get('complexity')
        
        entry = {
            "indicator_name": name,
            "params": params,         # Keeps the raw structure (list/dict/int)
            "complexity": complexity
        }
        data_list.append(entry)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully saved configuration to {output_path}")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Going up one level from 'Test' folder to root, then down to features
    project_root = os.path.dirname(base_dir) 
    
    json_file_path = os.path.join(project_root, 'financial_TSF', 'features', 'feature_config.json')
    output_csv_path = os.path.join(base_dir, 'feature_config.csv')
    
    if os.path.exists(json_file_path):
        convert_config_to_csv(json_file_path, output_csv_path)
    else:
        print(f"Error: Config file not found at {json_file_path}")