import json
import pandas as pd
import re
from pathlib import Path

# Target models to analyze
TARGET_MODELS = {
    'edsrnet': 'runs/AR-SW-10M-edsrnet',
    'UformerLite': 'runs/AR-ShallowWater-10M-UformerLite-model_UformerLite-s2025-20251221',
    'uno': 'runs/AR-SW-10M-uno',
    'UNet': 'runs/AR-SW-10M-UNet-model_UNet-s2025-20251228',
    'nafnet': 'runs/AR-SW-10M-nafnet'
}

def extract_from_log(log_path):
    """Extracts Params, FLOPs, and Latency from training.log"""
    info = {'params': 0, 'flops': 0, 'latency': 0}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
            # Extract Params (e.g., "模型参数量: 1,219,841")
            params_match = re.search(r"模型参数量:\s*([\d,]+)", log_content)
            if params_match:
                info['params'] = float(params_match.group(1).replace(',', '')) / 1e6 # Convert to M
                
            # Extract FLOPs (e.g., "FLOPs=19.950G")
            flops_match = re.search(r"FLOPs=([\d.]+)G", log_content)
            if flops_match:
                info['flops'] = float(flops_match.group(1))
                
            # Extract Latency (e.g., "延迟=2.13±1.45ms")
            latency_match = re.search(r"延迟=([\d.]+)±", log_content)
            if latency_match:
                info['latency'] = float(latency_match.group(1))
                
    except Exception as e:
        print(f"Error reading log {log_path}: {e}")
    return info

def update_csv():
    csv_path = "analysis_report_ar_sw_10m.csv"
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Iterate through ALL rows in the dataframe instead of a fixed dictionary
    for idx, row in df.iterrows():
        dir_name = row['Directory']
        model_name = row['Model']
        run_dir = Path("runs") / dir_name
        
        log_path = run_dir / "training.log"
        if log_path.exists():
            # print(f"Extracting info for {model_name} from {log_path}...")
            info = extract_from_log(log_path)
            
            # Update dataframe directly using index
            df.at[idx, 'Params (M)'] = info['params']
            df.at[idx, 'FLOPs (G)'] = info['flops']
            df.at[idx, 'Latency (ms/step)'] = info['latency']
        else:
            # Try to find log in project root relative path if not in runs/
            # Some directories might be full paths or relative to project root
            pass

    # Reorder columns to put new metrics near the front
    cols = list(df.columns)
    # Move Latency (ms/step) after FLOPs
    if 'Latency (ms/step)' in cols:
        cols.remove('Latency (ms/step)')
        insert_idx = cols.index('FLOPs (G)') + 1
        cols.insert(insert_idx, 'Latency (ms/step)')
    
    df = df[cols]
    df.to_csv(csv_path, index=False)
    print(f"✅ Updated {csv_path} with precise resource metrics.")
    
    # Print the updated rows for verification
    print("\nUpdated Rows (Top 5):")
    print(df[df['Model'].isin(TARGET_MODELS.keys())][['Model', 'Params (M)', 'FLOPs (G)', 'Latency (ms/step)', 'Test Loss']].to_string(index=False))

if __name__ == "__main__":
    update_csv()
