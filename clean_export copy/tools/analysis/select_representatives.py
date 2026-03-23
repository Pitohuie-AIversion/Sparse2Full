import pandas as pd
import re

def categorize_model(name):
    name = name.lower()
    if any(x in name for x in ['fno', 'uno', 'deeponet', 'operator']):
        return 'Neural Operator'
    elif any(x in name for x in ['former', 'swin', 'vit', 'mixer', 'attention']):
        return 'Transformer / Attention'
    elif any(x in name for x in ['mlp']):
        return 'MLP / Implicit'
    elif any(x in name for x in ['edsr', 'nafnet', 'resnet', 'unet', 'conv', 'bilinear']):
        return 'CNN'
    else:
        return 'Other'

def select_representatives(csv_file):
    df = pd.read_csv(csv_file)
    
    # Add Category column
    df['Category'] = df['Model'].apply(categorize_model)
    
    # Group by Category and find best model (min Test Loss)
    best_models = df.loc[df.groupby('Category')['Test Loss'].idxmin()]
    
    # Sort by Test Loss
    best_models = best_models.sort_values('Test Loss')
    
    print(best_models[['Category', 'Model', 'Test Loss', 'PSNR']].to_string(index=False))

if __name__ == "__main__":
    select_representatives("analysis_report_ar_sw_10m.csv")
