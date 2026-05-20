import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we are in the correct directory
if os.path.basename(os.getcwd()) != 'digit_model_training':
    if os.path.exists('digit_model_training'):
        os.chdir('digit_model_training')
    else:
        print("Warning: could not change working directory to digit_model_training")
print("Current Working Directory:", os.getcwd())

# Add path for imports
sys.path.append(os.getcwd())

from utils import setup_directories
setup_directories()

print("PyTorch version:", torch.__version__)
print("CUDA GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

from main import run_training_experiment

# Define experiments
experiments = [
    {
        'name': 'EffNetV2_B0_64_Standard_Digits',
        'model_name': 'tf_efficientnetv2_b0',
        'target_size': (64, 64),
        'preprocessing_approach': 'standard',
        'load_full_characters': False,
        'epochs': 3
    },
    {
        'name': 'EffNetV2_B0_64_Bitmask_Digits',
        'model_name': 'tf_efficientnetv2_b0',
        'target_size': (64, 64),
        'preprocessing_approach': 'bitmask',
        'load_full_characters': False,
        'epochs': 3
    },
    {
        'name': 'EffNetV2_B0_32_Bitmask_Digits',
        'model_name': 'tf_efficientnetv2_b0',
        'target_size': (32, 32),
        'preprocessing_approach': 'bitmask',
        'load_full_characters': False,
        'epochs': 3
    },
    {
        'name': 'MobileNetV2_64_Bitmask_Digits',
        'model_name': 'MobileNetV2',
        'target_size': (64, 64),
        'preprocessing_approach': 'bitmask',
        'load_full_characters': False,
        'epochs': 3
    },
    {
        'name': 'EffNetV2_B0_64_Bitmask_Alphanumeric',
        'model_name': 'tf_efficientnetv2_b0',
        'target_size': (64, 64),
        'preprocessing_approach': 'bitmask',
        'load_full_characters': True,
        'epochs': 3
    }
]

results = []

for exp in experiments:
    print(f"\n" + "="*60)
    print(f"RUNNING EXPERIMENT: {exp['name']}")
    print("="*60)
    try:
        res = run_training_experiment(
            model_name=exp['model_name'],
            dataset_path="../Dataset",
            target_size=exp['target_size'],
            preprocessing_approach=exp['preprocessing_approach'],
            load_full_characters=exp['load_full_characters'],
            epochs=exp['epochs'],
            batch_size=32, # Smaller batch size for fast training
            num_workers=0  # num_workers=0 works flawlessly in all environments on Windows
        )
        res['Experiment Name'] = exp['name']
        results.append(res)
    except Exception as e:
        print(f"Failed to run experiment {exp['name']}: {e}")
        import traceback
        traceback.print_exc()

# Convert to DataFrame and Display/Save
df = pd.DataFrame(results)

display_cols = [
    'Experiment Name', 'Model', 'Size', 'Preprocessing', 
    'Alphanumeric', 'Validation Accuracy', 'Test Accuracy', 
    'Model Size (MB)', 'Training Time Formatted'
]

df_display = df[display_cols]
print("\n--- EXPERIMENTS SUMMARY REPORT ---")
print(df_display.to_string(index=False))

# Save results
df_display.to_csv('reports/experiments_summary_report.csv', index=False)
print("Saved summary report to reports/experiments_summary_report.csv")

# Plot Results
plt.figure(figsize=(14, 6))

df_melt = pd.melt(
    df,
    id_vars=['Experiment Name'],
    value_vars=['Validation Accuracy', 'Test Accuracy'],
    var_name='Accuracy Type',
    value_name='Accuracy'
)

df_melt['Accuracy'] = pd.to_numeric(df_melt['Accuracy'], errors='coerce')

sns.barplot(
    data=df_melt,
    x='Experiment Name',
    y='Accuracy',
    hue='Accuracy Type',
    palette='Set2'
)

plt.title('Performance Comparison: Validation vs. Test Accuracy', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Experiment Name', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1.05)
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig('graphs/experiments_comparison.png', dpi=150)
print("Saved performance graph to graphs/experiments_comparison.png")
print("\nAll experiments finished successfully!")
