import numpy as np
import matplotlib.pyplot as plt

# Load the confusion matrix from the npy file


import json
import pandas as pd
import matplotlib.pyplot as plt
output_path = '/home/aih/gizem.mert/ml_24/ms_24/results_deit/table.png'

# Load the JSON file
with open('/home/aih/gizem.mert/ml_24/ms_24/results_deit/test_metrics.json', 'r') as f:
    metrics_data = json.load(f)

# Define model names manually (ensure order matches the JSON file)
model_names = [
    "DEIT-TINY"
]

# Create a detailed table
table_data = []
for model_name, model_metrics in zip(model_names, metrics_data):
    for class_name, class_metrics in model_metrics["class_metrics"].items():
        table_data.append({
            "Model": model_name,
            "Class": class_name,
            "Precision": f"{class_metrics.get('precision', 0):.3f}",
            "Recall": f"{class_metrics.get('recall (sensitivity)', 0):.3f}",
            "F1-Score": f"{class_metrics.get('f1_score', 0):.3f}"
        })

# Convert to a Pandas DataFrame
df = pd.DataFrame(table_data)

# Visualize the table using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))

plt.savefig(output_path, bbox_inches='tight', dpi=300)


