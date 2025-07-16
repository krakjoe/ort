import pandas as pd
import matplotlib.pyplot as plt
import os

csv_dir = "csv"
out_dir = "images"

for fname in os.listdir(csv_dir):
    if not fname.endswith(".csv"):
        continue

    # Expect file names like php.func.csv or py.func.csv
    parts = fname.split('.')
    if len(parts) < 3:
        continue
    func_name = parts[1].replace('.csv', '')

    # Load both PHP and Python CSVs for this function
    php_path = os.path.join(csv_dir, f"php.{func_name}.csv")
    py_path = os.path.join(csv_dir, f"py.{func_name}.csv")
    if not (os.path.exists(php_path) and os.path.exists(py_path)):
        continue

    php = pd.read_csv(php_path)
    py = pd.read_csv(py_path)

    # Filter for single core count (assume only one value present)
    core_count = php['cores'].unique()[0]
    php = php[php['cores'] == core_count]
    py = py[py['cores'] == core_count]

    # Plot trimmed_mops vs size
    plt.figure(figsize=(10, 6))
    plt.bar(php['size'] - 50, php['trimmed_mops'], width=100, label='PHP/ort', alpha=0.7)
    plt.bar(py['size'] + 50, py['trimmed_mops'], width=100, label='Python/numpy', alpha=0.7)
    plt.xlabel('Tensor Size')
    plt.ylabel('Mean Mop/Sec (Million Operations per Second)')
    plt.title(f'Performance Comparison: {func_name}')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{func_name}.png")
    plt.savefig(out_path)
    plt.close()