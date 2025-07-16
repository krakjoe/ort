import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df_php = pd.read_csv('bench_php.csv')
df_py = pd.read_csv('bench_py.csv')

# Plot trimmed mops vs cores for each size (best for visualization)
sizes = sorted(df_php['size'].unique())
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

plt.figure(figsize=(10, 6))
for i, size in enumerate(sizes):
    plt.plot(
        df_php[df_php['size'] == size]['cores'],
        df_php[df_php['size'] == size]['trimmed_mops'],
        marker='o', linestyle='-', color=colors[i % len(colors)],
        label=f'PHP/ort size={size} (trimmed)'
    )
    plt.plot(
        df_py[df_py['size'] == size]['cores'],
        df_py[df_py['size'] == size]['trimmed_mops'],
        marker='x', linestyle='--', color=colors[i % len(colors)],
        label=f'Python/numpy size={size} (trimmed)'
    )

plt.xlabel('Cores')
plt.ylabel('Million operations per second (trimmed mops)')
plt.title('Matrix Math: PHP/ort vs Python/numpy (trimmed mean)')
plt.legend()
plt.grid(True, which='both', linestyle=":")
plt.tight_layout()
plt.savefig('plot.png')
plt.show()
