import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../results/tsp_benchmark.csv')

df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

palette = {
    'BruteForce': '#2c3e50',    # Dark Blue/Grey
    'HeldKarp': '#2980b9',      # Blue
    'NearestNeigh': '#e67e22',   # Orange
    'NN_MultiStart': '#d35400', # Dark Orange
    'Christofides': '#c0392b'   # Red
}

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='n', y='time_ms', hue='algorithm', 
             palette=palette, marker='o', errorbar='sd')

plt.yscale('log')
plt.title('Algorithm Runtime Scalability (Log Scale)', fontsize=15)
plt.xlabel('Number of Cities (n)')
plt.ylabel('Execution Time (ms) - Log Scale')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('runtime_scalability.png')
plt.show()

approx_df = df[(df['type'] == 'approx') & (df['ratio'].notna())]

plt.figure(figsize=(10, 6))
sns.lineplot(data=approx_df, x='n', y='ratio', hue='algorithm', 
             palette=palette, marker='s', errorbar='sd')

plt.axhline(1.0, ls='--', color='black', alpha=0.5, label='Optimal (1.0)')
plt.title('Approximation Ratio vs. Problem Size', fontsize=15)
plt.xlabel('Number of Cities (n)')
plt.ylabel('Ratio (Cost / Optimal)')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('approximation_ratio.png')
plt.show()

df_100 = df[df['n'] == 100]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_100, x='algorithm', y='cost', palette=palette, hue='algorithm', legend=False)
sns.stripplot(data=df_100, x='algorithm', y='cost', color=".3", size=5, alpha=0.5)

plt.title('Cost Variance and Distribution (n=100)', fontsize=15)
plt.xlabel('Algorithm')
plt.ylabel('Tour Cost')
plt.tight_layout()
plt.savefig('cost_distribution_n100.png')
plt.show()