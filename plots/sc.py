import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
# Assuming the file 'sc_benchmark.csv' exists in the directory
df = pd.read_csv('../results/sc_benchmark.csv')

# Preprocessing: Ensure numeric types and handle 'N/A'
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df['optimal'] = pd.to_numeric(df['optimal'], errors='coerce')

# Define harmonic number function for the theoretical bound
def harmonic_number(n):
    return sum(1/i for i in range(1, int(n) + 1))

# Consistent Color Palette
# Exact = Blues/Greens (Cold), Greedy = Reds/Oranges (Warm)
palette = {
    'Exact_BT': '#2980b9', 
    'Greedy': '#e67e22'
}

sns.set_theme(style="whitegrid")

# --- 1. Theoretical vs. Empirical Performance ---
plt.figure(figsize=(10, 6))
greedy_df = df[df['algorithm'] == 'Greedy'].copy()

# Calculate H_n bound for each n_elements
unique_n = sorted(greedy_df['n_elements'].unique())
h_bounds = {n: harmonic_number(n) for n in unique_n}
greedy_df['H_n_bound'] = greedy_df['n_elements'].map(h_bounds)

# Plot actual ratio
sns.lineplot(data=greedy_df, x='n_elements', y='ratio', 
             label='Empirical Ratio', color=palette['Greedy'], marker='o', errorbar='sd')

# Plot theoretical bound
plt.plot(unique_n, [h_bounds[n] for n in unique_n], 
         label='Theoretical Bound (H_n)', color='#c0392b', linestyle='--')

plt.title('Greedy Performance: Empirical Ratio vs. Theoretical Bound', fontsize=14)
plt.ylabel('Ratio (Cover Size / Optimal)')
plt.xlabel('n_elements')
plt.legend()
plt.tight_layout()
plt.savefig('theoretical_vs_empirical.png')
plt.show()

# --- 2. Relative "Optimality Gap" (Bar Chart for n <= 20) ---
# Filter for small instances where optimal is known
gap_df = df[df['n_elements'] <= 20].copy()
gap_df['Instance'] = gap_df['n_elements'].astype(str) + " (S" + gap_df['seed'].astype(str) + ")"

plt.figure(figsize=(12, 6))
sns.barplot(data=gap_df, x='Instance', y='cover_size', hue='algorithm', palette=palette)
plt.xticks(rotation=45)
plt.title('Optimality Gap (Instances n ≤ 20)', fontsize=14)
plt.ylabel('Cover Size (Number of Sets)')
plt.tight_layout()
plt.savefig('optimality_gap.png')
plt.show()

# --- 3. Density Impact: n_subsets vs. Runtime ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='n_subsets', y='time_ms', hue='algorithm', 
             palette=palette, marker='o', errorbar='sd')
plt.yscale('log')
plt.title('Impact of Subset Count on Execution Time', fontsize=14)
plt.ylabel('Time (ms) - Log Scale')
plt.tight_layout()
plt.savefig('density_impact.png')
plt.show()

# --- 4. Convergence of Ratios (Box Plot for Greedy) ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=greedy_df, x='n_elements', y='ratio', color=palette['Greedy'])
plt.title('Stability of Greedy Algorithm Ratio', fontsize=14)
plt.ylabel('Approximation Ratio')
plt.tight_layout()
plt.savefig('ratio_stability.png')
plt.show()

# --- 5. Time-Efficiency Frontier (Scatter Plot) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='time_ms', y='cover_size', hue='algorithm', 
                palette=palette, style='algorithm', s=100, alpha=0.7)

plt.xscale('log')
plt.title('Time-Efficiency Frontier', fontsize=14)
plt.xlabel('Execution Time (ms) - Log Scale')
plt.ylabel('Cover Size')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('efficiency_frontier.png')
plt.show()