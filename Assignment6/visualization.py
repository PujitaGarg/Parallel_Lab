import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    'ArraySize': [50_000, 500_000, 5_000_000, 50_000_000],
    'Time(ms)': [0.022, 0.049, 0.275, 2.781]
}
df = pd.DataFrame(data)
df['Throughput (G elements/s)'] = (df['ArraySize'] / (df['Time(ms)'] / 1000)) / 1e9

plt.figure(figsize=(12, 6))

# Throughput plot
plt.subplot(1, 2, 1)
plt.plot(df['ArraySize'], df['Throughput (G elements/s)'], 'bo-', markersize=8)
plt.xscale('log')
plt.xticks([5e4, 5e5, 5e6, 5e7], ['50K', '500K', '5M', '50M'])
plt.title('RTX 3050 Throughput Scaling\n(Square Root Kernel)')
plt.xlabel('Array Size')
plt.ylabel('Throughput (G elements/s)')
plt.grid(True, which="both", ls="--")

# Time scaling plot
plt.subplot(1, 2, 2)
plt.plot(df['ArraySize'], df['Time(ms)'], 'r^-', markersize=8)
plt.xscale('log')
plt.yscale('log')
plt.xticks([5e4, 5e5, 5e6, 5e7], ['50K', '500K', '5M', '50M'])
plt.title('Execution Time Scaling')
plt.xlabel('Array Size')
plt.ylabel('Time (ms, log scale)')
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig('gpu_performance_analysis.png', dpi=300)
plt.show()
