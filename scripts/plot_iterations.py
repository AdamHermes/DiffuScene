import matplotlib.pyplot as plt
import numpy as np

# Data from collision_metrics_report.md
labels = ['1', '2-5', '6-10', '11-50']

bedroom_pct = [37.7, 62.3, 0.0, 0.0]
livingroom_pct = [27.1, 47.9, 17.7, 6.8]
diningroom_pct = [27.7, 48.6, 11.9, 9.0]

n_bedroom = 162
n_livingroom = 192
n_diningroom = 177

# Pastel colors for the 3 rooms
colors = ['#ADD8E6', '#FFDAB9', '#C1E1C1'] # light blue, peach/light orange, light green

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Convergence speed of the SAT resolver', fontsize=14, y=1.05)

data = [
    (bedroom_pct, f'Bedroom (n={n_bedroom})', colors[0]),
    (livingroom_pct, f'Living Room (n={n_livingroom})', colors[1]),
    (diningroom_pct, f'Dining Room (n={n_diningroom})', colors[2])
]

for i, (ax, (pct, title, color)) in enumerate(zip(axes, data)):
    bars = ax.bar(labels, pct, color=color, edgecolor='white', width=0.6)
    
    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xlabel('Iterations to convergence', fontsize=10)
    
    if i == 0:
        ax.set_ylabel('Scenes (%)', fontsize=10)
    
    ax.set_ylim(0, 100)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(colors='#333333')
    
    # Add percentage text on top of bars
    for bar, val in zip(bars, pct):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=9, color='#333333')

plt.tight_layout()
plt.savefig('iteration_histogram_3panel.png', dpi=300, bbox_inches='tight')
print("Saved chart to iteration_histogram_3panel.png")
