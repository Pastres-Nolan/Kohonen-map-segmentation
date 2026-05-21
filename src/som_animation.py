import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import imageio.v2 as imageio
import os
import tempfile
from PIL import Image


df       = pd.read_csv('data/customer_data.csv')
data     = df.drop(columns=['segment']).values
segments = df['segment'].values

nb_features = data.shape[1]
rows, cols  = 10, 10

data_min          = np.min(data, axis=0)
data_max          = np.max(data, axis=0)
standardized_data = (data - data_min) / (data_max - data_min)


segment_names = ['young_budget', 'middle_affluent', 'senior_cautious', 'young_professional']
seg_colors    = {
    'young_budget':        '#1D9E75',
    'middle_affluent':     '#7F77DD',
    'senior_cautious':     '#BA7517',
    'young_professional':  '#D85A30',
}
point_colors = [seg_colors[s] for s in segments]


pca     = PCA(n_components=2)
data_2d = pca.fit_transform(standardized_data)   # (250, 2)

def find_bmu(weights, x):
    best_dist = float('inf')
    best_pos  = (0, 0)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            dist = np.sqrt(np.sum((x - weights[i][j])**2))
            if dist < best_dist:
                best_dist = dist
                best_pos  = (i, j)
    return best_pos

def compute_neighbour(weights, bmu, radius):
    rows, cols = weights.shape[0], weights.shape[1]
    gauss = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            grid_dist  = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
            gauss[i,j] = np.exp(-grid_dist**2 / (2 * radius**2))
    return gauss

def update_weight(weights, gauss, lr, x):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i][j] += lr * gauss[i,j] * (x - weights[i][j])
    return weights

def get_weights_2d(weights):
    """Project 5D neuron weights into 2D via PCA."""
    w2 = pca.transform(weights.reshape(-1, nb_features))
    return w2.reshape(rows, cols, 2)

def draw_frame(ax, data_2d, point_colors, weights_2d, epoch, lr, radius):
    ax.clear()

    for i, (x, y) in enumerate(data_2d):
        ax.scatter(x, y, color=point_colors[i], s=18, alpha=0.6, zorder=2)

  
   
    for i in range(rows):
        for j in range(cols - 1):
            x_vals = [weights_2d[i, j, 0],   weights_2d[i, j+1, 0]]
            y_vals = [weights_2d[i, j, 1],   weights_2d[i, j+1, 1]]
            ax.plot(x_vals, y_vals, color='#444', linewidth=0.6, alpha=0.5, zorder=3)
    
    for i in range(rows - 1):
        for j in range(cols):
            x_vals = [weights_2d[i,   j, 0], weights_2d[i+1, j, 0]]
            y_vals = [weights_2d[i,   j, 1], weights_2d[i+1, j, 1]]
            ax.plot(x_vals, y_vals, color='#444', linewidth=0.6, alpha=0.5, zorder=3)

  
    neuron_x = weights_2d[:, :, 0].flatten()
    neuron_y = weights_2d[:, :, 1].flatten()
    ax.scatter(neuron_x, neuron_y, color='white', edgecolors='#333',
               s=22, linewidths=0.8, zorder=4)

   
    handles = [mpatches.Patch(color=seg_colors[s], label=s) for s in segment_names]
    ax.legend(handles=handles, fontsize=7, loc='upper right',
              framealpha=0.9, edgecolor='#ccc')

    ax.set_title(f'SOM training  —  epoch {epoch:03d}   '
                 f'lr={lr:.4f}   radius={radius:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('PCA component 1', fontsize=8)
    ax.set_ylabel('PCA component 2', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor('#f9f9f9')


np.random.seed(42)
weights  = np.random.rand(rows, cols, nb_features)

n_epochs   = 100
lr0        = 0.5
radius0    = 3.0
save_every = 2          

tmpdir  = tempfile.mkdtemp()
frames  = []

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('white')


weights_2d = get_weights_2d(weights)
draw_frame(ax, data_2d, point_colors, weights_2d, 0, lr0, radius0)
plt.tight_layout()
path = os.path.join(tmpdir, 'frame_000.png')
plt.savefig(path, dpi=90)
frames.append(path)
print("epoch   0 — initial random state captured")

for epoch in range(1, n_epochs + 1):
    lr     = lr0     * np.exp(-epoch / n_epochs)
    radius = radius0 * np.exp(-epoch / n_epochs)

    for i in range(len(standardized_data)):
        x     = standardized_data[i]
        bmu   = find_bmu(weights, x)
        gauss = compute_neighbour(weights, bmu, radius)
        weights = update_weight(weights, gauss, lr, x)

    if epoch % save_every == 0 or epoch == n_epochs:
        weights_2d = get_weights_2d(weights)
        draw_frame(ax, data_2d, point_colors, weights_2d, epoch, lr, radius)
        plt.tight_layout()
        path = os.path.join(tmpdir, f'frame_{epoch:03d}.png')
        plt.savefig(path, dpi=90)
        frames.append(path)
        print(f"epoch {epoch:3d} — frame saved  (lr={lr:.4f}, radius={radius:.4f})")

plt.close(fig)
gif_path = 'outputs/som_training.gif'
images   = [imageio.imread(f) for f in frames]

last_frame = images[-1]
images.extend([last_frame] * 40)  
durations = [0.15] * len(images)  




gif_path   = 'outputs/som_training.gif'
pil_images = [Image.open(f).convert('RGBA') for f in frames]

for _ in range(60):
    pil_images.append(pil_images[-1].copy())

pil_images[0].save(
    gif_path,
    save_all      = True,
    append_images = pil_images[1:],
    duration      = 150,   
    loop          = 0
)
print(f"done! → {gif_path}")
