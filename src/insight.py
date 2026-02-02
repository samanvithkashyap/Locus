import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import cv2
import sys

# configuration
ENCODING_FILE = "outputs/encodings.pickle"
OUTPUT_IMAGE = "outputs/cluster_audit.png"
DATASET_PATH = "dataset"

def audit_dataset():
    if not os.path.exists(ENCODING_FILE):
        print(f"{ENCODING_FILE} not found")
        sys.exit()

    with open(ENCODING_FILE, "rb") as f:
        db = pickle.loads(f.read())

    encodings = np.array(db["encodings"])
    names = np.array(db["names"])
    filenames = np.array(db["filenames"])

    print(f"[INFO] Analyzing {len(encodings)} face vectors...")

    n_samples = len(encodings)
    perp = min(30, n_samples - 1)
    if n_samples < 2:
        print("[ERROR] Need at least 2 images to plot.")
        return

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(encodings)

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 8)) # Standard size again
    
    unique_names = list(set(names))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_names)))
    color_map = {name: color for name, color in zip(unique_names, colors)}
    
    for name in unique_names:
        indices = np.where(names == name)[0]
        ax.scatter(X_2d[indices, 0], X_2d[indices, 1], 
                   label=name, color=color_map[name], s=100, alpha=0.8, picker=True)

    plt.title("Interactive Audit: Click a dot to see the image")
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- SMART LEGEND ---
    # loc='best': Finds the empty corner automatically
    # fontsize='x-small': Keeps it tiny
    # framealpha=0.6: Semi-transparent
    plt.legend(loc='best', fontsize='x-small', framealpha=0.6)
    plt.tight_layout()

    # --- SAVE THE IMAGE ---
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"[SUCCESS] Graph saved to {OUTPUT_IMAGE}")

    # --- INTERACTIVE CLICK ---
    def on_pick(event):
        try:
            if event.mouseevent.xdata is None or event.mouseevent.ydata is None: return
            click_x = event.mouseevent.xdata
            click_y = event.mouseevent.ydata
            
            distances = np.linalg.norm(X_2d - np.array([click_x, click_y]), axis=1)
            idx = np.argmin(distances)
            
            found_name = names[idx]
            found_file = filenames[idx]
            
            print(f"\n[CLICKED] Person: {found_name} | File: {found_file}")
            img_path = os.path.join(DATASET_PATH, found_name, found_file)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    if img.shape[0] > 800:
                        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow(f"PREVIEW: {found_name}", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print(f"   [ERROR] Could not find file: {img_path}")
        except Exception as e:
            print(f"Error: {e}")

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == "__main__":
    audit_dataset()