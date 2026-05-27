import re

with open('scripts/visualize_gedai_weibo.py', 'r') as f:
    content = f.read()

# 1. Replace SSI product/arithmetic mean with geometric mean
content = content.replace(
    "return float(np.mean(S))   # mean cosine, range [0, 1]",
    "return float(np.prod(S) ** (1.0 / len(S)))   # geometric mean matches MATLAB SENSAI_visualization"
)

# 2. Add custom_1d_silhouette
custom_sil = """def custom_1d_silhouette(x, y, target_class=1):
    idx_target = np.where(y == target_class)[0]
    idx_other = np.where(y != target_class)[0]
    n_target = len(idx_target)
    n_other = len(idx_other)
    
    if n_target <= 1 or n_other == 0:
        return np.nan
        
    x_target = x[idx_target]
    x_other = x[idx_other]
    sil_scores = np.zeros(n_target)
    
    for i in range(n_target):
        a_i = np.sum((x_target[i] - x_target)**2) / (n_target - 1)
        b_i = np.sum((x_target[i] - x_other)**2) / n_other
        max_val = max(a_i, b_i)
        if max_val == 0:
            sil_scores[i] = 0
        else:
            sil_scores[i] = (b_i - a_i) / max_val
            
    return float(np.mean(sil_scores))
"""

if "def custom_1d_silhouette" not in content:
    content = content.replace("from sklearn.metrics import silhouette_score", "from sklearn.metrics import silhouette_score\n\n" + custom_sil)

# 3. Replace silhouette calculation
content = content.replace(
    "sil_score = silhouette_score(X, y) if len(np.unique(y)) > 1 else 0.0",
    "sil_score = custom_1d_silhouette(X.flatten(), y, target_class=1)"
)

# 4. Replace SSI title in plot
content = content.replace(
    "SSI Silhouette Score",
    "SSI Silhouette Score"
)

with open('scripts/visualize_gedai_weibo.py', 'w') as f:
    f.write(content)

