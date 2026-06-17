# DECODELABS PROJECT 2 - AI DATA CLASSIFICATION OF IRIS DATASET USING KNN ALGORITHM
# Author Areeb Ahsan
# ─────────────────────────────────────────────
# STEP 1 — IMPORT LIBRARIES
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# ─────────────────────────────────────────────
# STEP 2 — LOAD & UNDERSTAND THE DATASET
# ─────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# ─────────────────────────────────────────────
# STEP 3 — SEPARATE FEATURES (X) AND LABEL (y)
# ─────────────────────────────────────────────
X, y = iris.data, iris.target
# ─────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# ─────────────────────────────────────────────
# STEP 5 — FEATURE SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 6 — FIND OPTIMAL K
# ─────────────────────────────────────────────
k_range, error_rates = range(1, 21), []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    error_rates.append(1 - accuracy_score(y_test, knn.predict(X_test_scaled)))

optimal_k = list(k_range)[error_rates.index(min(error_rates))]
# ─────────────────────────────────────────────
# STEP 7 — TRAIN THE FINAL MODEL
# ─────────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train_scaled, y_train)
# ─────────────────────────────────────────────
# STEP 8 — PREDICT
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
# ─────────────────────────────────────────────
# STEP 9 — EVALUATE
# ─────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

# ─────────────────────────────────────────────
# STEP 10 — VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle(
    "DecodeLabs Project 2 — KNN Iris Classification",
    fontsize=16, fontweight="bold"
)

# Reserve top 8% for suptitle, bottom 8% for cut-off prevention
plt.subplots_adjust(
    left=0.07,
    right=0.88,   # leave room for heatmap colorbar
    top=0.88,     # suptitle lives above this line
    bottom=0.10,  # prevents bottom labels being clipped
    hspace=0.55,  # row gap
    wspace=0.40,  # column gap
)

# ── Plot 1: Elbow Curve ──────────────────────────────────────
ax1 = axes[0, 0]
ax1.plot(list(k_range), error_rates, marker="o", color="#1a3a5c", linewidth=2)
ax1.axvline(x=optimal_k, color="#e05a00", linestyle="--", label=f"Optimal K={optimal_k}")
ax1.set_title("Elbow Curve — Finding Optimal K", pad=10, fontsize=11)
ax1.set_xlabel("K Value", labelpad=10)
ax1.set_ylabel("Error Rate", labelpad=10)
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.margins(x=0.05)

# ── Plot 2: Confusion Matrix ─────────────────────────────────
ax2 = axes[0, 1]
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=ax2,
    linewidths=0.5,
    annot_kws={"size": 13},
    cbar_kws={"shrink": 0.70, "pad": 0.05}
)
ax2.set_title("Confusion Matrix", pad=12, fontsize=11)
# Fix: use set_xlabel/ylabel directly (not labelpad trick) and force rotation=0 on ylabel
ax2.set_xlabel("Predicted", labelpad=12)
ax2.set_ylabel("Actual", labelpad=12)
ax2.yaxis.label.set_rotation(90)         # keep "Actual" upright on its axis
ax2.set_xticklabels(iris.target_names, rotation=20, ha="right", fontsize=10)
ax2.set_yticklabels(iris.target_names, rotation=0,  va="center", fontsize=10)

# ── Plot 3: Feature Scatter ──────────────────────────────────
ax3 = axes[1, 0]
colors = {"setosa": "#1a3a5c", "versicolor": "#e05a00", "virginica": "#4a9c6f"}
for species, color in colors.items():
    subset = df[df["species"] == species]
    ax3.scatter(
        subset["petal length (cm)"], subset["petal width (cm)"],
        label=species, color=color, alpha=0.75,
        edgecolors="white", linewidths=0.4, s=65
    )
ax3.set_title("Petal Length vs Petal Width", pad=10, fontsize=11)
ax3.set_xlabel("Petal Length (cm)", labelpad=10)
ax3.set_ylabel("Petal Width (cm)", labelpad=10)
ax3.legend(loc="upper left", framealpha=0.9)
ax3.grid(True, alpha=0.3)
ax3.margins(0.08)

# ── Plot 4: Actual vs Predicted Counts ──────────────────────
ax4 = axes[1, 1]
actual_counts    = [np.sum(y_test == i) for i in range(3)]
predicted_counts = [np.sum(y_pred == i) for i in range(3)]
x, width = np.arange(3), 0.32
bars1 = ax4.bar(x - width/2, actual_counts,    width, label="Actual",    color="#1a3a5c")
bars2 = ax4.bar(x + width/2, predicted_counts, width, label="Predicted", color="#e05a00")
ax4.set_title("Actual vs Predicted Counts", pad=10, fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(iris.target_names, rotation=0)
ax4.set_xlabel("Species", labelpad=10)          # proper x-axis label, not "Predicted"
ax4.set_ylabel("Count", labelpad=10)
ax4.set_ylim(0, max(actual_counts + predicted_counts) + 2)
ax4.legend(loc="upper right", framealpha=0.9)
ax4.grid(True, alpha=0.3, axis="y")

for bar in list(bars1) + list(bars2):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        str(int(bar.get_height())),
        ha="center", va="bottom", fontsize=10
    )

# Use tight_layout AFTER subplots_adjust — rect keeps space for suptitle
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("project2_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved.")

# ─────────────────────────────────────────────
# STEP 11 — PREDICT A NEW FLOWER
# ─────────────────────────────────────────────
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
print(f"\n🌸 New flower prediction: {iris.target_names[prediction[0]].upper()}")

print("\n" + "=" * 60)
print(f"  ✅ PROJECT COMPLETE | Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
print("=" * 60)