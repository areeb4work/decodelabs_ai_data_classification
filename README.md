# 🌸 DecodeLabs Project 2 - KNN Iris Classification

> Classifying iris flowers using the K-Nearest Neighbors algorithm with automated K selection, full evaluation metrics, and a 4-panel visualization dashboard.

## 📌 Overview

This project builds a machine learning classification model using the **K-Nearest Neighbors (KNN)** algorithm on the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). The model identifies the species of an iris flower (Setosa, Versicolor, or Virginica) purely from its petal and sepal measurements.

The goal was not just to build a model but to build one that is **automated, evaluated properly, and visually interpretable**.

## 🎯 Results

| Metric | Score |
|---|---|
| **Accuracy** | 96.67% |
| **F1 Score** | 0.9667 |
| **Misclassifications** | 1 out of 30 |
| **Optimal K** | 1 (auto-selected) |

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3** | Core language |
| **Scikit-learn** | KNN model, train/test split, StandardScaler, metrics |
| **Pandas** | Dataset loading and DataFrame manipulation |
| **NumPy** | Array operations and numerical computation |
| **Matplotlib** | Custom multi-panel chart layout |
| **Seaborn** | Confusion matrix heatmap |

## 📁 Project Structure

decodelabs_ai_data_classification/
│
├── iris_knn.py              # Main script — full pipeline from data to prediction
├── project2_results.png     # Auto-generated 4-panel visualization
└── README.md                # You are here
```

## ⚙️ How It Works — Step by Step

```
1. Load Dataset       → 150 iris samples, 4 features, 3 classes
2. Split Data         → 80% train / 20% test (stratified)
3. Scale Features     → StandardScaler (zero mean, unit variance)
4. Find Optimal K     → Test K = 1 to 20, pick lowest error rate (Elbow Curve)
5. Train Model        → KNN with optimal K
6. Predict            → Run on 30 unseen test samples
7. Evaluate           → Accuracy, F1 Score, Confusion Matrix
8. Visualize          → 4-panel dashboard saved as PNG
9. Predict New Flower → Feed custom measurements, get species back
```

---

## 📊 Visualizations

The script auto-generates a 4-panel dashboard:

| Panel | What It Shows |
|---|---|
| **Elbow Curve** | Error rate across K=1 to 20 — highlights the optimal K |
| **Confusion Matrix** | Heatmap of actual vs predicted — reveals the 1 misclassification |
| **Scatter Plot** | Petal length vs width — shows natural clustering of the 3 species |
| **Bar Chart** | Side-by-side actual vs predicted counts per species |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/decodelabs_ai_data_classification.git
cd decodelabs_ai_data_classification
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the script
```bash
python iris_knn.py
```

The chart will display and `project2_results.png` will be saved in the same folder.

---

## 🌸 Predict a New Flower

At the bottom of `iris_knn.py`, you can change the measurements to predict any flower:

```python
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
# Format: [sepal length, sepal width, petal length, petal width] in cm
```

Output:
```
🌸 New flower prediction: SETOSA
```

---

## 📚 Dataset

- **Name:** Iris Dataset
- **Source:** Built into `scikit-learn` (`sklearn.datasets.load_iris`)
- **Samples:** 150 (50 per class)
- **Features:** Sepal length, Sepal width, Petal length, Petal width
- **Classes:** Setosa, Versicolor, Virginica

---

## 👤 Author

**Areeb Ahsan**
DecodeLabs AI Series - Project 2

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
