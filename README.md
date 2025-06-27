# K-Nearest Neighbors (KNN) – Beginner-Friendly Notes with Analogies & Python Example

---

## 1. What is KNN?

**K-Nearest Neighbors (KNN)** is a **supervised learning algorithm** used for **classification and regression**.

But most commonly, it's used to classify things by asking:

> "What are the labels of the closest items around this one?"

### Analogy:

Imagine you move to a new neighborhood. You don’t know anyone. So you ask: “What do most of my neighbors do for a living?” If most are doctors, you assume people like you in this area are doctors. That’s KNN in action!

Another example: you’re trying to guess someone's favorite ice cream flavor. You check their 3 closest friends (K=3). If 2 love chocolate, you guess this person loves chocolate too.

---

## 2. How KNN Works (Step-by-Step)

1. Pick a number `K` (e.g., 3, 5, 7)
2. Calculate the **distance** from the new point to all other points

   * Common methods: **Euclidean**, **Manhattan**, or **Minkowski** distance
3. Find the **K nearest neighbors**
4. Look at their labels
5. Use **majority vote** (for classification) or **average** (for regression)

### Example:

You want to classify a fruit as an apple or orange based on sweetness and color. You measure distances to all other fruits in your dataset. The 3 nearest fruits are apples → So, your prediction is "apple."

---

## 3. Distance Metrics in KNN

Distance metrics help us **quantify how close** two data points are.

### A. Euclidean Distance (Most common)

```math
\text{distance} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
```

This is the **straight-line distance** between two points in space.

**Analogy**: It’s like drawing a straight line from your house to your friend’s house.

### B. Manhattan Distance

```math
\text{distance} = |x_1 - x_2| + |y_1 - y_2|
```

Also called **city block** distance.

**Analogy**: Walking in a grid-like city (like New York) — you can’t cut diagonally.

### C. Minkowski Distance

A generalization of both Euclidean and Manhattan distances.

```math
\text{distance} = \left(\sum |x_i - y_i|^p\right)^{1/p}
```

* When `p = 1` → Manhattan Distance
* When `p = 2` → Euclidean Distance

**Analogy**: Think of it like a flexible measuring tool where you can tune how sensitive it is.

> KNN works best when distances reflect **real-world similarities** between features. Always **scale your data** before calculating distances!

---

## 4. Choosing the Value of K

### Low K (e.g., K = 1 or 2):

* Very sensitive to noise
* Might overfit

### High K (e.g., K = 20 or 50):

* Might overlook local patterns
* Might underfit

> Usually, **odd values of K** are used to avoid ties
> Use **cross-validation** to find the best K

### Analogy:

* Choosing K is like asking 1 friend vs asking 20 people. One friend may be biased, but 20 might be too generic. Find a sweet spot!

---

## 5. Decision Boundaries

KNN creates **non-linear** decision boundaries. The boundaries change depending on how close a point is to its labeled neighbors.

### Visual Analogy:

Imagine pouring colored ink drops on a paper — blue for class 0 and red for class 1. As the drops spread, they form boundaries. When new points fall near blue, they get classified as blue, and vice versa.

These boundaries get **more wiggly** for smaller K and **smoother** for larger K.

---

## 6. Evaluation Metrics for KNN

To check how well your KNN model is doing:

### Confusion Matrix

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

### Accuracy

```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

### Precision

```math
\text{Precision} = \frac{TP}{TP + FP}
```

### Recall

```math
\text{Recall} = \frac{TP}{TP + FN}
```

### F1 Score

```math
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

---

## 7. Python Code Example – KNN for Classification

We'll use the **Iris dataset** for simplicity.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 8. When to Use KNN

* Simple, intuitive algorithm for classification problems
* Works well with small to medium-sized datasets
* Great for **pattern recognition**, like:

  * Recommender systems
  * Image recognition
  * Customer segmentation

### Downside:

* Slow on large datasets (lazy learner)
* Needs **feature scaling**
* Sensitive to irrelevant features

---
