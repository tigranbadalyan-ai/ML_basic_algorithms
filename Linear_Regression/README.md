# 📈 Linear Regression: Deep Dive into the Analytical Solution

This directory contains a complete implementation of **Linear Regression** using the Closed-Form solution (Normal Equation). Unlike iterative methods, this approach solves for the optimal weights in a single mathematical "strike."

---

## 📐 Mathematical Derivation (My Notes)

The goal of Linear Regression is to find a vector of weights $W$ that minimizes the difference between our predictions and the actual values.

### 1. Model Representation
We represent our data as a **Feature Matrix** $X$ and our targets as a **Response Vector** $y$:
* **The Bias Trick:** We augment $X$ with a column of ones to account for the intercept ($w_0$).
$$y = XW + \epsilon$$

### 2. Minimizing the Loss (RSS)
We minimize the **Residual Sum of Squares (RSS)**, which is the squared Euclidean norm of the error:
$$RSS(W) = \|y - XW\|^2 = (y - XW)^T (y - XW)$$

Expanding this expression:
$$L(W) = y^T y - 2W^T X^T y + W^T X^T X W$$

### 3. Finding the Optimum
To find the minimum, we take the partial derivative with respect to $W$ and set it to zero:
$$\nabla_W L(W) = -2X^T y + 2X^T X W = 0$$
$$X^T X W = X^T y$$

If the matrix $X^T X$ is invertible ($\det \neq 0$), the optimal weights are:
$$W = (X^T X)^{-1} X^T y$$

---

## 📉 Visualizing the Fit
Linear regression attempts to fit a "Best Fit Line" (or hyperplane in higher dimensions) that minimizes the vertical distance between the data points and the line.

![Linear Regression Concept](https://upload.wikimedia.org/wikipedia/commons/b/be/Normdist_regression.png)
*Red line = Prediction, Blue dots = Actual data.*

---

## ⚠️ Challenges & Advanced Topics

### 1. Overfitting vs. Underfitting
* **Underfitting:** The model is too simple to capture the trend (e.g., using a straight line for curved data).
* **Overfitting:** The model is too complex and "memorizes" the noise in the training data. This often happens when weights ($W$) become extremely large.

### 2. Regularization (Solving $\det = 0$)
What if $X^T X$ is not invertible? Or what if we want to prevent overfitting? We use **Regularization**:

* **Ridge Regression (L2):** Adds a penalty term $\lambda \|W\|^2$.
    * Formula: $W = (X^T X + \lambda I)^{-1} X^T y$
    * *Effect:* Shrinks coefficients towards zero and ensures the matrix is always invertible.
* **Lasso Regression (L1):** Adds a penalty $\lambda |W|$.
    * *Effect:* Can set some coefficients to exactly zero, performing feature selection.

### 3. Numerical Stability
In my implementation, I use the **Moore-Penrose Pseudoinverse** (`np.linalg.pinv`). This is a more robust way to compute $(X^T X)^{-1}$ that handles singular matrices without crashing the code.

---

## 🚀 Implementation Highlights
* **Zero Loops:** Using NumPy's vectorized operations for speed.
* **Separation of Concerns:** Distinct `fit` (training) and `predict` (inference) methods.
* **Comparison:** Validated against `sklearn.linear_model.LinearRegression` to ensure $100\%$ mathematical parity.
