# src/ml_models/model_builder.py

from typing import Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

class ModelBuilder:
    """
    Factory for building diverse ML pipelines with optional dimensionality reduction.
    Provides:
      - Logistic Regression (linear)
      - Random Forest (bagged trees)
      - Gradient Boosting (boosted trees)
      - SVM with RBF kernel (kernel methods)
    All pipelines apply StandardScaler → (optional PCA) → model.
    """

    @staticmethod
    def build_logistic_regression(
        pca_components: Optional[int] = None,
        C: float = 1.0,
        penalty: str = 'l2',
        random_state: int = 42
    ) -> Pipeline:
        steps = [('scaler', StandardScaler())]
        if pca_components:
            steps.append(('pca', PCA(n_components=pca_components)))
        steps.append((
            'clf',
            LogisticRegression(C=C, penalty=penalty, solver='liblinear', random_state=random_state)
        ))
        return Pipeline(steps)

    @staticmethod
    def build_random_forest(
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        pca_components: Optional[int] = None
    ) -> Pipeline:
        steps = [('scaler', StandardScaler())]
        if pca_components:
            steps.append(('pca', PCA(n_components=pca_components)))
        steps.append((
            'clf',
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        ))
        return Pipeline(steps)

    @staticmethod
    def build_gradient_boosting(
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
        pca_components: Optional[int] = None
    ) -> Pipeline:
        steps = [('scaler', StandardScaler())]
        if pca_components:
            steps.append(('pca', PCA(n_components=pca_components)))
        steps.append((
            'clf',
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        ))
        return Pipeline(steps)

    @staticmethod
    def build_svm(
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        random_state: int = 42,
        pca_components: Optional[int] = None
    ) -> Pipeline:
        steps = [('scaler', StandardScaler())]
        if pca_components:
            steps.append(('pca', PCA(n_components=pca_components)))
        steps.append((
            'clf',
            SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                random_state=random_state
            )
        ))
        return Pipeline(steps)
# Expose module-level builder functions for convenience
build_logistic_regression = ModelBuilder.build_logistic_regression
build_random_forest       = ModelBuilder.build_random_forest
build_gradient_boosting    = ModelBuilder.build_gradient_boosting
build_svm                  = ModelBuilder.build_svm
