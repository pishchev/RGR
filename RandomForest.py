import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifierImpl:
    def __init__(self, n_estimators = 100, max_depth = None, random_state: int=0) -> None:
        self.classifiers = [DecisionTreeClassifier(criterion='gini', max_depth=max_depth) for _ in range(n_estimators)]
        self.subset_size = 0.7
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> any:
        gen = np.random.RandomState(self.random_state)
        subset_size = int(x.shape[0] * self.subset_size)

        for cls in self.classifiers:
            subset_indices = gen.choice(x.shape[0], subset_size)
            x_subset = x[subset_indices, ...]
            y_subset = y[subset_indices, ...]
            cls.fit(x_subset, y_subset)

        return self

    def predict(self, x) -> np.ndarray:
        pred_table = np.zeros((x.shape[0], len(self.classifiers)), dtype=np.int64)
        for i, cls in enumerate(self.classifiers):
            pred_table[..., i] = cls.predict(x)
        result_pred = np.zeros((x.shape[0]), dtype=np.int64)

        for i, pred in enumerate(pred_table):
            clases, counts = np.unique(pred, return_counts=True)
            result_pred[i] = clases[np.argmax(counts)]

        return result_pred

    def score(self, x: np.ndarray, y: np.ndarray):
        preds = self.predict(x)
        return (preds == y).sum() / y.shape[0] 