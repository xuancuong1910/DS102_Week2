import numpy as np
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def calculate_loss(self, y, y_hat):
        epsilon = 1e-9
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        self.theta = np.zeros(X_bias.shape[1])
        
        progress_bar = tqdm(range(self.epochs), desc="Training", leave=True)
        for epoch in progress_bar:
            z = X_bias @ self.theta
            y_hat = self.sigmoid(z)
            
            loss = self.calculate_loss(y, y_hat)
            self.loss_history.append(loss)
            
            gradient = (X_bias.T @ (y_hat - y)) / y.size
            self.theta -= self.lr * gradient
            
            progress_bar.set_postfix({"Loss": f"{loss:.4f}"})

    def predict_proba(self, X):
        """Trả về xác suất ảnh là nhãn 1"""
        X_bias = np.c_[np.ones(X.shape[0]), X]
        z = X_bias @ self.theta
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Trả về nhãn phân loại cuối cùng (0 hoặc 1)"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []

    def softmax(self, z):
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def calculate_loss(self, y_onehot, y_hat):
        epsilon = 1e-9
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_onehot * np.log(y_hat), axis=1))
        return loss

    def fit(self, X, y_onehot):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        num_features = X_bias.shape[1]
        num_classes = y_onehot.shape[1]
        self.theta = np.zeros((num_features, num_classes))
        
        progress_bar = tqdm(range(self.epochs), desc="Training", leave=True)
        for epoch in progress_bar:
            z = X_bias @ self.theta
            y_hat = self.softmax(z)
            
            loss = self.calculate_loss(y_onehot, y_hat)
            self.loss_history.append(loss)
            
            gradient = (X_bias.T @ (y_hat - y_onehot)) / X.shape[0]
            self.theta -= self.lr * gradient

            progress_bar.set_postfix({"Loss": f"{loss:.4f}"})

    def predict_proba(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        z = X_bias @ self.theta
        return self.softmax(z)

    def predict(self, X):
        """
        Trả về nhãn dự đoán bằng cách tìm cột có xác suất cao nhất (argmax).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    