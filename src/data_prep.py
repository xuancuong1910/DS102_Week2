import numpy as np
import idx2numpy

def load_raw_data(image_path, label_path):
    X = idx2numpy.convert_from_file(image_path)
    y = idx2numpy.convert_from_file(label_path)
    return X, y

def preprocess_data(X, y, target_classes=None):
    # Filter
    if target_classes is not None:
        mask = np.isin(y, target_classes)
        X = X[mask]
        y = y[mask]
        # y = (y == target_classes[1]).astype(int)

    # Flatten
    X_flattened = X.reshape(X.shape[0], -1)

    # Normalization
    # Ảnh grayscale có giá trị pixel từ 0-255. Chia 255.0 để đưa về khoảng [0, 1]
    X_normalized = X_flattened / 255.0

    return X_normalized, y

def to_one_hot(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    
    return one_hot