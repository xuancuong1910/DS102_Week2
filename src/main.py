import numpy as np
from data_prep import load_raw_data, preprocess_data, to_one_hot
from models import LogisticRegression, SoftmaxRegression
from evaluate import evaluate_classification, evaluate_multiclass, plot_loss
from sklearn.linear_model import LogisticRegression as SklearnLR

def assignment_1():
    print("ASSIGNMENT 1")

    # Load data
    X_train_raw, y_train_raw = load_raw_data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    X_test_raw, y_test_raw = load_raw_data('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')

    # Prep
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw, target_classes=[0, 1])
    X_test, y_test = preprocess_data(X_test_raw, y_test_raw, target_classes=[0, 1])
    
    print(f"Kích thước tập Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Kích thước tập Test : X={X_test.shape}, y={y_test.shape}")

    model = LogisticRegression(learning_rate=0.1, epochs=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluate_classification(y_test, y_pred)

    plot_loss(model.loss_history)
    print("===")

def assignment_2():
    print("ASSIGNMENT 2")

    # Load data
    X_train_raw, y_train_raw = load_raw_data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    X_test_raw, y_test_raw = load_raw_data('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')

    # Prep
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw, target_classes=None)
    X_test, y_test = preprocess_data(X_test_raw, y_test_raw, target_classes=None)
    y_train_onehot = to_one_hot(y_train, num_classes=10)

    model = SoftmaxRegression(learning_rate=0.1, epochs=500)
    model.fit(X_train, y_train_onehot)

    y_pred = model.predict(X_test)
    evaluate_multiclass(y_test, y_pred, num_classes=10)

    plot_loss(model.loss_history)
    print("===")


def assignment_3():
    print("ASSIGNMENT 3")

    # Load data
    X_train_raw, y_train_raw = load_raw_data('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    X_test_raw, y_test_raw = load_raw_data('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')

    print("SKLEARN: LOGISTIC REGRESSION")
    X_tr_log, y_tr_log = preprocess_data(X_train_raw, y_train_raw, target_classes=[0, 1])
    X_te_log, y_te_log = preprocess_data(X_test_raw, y_test_raw, target_classes=[0, 1])

    sk_log_model = SklearnLR(max_iter=500)
    sk_log_model.fit(X_tr_log, y_tr_log)
    
    y_pred_log = sk_log_model.predict(X_te_log)
    evaluate_classification(y_te_log, y_pred_log)
    print("===")

    print("SKLEARN: SOFTMAX REGRESSION")
    X_tr_soft, y_tr_soft = preprocess_data(X_train_raw, y_train_raw, target_classes=None)
    X_te_soft, y_te_soft = preprocess_data(X_test_raw, y_test_raw, target_classes=None)

    sk_soft_model = SklearnLR(solver='saga', max_iter=100, tol=0.0001)
    sk_soft_model.fit(X_tr_soft, y_tr_soft)
    
    y_pred_soft = sk_soft_model.predict(X_te_soft)
    evaluate_multiclass(y_te_soft, y_pred_soft)
    print("===")

if __name__ == "__main__":
    assignment_1()
    assignment_2()
    assignment_3()