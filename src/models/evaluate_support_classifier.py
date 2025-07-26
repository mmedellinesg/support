import sys
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

def score_to_label(score, task='support'):
    """Convert a float score to a categorical label"""
    if task == 'offensiveness':
        return 'Offensive' if score >= 1.5 else 'Neutral'
    else:
        if score > 3.5:
            return 'Positive'
        elif score < 2.5:
            return 'Negative'
        else:
            return 'Neutral'

def load_data(zero_file, task='support'):
    y_true = []
    X = []

    with open(zero_file) as f:
        for line_no, line in enumerate(f):
            cols = line.strip().split('\t')
            if len(cols) < 5:
                continue

            # Map score index
            task_index = {
                'agreement': 0,
                'offensiveness': 1,
                'politeness': 2,
                'support': 3,
            }[task]

            val = float(cols[task_index])
            label = score_to_label(val, task)
            y_true.append(label)

            # Feature vector
            feat_vec = np.array([float(x) for x in cols[4].split()])
            X.append(feat_vec)

    return X, y_true

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python evaluate_support_classifier.py model.pkl test_data.zero.tsv [task]")
        print("Default task: support")
        sys.exit(1)

    model_path = sys.argv[1]
    zero_tsv = sys.argv[2]
    task = sys.argv[3] if len(sys.argv) == 4 else 'support'

    print(f"Evaluating task: {task}")
    print("Loading model...")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    print("Loading data...")
    X, y_true = load_data(zero_tsv, task=task)

    print("Predicting...")
    y_pred = clf.predict(X)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=['Positive', 'Neutral', 'Negative']))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    print("\nLabel distribution in true labels:")
    print(Counter(y_true))

if __name__ == '__main__':
    main()
