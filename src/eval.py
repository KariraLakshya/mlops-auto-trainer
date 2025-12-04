import argparse, joblib, pandas as pd
from sklearn.metrics import accuracy_score

def main(model_path, data_path):
    m = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    acc = accuracy_score(y, m.predict(X))
    print(f"EVAL_ACC={acc:.4f}")
    return acc

if __name__ == "__main__":
    import sys
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    args = p.parse_args()
    main(args.model, args.data)
