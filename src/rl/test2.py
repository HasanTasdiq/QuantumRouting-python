import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8000"  # change port if needed

def to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [to_serializable(i) for i in x]
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    return x

def call_learn_predict(reqIndex, ent_matrix, req_matrix, dist_matrix):
    url = f"{BASE_URL}/learn_predict"
    payload = {
        "reqIndex": reqIndex,
        "ent_matrix": to_serializable(ent_matrix),
        "req_matrix": to_serializable(req_matrix),
        "dist_matrix": to_serializable(dist_matrix)
    }
    r = requests.post(url, json=payload)
    return r.json()


if __name__ == "__main__":
    payload = {
        "reqIndex": 2,
        "ent_matrix": [
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ],
        "req_matrix": [
            [5, 8, 5, 0, 0, 0],
            [1, 7, 1, 0, 1, 0],
            [8, 1, 8, 0, 2, 0]
        ],
        "dist_matrix": [
            [0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0]
        ]
    }

    print("Calling learn_predict with payload...")
    print(call_learn_predict(payload["reqIndex"], payload["ent_matrix"], payload["req_matrix"], payload["dist_matrix"]))
