from http.server import BaseHTTPRequestHandler
import json
import pickle
import numpy as np
import os


def load_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


model = load_model()


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body   = json.loads(self.rfile.read(length))

        features = np.array([[
            body['gender'],
            body['age'],
            body['income'],
            body['reward'],
            body['difficulty'],
            body['duration'],
            body['offer_type'],
            body['member_tenure_days'],
            body['channel_email'],
            body['channel_web'],
            body['channel_mobile'],
            body['channel_social'],
        ]])

        pred = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1])

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'prediction': pred, 'probability': prob}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
