"""API HTTP minimale (bibliothèque standard) autour de ClinicalPredictor.

Lancement : python -m virus_diag.app.api
Requête   : POST /diagnose avec un JSON contenant les 6 variables patient.
Jeu de données synthétique : usage pédagogique uniquement.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from virus_diag.core.clinical_predictor import ClinicalPredictor
from virus_diag.pipeline.trainer import Trainer
from virus_diag.utils.preprocessing import FEATURES


class DiagnosisAPI:
    def __init__(self, predictor):
        self.predictor = predictor

    def handle(self, payload):
        missing = [f for f in FEATURES if f not in payload]
        if missing:
            return 400, {"erreur": f"Champs manquants : {missing}"}
        return 200, {"diagnostic": self.predictor.diagnose(payload)}


def make_handler(api):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/diagnose":
                return self._send(404, {"erreur": "Route inconnue"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
            except ValueError:
                return self._send(400, {"erreur": "JSON invalide"})
            self._send(*api.handle(payload))

        def _send(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(port=8000):
    model = Trainer().train()
    api = DiagnosisAPI(ClinicalPredictor(model))
    print(f"API sur http://localhost:{port}/diagnose")
    HTTPServer(("", port), make_handler(api)).serve_forever()


if __name__ == "__main__":
    serve()
