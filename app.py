from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Flask başlat
app = Flask(__name__)

# Kodlanmış verileri yükle
home_encoding = pd.read_csv("models/home_encoding.csv").set_index("home_team")["home_team_encoded"].to_dict()
away_encoding = pd.read_csv("models/away_encoding.csv").set_index("away_team")["away_team_encoded"].to_dict()
avg_home_goals = pd.read_csv("models/avg_home_goals.csv", index_col=0)["home_score"].to_dict()
avg_away_goals = pd.read_csv("models/avg_away_goals.csv", index_col=0)["away_score"].to_dict()

# Modelleri yükle
multi_target_model = joblib.load("models/multi_target_model.pkl")
classifier = joblib.load("models/classifier_model.pkl")

@app.route("/", methods=["GET"])
def index():
    return "Premier League Tahmin API'si çalışıyor! /predict endpointine POST isteği gönderin.", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if home_team not in home_encoding or away_team not in away_encoding:
        return jsonify({"error": "Takımlar veri setinde bulunamadı"}), 400

    home_encoded = home_encoding[home_team]
    away_encoded = away_encoding[away_team]

    home_avg = avg_home_goals.get(home_team, 0)
    away_avg = avg_away_goals.get(away_team, 0)
    goal_diff = home_avg - away_avg
    total_goals = home_avg + away_avg

    # Regresyon tahmini
    features_reg = [[home_encoded, away_encoded, goal_diff, total_goals]]
    predicted_scores = multi_target_model.predict(features_reg)[0]

    # Sınıflandırma tahmini
    features_class = [[home_encoded, away_encoded]]
    predicted_result = classifier.predict(features_class)[0]

    return jsonify({
        "home_team": home_team,
        "away_team": away_team,
        "predicted_home_score": int(predicted_scores[0]),
        "predicted_away_score": int(predicted_scores[1]),
        "predicted_result": predicted_result
    })

if __name__ == "__main__":
    app.run(debug=True)
