from flask import Flask, request, jsonify, Response

app = Flask(__name__)

seed_value = 0

@app.route("/", methods=["GET"])
def get_seed() -> tuple[str, int]:
    """Returns the current seed value as a string"""
    return str(seed_value), 200

@app.route("/", methods=["POST"])
def update_seed() -> Response:
    """Updates the seed value from the JSON request"""
    global seed_value
    data = request.get_json()

    if data is None or "num" not in data:
        return jsonify({"error": "Missing 'num' in request"}), 400

    if isinstance(data["num"], int) is False:
        return jsonify({"error": "'num' must be an integer"}), 400

    seed_value = data["num"]
    return jsonify({"message": f"Seed value updated to {seed_value}"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)