from flask import Flask, render_template, request, jsonify
from chatbot import Chatbot

app = Flask(__name__)
bot = Chatbot()

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_msg = request.form["msg"]
    result = bot.get_answer(user_msg)
    return jsonify(result)

@app.route("/train", methods=["POST"])
def train_bot():
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")
    response = bot.learn(question, answer)
    return jsonify({"message": response})

if __name__ == "__main__":
    app.run(debug=True)
