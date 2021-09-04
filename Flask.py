from flask import Flask, render_template, request
import chatbot
app = Flask(__name__)
#app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot.chatbot_response(userText))



if __name__ == '__main__':
    app.run(threaded = False)
