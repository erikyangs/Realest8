from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return "Hello World"
