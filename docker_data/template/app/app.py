from flask import Flask, request
import os
import socket

app = Flask(__name__)


@app.route("/api", methods=['POST'])  # GET requests will be blocked
def get_string():
    """
    This method will accept JSON data in POST request and it will write to the file
    """
    string_data = request.get_json()

    user_command = string_data['command']

    try:  # writing it to channel so that the pipeline script can read it
        channel_file = open("../pos_tagger/resources/channel.txt", "w")
        channel_file.write("%s" % user_command)
        channel_file.close()
    except Exception as e:
        return "Error"

    return "OK"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
