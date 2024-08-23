from flask import Flask, request, jsonify
from pyngrok import ngrok

app = Flask(__name__)

# Set ngrok authtoken
ngrok.set_auth_token("2hZ1a3ktCeJBhkG9DsThddItHbW_4rd6NSVgphvNE5Efti4A9")

@app.route('/data', methods=['POST'])
def receive_data():
    req_data = request.get_json()
    response = {'message': 'Received data successfully', 'data': req_data}
    return jsonify(response)

if __name__ == '__main__':
    # Start ngrok and run Flask app
    public_url = ngrok.connect(5000)  # Replace with your Flask app port
    print(f"ngrok tunnel running at {public_url}")
    app.run()