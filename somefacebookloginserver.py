# server.py
from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('query')
    
    # Call the standalone script
    result = subprocess.run(['python', 'scraper_script.py', query], capture_output=True, text=True)

    # Parse the result from the script
    output = result.stdout.strip()
    print(output)
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return jsonify({'error': 'Failed to decode JSON from script output'}), 500

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
