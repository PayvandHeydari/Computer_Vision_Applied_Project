from flask import Flask, jsonify, request, send_from_directory, send_file, Response
from flask_cors import CORS
import os
import re

app = Flask(__name__, static_folder="static")
CORS(app)

# Sample hardware data
hardware_data = {
    "GPU Name": "NVIDIA GeForce RTX 3090",
    "GPU Throughput": "30000 GFLOPS",
    "IOPS": "500000"
}

# Sample object detection data
object_data = [
    {"object": "car", "count": 20},
    {"object": "pedestrian", "count": 10},
    {"object": "bicycle", "count": 5}
]

@app.route("/api/hardware", methods=["GET"])
def get_hardware_data():
    return jsonify(hardware_data)

@app.route("/api/object-data", methods=["GET"])
def get_object_data():
    return jsonify(object_data)

@app.route('/api/static/videos/<path:filename>')
def serve_video_file(filename):
    video_path = os.path.join(app.static_folder, 'videos', filename)
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(video_path, mimetype='video/mp4')

    size = os.path.getsize(video_path)
    byte1, byte2 = 0, None
    match = re.search(r'(\d+)-(\d*)', range_header)
    g = match.groups()

    if g[0]:
        byte1 = int(g[0])
    if g[1]:
        byte2 = int(g[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1 + 1

    data = None
    with open(video_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    response = Response(data, 206, mimetype='video/mp4', content_type='video/mp4', direct_passthrough=True)
    response.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(byte1, byte1 + length - 1, size))
    response.headers.add('Content-Disposition', 'inline; filename={}'.format(filename))
    response.headers.add('Accept-Ranges', 'bytes')  # Added header to support range requests

    return response

@app.route('/api/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

if __name__ == "__main__":
    app.run(debug=True)
