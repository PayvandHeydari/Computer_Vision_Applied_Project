from flask import Blueprint, jsonify

hardware_blueprint = Blueprint("hardware", __name__)

# Sample hardware data
hardware_data = {
    "GPU Name": "NVIDIA GeForce RTX 3090",
    "GPU Throughput": "30000 GFLOPS",
    "IOPS": "500000"
}

@hardware_blueprint.route("/hardware", methods=["GET"])
def get_hardware_data():
    return jsonify(hardware_data)
