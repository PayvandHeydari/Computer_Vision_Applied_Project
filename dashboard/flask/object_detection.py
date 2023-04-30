from flask import Blueprint, jsonify

object_data_blueprint = Blueprint("object_data", __name__)

# Sample object detection data
object_data = [
    {"object": "car", "count": 20},
    {"object": "pedestrian", "count": 10},
    {"object": "bicycle", "count": 5}
]

@object_data_blueprint.route("/object-data", methods=["GET"])
def get_object_data():
    return jsonify(object_data)
