from flask import Flask
import subprocess
import socket

app = Flask(__name__)

@app.route("/", methods=["POST"])
def stress_cpu() -> tuple[str, int]:
    """
    Runs an intensive computation loop to stress the CPUs to 100% capacity
    """
    subprocess.Popen(["python3", "stress_cpu.py"])  # Runs in background
    return "CPU Stress Test Started", 202

@app.route("/", methods=["GET"])
def get_private_ip() -> tuple[str, int]:
    """
    Returns the private IP address of the EC2 instance
    """
    private_ip = socket.gethostbyname(socket.gethostname())
    return private_ip, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
