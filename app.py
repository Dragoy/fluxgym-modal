import subprocess
import os
from modal import (App, Image, web_server, Secret, Volume)

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

GRADIO_PORT = 7860

# Define Modal volumes
output_volume = Volume.from_name("fluxgym-output", create_if_missing=True)
models_volume = Volume.from_name("fluxgym-models", create_if_missing=True)
datasets_volume = Volume.from_name("fluxgym-datasets", create_if_missing=True)

image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("git clone https://github.com/cocktailpeanut/fluxgym.git /root/fluxgym")
    .run_commands("cd /root/fluxgym && pip install -r requirements.txt")
    .run_commands("cd /root/fluxgym && git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git /root/fluxgym/sd-scripts")
    .run_commands("cd /root/fluxgym/sd-scripts && pip install -r requirements.txt")
    # Ensure mount points don't exist
    .run_commands("rm -rf /root/fluxgym/outputs /root/fluxgym/models /root/fluxgym/datasets")
)

app = App(
    "fluxgym-trainer",
    image=image,
    secrets=[Secret.from_name("huggingface-secret")]
)

@app.cls(
    gpu="A10G",
    image=image,
    concurrency_limit=1,
    timeout=7200,
    allow_concurrent_inputs=100,
    volumes={
        "/root/fluxgym/outputs": output_volume,
        "/root/fluxgym/models": models_volume,
        "/root/fluxgym/datasets": datasets_volume
    }
)

class FluxGymApp:
    def run_gradio(self):
        os.chdir("/root/fluxgym")
        print("Changed directory to /root/fluxgym")
        os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_SECRET", "")
        os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
        os.environ["GRADIO_SERVER_PORT"] = str(GRADIO_PORT)
        os.environ["GRADIO_SERVER_HEARTBEAT_TIMEOUT"] = "7200"
        
        # Ensure models directory structure exists
        os.makedirs("/root/fluxgym/models/unet", exist_ok=True)
        os.makedirs("/root/fluxgym/models/clip", exist_ok=True)
        os.makedirs("/root/fluxgym/models/vae", exist_ok=True)
        
        # Ensure datasets directory exists
        os.makedirs("/root/fluxgym/datasets", exist_ok=True)
        
        cmd = "python app.py"
        subprocess.Popen(cmd, shell=True)

    @web_server(GRADIO_PORT, startup_timeout=120)
    def ui(self):
        print("Starting FluxGym application...")
        self.run_gradio()

if __name__ == "__main__":
    app.serve()