import subprocess
import os
from modal import (App, Image, web_server, Secret, Volume)

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

GRADIO_PORT = 7860

volume = Volume.from_name("fluxgym-output", create_if_missing=True)

image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("git clone https://github.com/cocktailpeanut/fluxgym.git /root/fluxgym")
    .run_commands("cd /root/fluxgym && pip install -r requirements.txt")
    .run_commands("cd /root/fluxgym && git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git /root/fluxgym/sd-scripts")
    .run_commands("cd /root/fluxgym/sd-scripts && pip install -r requirements.txt")
    .run_commands("rm -rf /root/fluxgym/outputs")
)

app = App(
    "fluxgym-trainer",
    image=image,
    secrets=[Secret.from_name("huggingface-secret")]
)

@app.cls(
    gpu="A100",
    image=image,
    concurrency_limit=1,
    timeout=7200,
    allow_concurrent_inputs=100,
    volumes={"/root/fluxgym/outputs": volume}
)

class FluxGymApp:
    def run_gradio(self):
        os.chdir("/root/fluxgym")
        print("Changed directory to /root/fluxgym")
        os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_SECRET", "")
        os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
        os.environ["GRADIO_SERVER_PORT"] = str(GRADIO_PORT)
        os.environ["GRADIO_SERVER_HEARTBEAT_TIMEOUT"] = "7200"
        os.environ["TRANSFORMERS_OFFLINE"] = "1" 
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" 
        cmd = "python app.py"
        subprocess.Popen(cmd, shell=True)

    @web_server(GRADIO_PORT, startup_timeout=120)
    def ui(self):
        print("Starting FluxGym application...")
        self.run_gradio()

if __name__ == "__main__":
    app.serve()