import subprocess
import os
from modal import App, Image, Volume, Secret

# CUDA configuration
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Define Modal image
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

# Define Modal volumes
output_volume = Volume.from_name("fluxgym-output", create_if_missing=True)
models_volume = Volume.from_name("fluxgym-models", create_if_missing=True)
datasets_volume = Volume.from_name("fluxgym-datasets", create_if_missing=True)

# Create Modal app
app = App("fluxgym-train-launcher")

@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/root/fluxgym/outputs": output_volume,
        "/root/fluxgym/models": models_volume,
        "/root/fluxgym/datasets": datasets_volume
    },
    secrets=[Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours timeout, adjust if needed
)
def run_lora_training(model_name: str):
    """
    Run LoRA training for a specific model using the generated train.sh script.
    
    Args:
        model_name: Name of the model/output directory containing train.sh
    """
    os.chdir("/root/fluxgym")
    print(f"Starting LoRA training for model: {model_name}")
    
    # Ensure models directory structure exists
    os.makedirs("/root/fluxgym/models/unet", exist_ok=True)
    os.makedirs("/root/fluxgym/models/clip", exist_ok=True)
    os.makedirs("/root/fluxgym/models/vae", exist_ok=True)
    
    # Ensure datasets directory exists
    os.makedirs("/root/fluxgym/datasets", exist_ok=True)
    
    train_script_path = f"/root/fluxgym/outputs/{model_name}/train.sh"

    if not os.path.exists(train_script_path):
        print(f"Error: train.sh script not found at {train_script_path}")
        return

    # Set HuggingFace token from secrets
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_SECRET", "")

    # Run train.sh through bash and stream output
    command = ["/bin/bash", train_script_path]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Stream and print output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    return_code = process.poll()
    if return_code == 0:
        print(f"LoRA training for model '{model_name}' completed successfully.")
    else:
        print(f"LoRA training for model '{model_name}' failed with return code: {return_code}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        app.run(run_lora_training, args=(model_name,))
    else:
        print("Usage: modal run modal_train_launcher.py::run_lora_training <model_name>") 