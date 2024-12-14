# FluxGym Trainer

A Modal-based application for training Stable Diffusion models using FluxGym interface.

## Features

- Runs on Modal cloud platform
- Uses NVIDIA CUDA for GPU acceleration
- Integrates with HuggingFace
- Web-based UI powered by Gradio
- Supports A100 GPU for training

## Prerequisites

- Modal account
- HuggingFace account and API token
- Python 3.11+

## Setup

1. Install Modal CLI:
```bash
pip install modal
```

2. Configure Modal:
```bash
modal token new
```

3. Set up HuggingFace secret in Modal:
```bash
modal secret create huggingface-secret HUGGINGFACE_SECRET=your_token_here
```

## Running the Application

1. Deploy the application:
```bash
modal deploy app.py
```

2. To run locally:
```bash
modal run app.py
```

## Environment

- CUDA Version: 12.4.0
- Base OS: Ubuntu 22.04
- Python Version: 3.11
- GPU: NVIDIA A100

## Architecture

The application uses Modal's serverless infrastructure to run a Gradio-based web interface for training Stable Diffusion models. It utilizes persistent volume storage for outputs and integrates with HuggingFace for model management.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 