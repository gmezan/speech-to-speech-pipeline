Pipeline 1: Data Preprocessing (Continuous Integration)
Goal: Transform raw audio into a normalized format and store it immutably.
Trigger: Push to the data/ branch, or manually via GitHub Actions UI (workflow_dispatch).

Execution Environment: GitHub Actions standard runner (Ubuntu).

The Process:

Checkout: GitHub Action pulls the repository.

Setup: Installs Python, uv, and librosa/ffmpeg.

Process: Runs a script to standardize all raw .wav files to 16kHz, single-channel arrays.

Validate: Runs basic assertions (e.g., ensuring no file is empty, all sample rates are exactly 16kHz).

Push to Hub: Authenticates via a GitHub Secret (HF_TOKEN) and pushes the processed dataset directly to Hugging Face (datasets.push_to_hub).

Best Practice Applied: Storing data on Hugging Face instead of AWS S3 natively versions your datasets. If a data update ruins the model, you can instantly rollback to a previous dataset commit.

Pipeline 2: Ephemeral GPU Training (Continuous Training)
Goal: Train STT (Whisper) or TTS (VITS) on GPUs without paying for idle server time.
Trigger: Manually via GitHub Actions UI, triggered after a successful Pipeline 1 run.

Execution Environment: GitHub Actions triggers AWS; AWS runs the workload.

The Process:

Fire-and-Forget Launch: GitHub Actions uses the AWS CLI to launch an EC2 g5.xlarge instance.

Instance User Data (Cloud-Init): Instead of SSHing into the machine, GitHub Actions passes a startup bash script (user-data.sh) to the EC2 instance during creation.

Autonomous Execution: The EC2 instance boots, installs uv, downloads the HF dataset, and runs train_lora.py.

Merge & Upload: The script merges the LoRA adapters with the base model and pushes the final weights to the Hugging Face Hub using the HF_TOKEN.

Self-Termination (Crucial): As its very last step, the EC2 instance runs sudo shutdown -h now. The AWS instance is configured to terminate on shutdown, guaranteeing you are never billed for idle GPU time.

Best Practice Applied: Tying up a GitHub Action runner for 10 hours while waiting for a model to train eats through CI/CD billing limits. The "Fire-and-Forget" method offloads the workload to AWS and terminates itself independently.

Pipeline 3: API Deployment (Continuous Deployment)
Goal: Serve the compiled model reliably as a low-latency API endpoint.
Trigger: GitHub Release created, or a manual push to the main branch.

Execution Environment: GitHub Actions builds the image; a persistent EC2 instance serves it.

The Process:

Containerize: GitHub Actions builds a Docker image containing a FastAPI WebSocket server, PyTorch, and your inference scripts.

Publish Image: The Action pushes this Docker image to GitHub Container Registry (GHCR) or Amazon ECR.

Deploy to Inference Server: The Action connects via SSH to your persistent deployment EC2 instance (a smaller, cheaper GPU instance like g4dn.xlarge or CPU instance if using ONNX).

Rolling Update: The server pulls the new Docker image, gracefully stops the old container, and starts the new one, minimizing downtime.

Best Practice Applied: Dockerizing the API ensures absolute replicability. The exact same environment that runs on your deployment EC2 can be run on your Mac by simply typing docker run.

GitHub Secrets Checklist
To orchestrate this, your GitHub repository will need these secrets configured:

AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY: For provisioning EC2 instances.

HF_TOKEN: For pushing/pulling datasets and model weights to Hugging Face.

SSH_PRIVATE_KEY: To access the Pipeline 3 deployment server for rolling updates.