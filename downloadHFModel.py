import modal
import os

app = modal.App()
volume = modal.Volume.from_name("my-volume", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "joblib",
    "huggingface_hub",
)

with image.imports():
    from huggingface_hub import hf_hub_download

@app.function(volumes={"/vol": volume}, image=image, _allow_background_volume_commits=True)
def download_model():
    model = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.ckpt", local_dir="/vol/models/sd15")
    print(model)

@app.local_entrypoint()
def main():
    download_model.remote()