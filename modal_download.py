from modal import Volume, Image, App
import pathlib
pose_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth"

app = App()
volume = Volume.from_name("my-volume", create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.3.1",
    "torchvision",
    "accelerate",
    "diffusers",
    "ftfy",
    "safetensors",
    "torch",
    "torchvision",
    "transformers",
    "triton",
    "httpx",
    "tqdm",
).pip_install('xformers')

with image.imports():
    import httpx
    from tqdm import tqdm

def download_file(url: str, output_path: pathlib.Path):
    with open(output_path, "wb") as download_file:
        with httpx.stream("GET", url, follow_redirects=True) as response:
            total = int(response.headers["Content-Length"])
            with tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = response.num_bytes_downloaded
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(
                        response.num_bytes_downloaded - num_bytes_downloaded
                    )
                    num_bytes_downloaded = response.num_bytes_downloaded

@app.function(volumes={"/vol": volume}, image=image, _allow_background_volume_commits=True, timeout=600)
def download_demo_files() -> None:
    models_dir = pathlib.Path("/vol/controlnets/pose")
    filepath = "control_sd15_openpose.pth"
    download_file(url=pose_path, output_path=models_dir)
    print(f"download complete for {filepath}")

@app.local_entrypoint()
def main():
    download_demo_files.remote()