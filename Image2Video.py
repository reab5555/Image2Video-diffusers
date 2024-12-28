from google.cloud import storage
import os
import torch
from diffusers import LTXImageToVideoPipeline
from PIL import Image
from diffusers.utils import export_to_video
from threading import Thread
import tempfile
import GPUtil
import time
from queue import Queue


def print_gpu_usage():
    while True:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f'GPU {gpu.id}: Memory Used: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)')
        print('-' * 50)
        time.sleep(10)


def download_from_gcs(bucket_name, source_blob_name, dest_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(dest_file_name)


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def process_image(input_gcs_path, output_gcs_path, gpu_id):

    bucket_name = input_gcs_path.split('/')[2]
    input_path = input_gcs_path.split('/', 3)[3]
    output_path = output_gcs_path.split('/', 3)[3]

    with tempfile.TemporaryDirectory() as temp_dir:
        local_input = os.path.join(temp_dir, "input.png")
        local_output = os.path.join(temp_dir, "output.mp4")

        download_from_gcs(bucket_name, input_path, local_input)
        print(f"Downloaded {input_path}")

        # Load the pipeline and assign to a specific GPU
        pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16
        ).to(f"cuda:{gpu_id}")

        # Open and process image
        image = Image.open(local_input)

        # Set fixed dimensions for 480p resolution, ensuring divisibility by 32
        width = 848 - (848 % 32)  # Closest width to 480p divisible by 32
        height = 480 - (480 % 32)  # Closest height to 480p divisible by 32

        print(f"Processing image {input_path} on GPU {gpu_id} with dimensions {width}x{height}")
        output = pipe(
            image=image,
            prompt="make this image a realistic video",
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            width=width,
            height=height,
            num_frames=241,
            num_inference_steps=75
        ).frames[0]

        # Export and upload video
        export_to_video(output, local_output, fps=24)
        upload_to_gcs(bucket_name, local_output, output_path)
        print(f"Uploaded {output_path} from GPU {gpu_id}")

        # Clear GPU memory
        torch.cuda.empty_cache()
        print(f"Cleared GPU {gpu_id} memory")



def worker_thread(gpu_id, task_queue):
    while not task_queue.empty():
        input_gcs_path, output_gcs_path = task_queue.get()
        try:
            process_image(input_gcs_path, output_gcs_path, gpu_id)
        except Exception as e:
            print(f"Error processing {input_gcs_path} on GPU {gpu_id}: {e}")
        task_queue.task_done()


def main():
    storage_client = storage.Client()
    input_bucket = "gs://bucket/Inputs"
    output_bucket = "gs://bucket/Outputs"

    monitor_thread = Thread(target=print_gpu_usage, daemon=True)
    monitor_thread.start()

    bucket_name = input_bucket.split('/')[2]
    input_prefix = '/'.join(input_bucket.split('/')[3:]) + '/'
    bucket = storage_client.bucket(bucket_name)

    # List input files
    blobs = list(bucket.list_blobs(prefix=input_prefix))
    input_files = [blob.name for blob in blobs if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(input_files)} images to process.")

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    # Create a task queue
    task_queue = Queue()
    for input_file in input_files:
        output_path = os.path.join(
            '/'.join(output_bucket.split('/')[3:]),
            os.path.splitext(os.path.basename(input_file))[0] + '.mp4'
        )
        input_gcs_path = f"gs://{bucket_name}/{input_file}"
        output_gcs_path = f"gs://{bucket_name}/{output_path}"
        task_queue.put((input_gcs_path, output_gcs_path))

    # Launch one thread per GPU
    threads = []
    for gpu_id in range(num_gpus):
        thread = Thread(target=worker_thread, args=(gpu_id, task_queue))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
