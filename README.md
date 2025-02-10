# Image to Video Conversion Pipeline

This project provides a scalable and GPU-optimized pipeline designed to convert high-resolution images into videos. It utilizes advanced AI models and supports seamless integration with Google Cloud Storage. The pipeline is equipped to handle multiple GPUs for efficient processing and includes features for real-time GPU monitoring.

---

```python
width = 848 - (848 % 32)  # Closest width to 480p divisible by 32
height = 480 - (480 % 32)  # Closest height to 480p divisible by 32
```

Here, `848` and `480` are the approximate dimensions for 480p resolution, adjusted to be divisible by 32 for optimal processing. The code processes the image by assigning it to a specific GPU and uses these dimensions to generate the video.

The `pipe` function processes the image with the following parameters:
- **image**: The input image to be transformed into a video.
- **prompt**: Describes the desired outcome, e.g., "make this image a realistic video".
- **negative_prompt**: Specifies what to avoid in the output, such as "worst quality, inconsistent motion, blurry, jittery, distorted".
- **width** and **height**: Specify the resolution for the output video, ensuring quality and compatibility.
- **num_frames**: Defines the total number of frames in the output video, set here to `241`.
- **num_inference_steps**: Determines the number of inference steps for generating each frame, set to `75` for balancing speed and quality.

The pipeline dynamically processes images and exports the results as videos at 24 frames per second. After processing, the GPU memory is cleared to maintain system performance.

---

## How to Use the Pipeline

1. Set up Google Cloud credentials by exporting the path to your service account key:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
   ```
2. Run the main script:
   ```bash
   python Image2Video.py
   ```
   The script automatically downloads input images from a Google Cloud Storage bucket, processes them, and uploads the resulting videos back to the specified output bucket.

---
