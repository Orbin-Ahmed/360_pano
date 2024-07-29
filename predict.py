import os
import random
from typing import Dict
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from model_code.text_to_360panorama_image_pipeline import Text2360PanoramaImagePipeline
from model_code.image_to_360panorama_image_pipeline import Image2360PanoramaImagePipeline

MODEL_DIR = os.path.abspath("./diffusers-cache")

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipeline_text = Text2360PanoramaImagePipeline(model=MODEL_DIR, device='cuda', torch_dtype=torch.float16)
        self.pipeline_image = Image2360PanoramaImagePipeline(model=MODEL_DIR, device='cuda', torch_dtype=torch.float16)
        self.mask_image = Image.open(os.path.abspath("./diffusers-cache/i2p-mask.jpg"))

    def predict(
        self,
        image: Path = Input(description="Input image", default=None),
        prompt: str = Input(
            description="Input prompt",
            default="A living room"
        ),
        upscale: bool = Input(
            description="Whether to upscale the image",
            default=False
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        input_dict = {
            'prompt': prompt,
            'upscale': upscale,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
        }
        
        if image:
            img = Image.open(str(image))
            input_dict['image'] = img.resize((512, 512))
            input_dict['mask'] = self.mask_image
            print("Image Pipeline is running!")
            output_img = self.pipeline_image(input_dict)
        else:
            print("Text Pipeline is running!")
            output_img = self.pipeline_text(input_dict)
        
        output_path = "/tmp/result.png"
        output_img.save(output_path)

        return Path(output_path)
