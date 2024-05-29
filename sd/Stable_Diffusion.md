# Where NLP and image generation meet




```python
prompt = "A cinematic shot of a kitten wearing an intricate italian knight's armor."
```


```python
from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline (model and tokenizer)
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True).to("cuda")


image = pipeline(prompt=prompt).images[0]

# Save the image
image_path = "generated_image.png"
image.save(image_path)
```




![png](generated_image.png)



```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0].save("sdxl-turbo.png")

```



![sdxl-turbo](sdxl-turbo.png)


```python
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
# Take 2-steps lora as an example
ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
# lower eta results in more detail

image=pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0).images[0].save("HyperSD.png")

```




![HyperSD](HyperSD.png)


```python
import torch
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-1step-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Use TCD scheduler to achieve better image quality
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# Lower eta results in more detail for multi-steps inference
eta=1.0

image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, eta=eta).images[0].save("HyperSD2.png")

```



![HyperSD2](HyperSD2.png)
