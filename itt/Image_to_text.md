# Image to Text and Speech

In a separate notebook, I created some cute cat pictures for fun using Stable Diffusion text to image. Now, it is time to convert an image to text. This project is a bit less cute but likely just as useful, if not more so. Image-to-text conversion can mean rapid labeling of data for use in a number of applications, from detecting violent content on the internet to converting presentation slides into text for those who are vision impaired. The description of the images is important because it can convey more nuance than a simple CNN output and thus allow for more accurate classification of images.

In the spirit of image-to-text being used to help people with disabilities, I will make the pipeline image-to-text-to-audio. Without any further ado, let's begin.



## Cat Image

Let's start by defining our base image. Here, I will use a cat photo from my previous notebook. It is fun and offers the models a lot of detail to catch and report on. 


```python
image = "HyperSD.png"
```

<div style="text-align: center;">
  <img src="HyperSD.png" alt="HyperSD" style="width: 50%;"/>
</div>


## The Models

Here are the models I am running. These are some of the top downloaded image-to-text models on Huggingface. Let's see how they perform!



### Salesforce/blip-image-captioning-base


```python
from transformers import pipeline

captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
text = captioner(image)
print(text)
```

   
    [{'generated_text': 'a cat dressed as a warrior'}]
    

This output is technically correct, but warrior is a bit broad of a description. Warrior like Conan the Barbarian? Warrior like a modern soldier? Warrior like a Roman Soldier?

#### Audio for Salesforce/blip-image-captioning-base


```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

audio_array = generate_audio(text[0]['generated_text'], history_prompt="v2/en_speaker_9")

# save audio to disk
write_wav("blip-image-captioning-base.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 151/151 [00:01<00:00, 94.89it/s]
    100%|████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:03<00:00,  2.18it/s]
    


<audio  controls="controls" >

    <source src="blip-image-captioning-base.wav" type="audio/wav" />

    Your browser does not support the audio element.

</audio>





On the otherhand this Suno Text-to-Speech (TTS) model is pretty nice given that it is run locally!

### Salesforce/blip-image-captioning-large


```python
from transformers import pipeline

captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-large")
text = captioner(image)
print(text)
```

    [{'generated_text': 'there is a cat that is dressed up like a warrior'}]
    

This model is the larger version of the previous model, but the output is effectively the same. It is more verbose, but it is not any more descriptive


```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

audio_array = generate_audio(text[0]['generated_text'], history_prompt="v2/en_speaker_9")

# save audio to disk
write_wav("blip-image-captioning-large.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

    100%|███████████████████████████████████████████████████████████████████████████████| 254/254 [00:02<00:00, 103.61it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 13/13 [00:06<00:00,  2.12it/s]
    





<audio  controls="controls" >

    <source src="blip-image-captioning-large.wav" type="audio/wav" />

    Your browser does not support the audio element.

</audio>






### microsoft/git-base



```python
from transformers import pipeline

captioner = pipeline("image-to-text",model="microsoft/git-base")
text = captioner(image)
print(text)
```

    [{'generated_text': 'a cat in a costume'}]
    

Again this is technically true but even less informative than the Salesforce models.


```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

audio_array = generate_audio(text[0]['generated_text'], history_prompt="v2/en_speaker_9")

# save audio to disk
write_wav("git-base.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 154/154 [00:01<00:00, 99.08it/s]
    100%|████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:03<00:00,  2.14it/s]
    





<audio  controls="controls" >

    <source src="git-base.wav" type="audio/wav" />

    Your browser does not support the audio element.

</audio>





### unum-cloud/uform-gen2-dpo


```python
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

model = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
prompt = "Describe what is in this image"
images = Image.open(image)
inputs = processor(text=[prompt], images=[images], return_tensors="pt")
with torch.inference_mode():
     output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=256,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
    )
prompt_len = inputs["input_ids"].shape[1]
decoded_text = processor.batch_decode(output[:, prompt_len:])[0]

```


    Downloading shards:   0%|          | 0/2 [00:00<?, ?it/s]



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



```python
print(decoded_text)
```

    The image features a cute kitten wearing a medieval armor, sitting on a stone floor. The kitten has striking blue eyes and is dressed in a gray and white striped coat with a collar. The armor is ornate and has a golden hue, and the kitten is positioned in the center of the image. The background is a bit blurry, but it appears to be an old-world setting with stone arches and pillars.<|im_end|>
    

This model performed very well. It noticed that it was a kitten and not a cat. It noticed that the kitten was in medieval armor on a stone floor. It noticed the color of the cat's eyes and the color of its fur. The model said it was dressed in a grey and white striped coat, which is true, but one needed to know the coat was its fur coat for it to be completely accurate. The model was descriptive about its armor, the location of the cat, and the background. I would say this did a very good job overall!


```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

audio_array = generate_audio(decoded_text, history_prompt="v2/en_speaker_9")
# save audio to disk
write_wav("uform-gen2-dpo.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

    100%|███████████████████████████████████████████████████████████████████████████████| 698/698 [00:05<00:00, 116.77it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 35/35 [00:16<00:00,  2.08it/s]
    




<audio  controls="controls" >

    <source src="uform-gen2-dpo.wav" type="audio/wav" />

    Your browser does not support the audio element.

</audio>





Here, the audio model failed to capture the entire text because it was limited to 14 seconds. I will look for a version of the model that can handle longer text or break up the text into manageable bites and then splice the audio together. 


```python

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


predict_step([image])

```




    ['a cat wearing a bow tie sitting on the ground']



Nope, not even close. That is very strange, given that this is one of the top downloads on Huggingface. I tried copying the code from the model card verbatim, but I still ended up with this result. Maybe there is an error somewhere? In any case, the unum-cloud/uform-gen2-dpo model offered greater detail anyway.

Next I will attempt to build a function that will read power point presentations and describe the slides. Then hopefully read the text back as well. 

I hope this page was helpful!

Dan Baissa
