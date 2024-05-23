---
title: "Using Transformers on Natural Language Data Locally!"
author: "Daniel K Baissa"
---

[//]: # (># Using Transformers on Natural Language Data Locally!)

[//]: # (> **by Daniel K Baissa**)


ChatGPT has put Transformer models on the map. But there are actually many off the shelf transformer models to use. The goal here is to show some of the basics of using these models, and what they can do. The attention mask that transformer models has really gives them an edge in time series data such as Natural Language Processing (NLP) 

## Table of Contents
1. [Set up your environment](#set-up-your-environment)
3. [A Story to Practice NLP On](#a-story-to-practice-nlp-on)
4. [Text Classification](#text-classification)
4. [Named Entity Recognition](#named-entity-recognition)
5. [Question Answering](#question-answering)
6. [Summarization](#summarization)
7. [Translation](#translation)
8. [Text Generation](#text-generation)

## Set up your environment

```python
import torch
from transformers import AutoModel

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready for use.")
    device = torch.device("cuda")  # Set device to GPU
else:
    
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")  # Set device to CPU
```

    CUDA is available. GPU is ready for use.
    

## A Story to Practice NLP On

I used Chatgpt to throw together a quick story for us to practice on. 


```python
text = """Once upon a time, in a cozy little house at the edge of a bustling town Utica in New York State, there lived a baby girl named Maia. Maia wasn't just any baby girl; she was curious, smart, and had a knack for getting into all sorts of funny adventures.

One sunny morning, Maia woke up with a brilliant idea. She decided it was the perfect day to explore the world beyond her crib. With a determined look in her bright eyes, she wiggled her way out of bed and toddled towards the door.

First stop: the kitchen. Maia's nose twitched at the delicious smells wafting from the oven. She reached up on her tiptoes, managing to grab a cookie from the counter. "Mmm, breakfast of champions," she thought with a giggle as she munched on her treat.

Next, she made her way to the living room, where her family's cat, Whiskers, was napping. Maia poked Whiskers gently, curious about the fluffy creature. Whiskers opened one eye, gave a lazy yawn, and decided that Maia's curiosity was too much for a morning nap. With a flick of his tail, he trotted off, leaving Maia to find her next adventure.

The garden was full of wonders. Maia found flowers to smell, butterflies to chase, and even a friendly worm wriggling its way across the soil. She giggled with delight, her tiny hands clapping at the sight of every new discovery.

She had seen so much, learned so many new things, and had so much fun. And though she was just a baby, Maia knew that the world was full of endless adventures waiting for her to discover."""
```

## Text Classification

Here we will use a transformer from Hugging Face🤗. Let's start by importing the transformer library


```python
from transformers import pipeline
import pandas as pd
```

We will use the defaul Distilbert model


```python
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)
```

 
    

Now that we have the classifier all we need to do is classify the text as negative or positive. I will use pandas to make this into a table so it will look nice.


```python
outputs = classifier(text)
pd.DataFrame(outputs)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POSITIVE</td>
      <td>0.999142</td>
    </tr>
  </tbody>
</table>
</div>



Here we can see that this is labled as positive with an extremely high degree of confidince. This is not suprising given that it the prompt given to Chatgpt was to make a cute story about a baby girl named Maia. 

## Named Entity Recognition

We were able to successfuly estimate the sentiment of the story, which is nice. But it might also be nice to know what they text is actually talking about. Who or what is the text about. Here we will use a named entity recognition model "ner". Here we will use Note that the default model "dbmdz/bert-large-cased-finetuned-conll03-english". That is ok for this task


```python
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
revision = "f2482bf"
ner_tagger = pipeline("ner", model=model_name, revision=revision, aggregation_strategy="simple")
```


```python
outputs = ner_tagger(text)
pd.DataFrame(outputs)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>entity_group</th>
      <th>score</th>
      <th>word</th>
      <th>start</th>
      <th>end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LOC</td>
      <td>0.988708</td>
      <td>Utica</td>
      <td>72</td>
      <td>77</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOC</td>
      <td>0.982091</td>
      <td>New York State</td>
      <td>81</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PER</td>
      <td>0.987294</td>
      <td>Maia</td>
      <td>127</td>
      <td>131</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PER</td>
      <td>0.990340</td>
      <td>Maia</td>
      <td>133</td>
      <td>137</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PER</td>
      <td>0.982831</td>
      <td>Maia</td>
      <td>273</td>
      <td>277</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PER</td>
      <td>0.973810</td>
      <td>Maia</td>
      <td>512</td>
      <td>516</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PER</td>
      <td>0.954940</td>
      <td>Whiskers</td>
      <td>809</td>
      <td>817</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PER</td>
      <td>0.965617</td>
      <td>Maia</td>
      <td>832</td>
      <td>836</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PER</td>
      <td>0.918289</td>
      <td>Whiskers</td>
      <td>843</td>
      <td>851</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PER</td>
      <td>0.941341</td>
      <td>Whiskers</td>
      <td>895</td>
      <td>903</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PER</td>
      <td>0.972468</td>
      <td>Maia</td>
      <td>955</td>
      <td>959</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PER</td>
      <td>0.945366</td>
      <td>Maia</td>
      <td>1054</td>
      <td>1058</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PER</td>
      <td>0.962110</td>
      <td>Maia</td>
      <td>1120</td>
      <td>1124</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PER</td>
      <td>0.971372</td>
      <td>Maia</td>
      <td>1422</td>
      <td>1426</td>
    </tr>
  </tbody>
</table>
</div>



Here we see that Maia is the most common entity in the text. This is of course good, because she is the main subject of most of the text. Whiskers the cat, Utica, and New York State also show up as a named entities. We can also see that it groups the entities, Maia is listed as a person, Utica and New York show up as Locations. Funnily enough, Whiskers shows up as a person. 

## Question Answering

This will respond with a span of the corresponding text to answer the question that we give it. This is useful because you can identify sections of the data without asking the model for an opinion so to speak.


```python
model_name = "distilbert/distilbert-base-cased-distilled-squad"
revision = "626af31"


reader = pipeline("question-answering", model=model_name, revision=revision)
question = "Where did Maia live?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>start</th>
      <th>end</th>
      <th>answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.434803</td>
      <td>72</td>
      <td>95</td>
      <td>Utica in New York State</td>
    </tr>
  </tbody>
</table>
</div>



Here we can see that it correctly identified Utica, New York as the correct location. Let's see if it can handle a slightly more complicated question.


```python
reader = pipeline("question-answering", model=model_name, revision=revision)
question = "What is Maia's cat's name?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>start</th>
      <th>end</th>
      <th>answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.995755</td>
      <td>809</td>
      <td>817</td>
      <td>Whiskers</td>
    </tr>
  </tbody>
</table>
</div>



Here we have a high degree of confidence that it is Whiskers, which is correct!

## Summarization

Here we will use a transformer model to summarize the text. This will actually have the model generate text. 


```python
model_name = "sshleifer/distilbart-cnn-12-6"
revision = "a4f8f3e"
```


```python
summarizer = pipeline("summarization", model=model_name, revision=revision)
outputs = summarizer(text, max_length=75, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
```

     A baby girl named Maia was born in Utica, New York State. She was curious, smart, smart and had a knack for getting into funny adventures. One sunny morning, Maia woke up with a brilliant idea to explore the world beyond her crib. She found flowers to smell, butterflies to chase, and even a friendly worm in the garden.
    

Not bad at all for a quick model run locally!

## Translation

Let's see if we can translate this text into multiple languages! Again, we are using basic models.


```python
translator = pipeline("translation_en_to_de", 
                      model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]["translation_text"])
```

    Es war einmal, in einem gemütlichen kleinen Haus am Rande einer geschäftigen Stadt Utica in New York State, lebte ein Baby Mädchen namens Maia. Maia war nicht nur ein Baby Mädchen; sie war neugierig, klug, und hatte ein Kniff für die Aufnahme in alle Arten von lustigen Abenteuern. Eines sonnigen Morgens, erwachte Maia mit einer brillanten Idee. Sie entschied, es war der perfekte Tag, um die Welt jenseits ihrer Krippe zu erkunden. Mit einem entschlossenen Blick in ihren hellen Augen, sie wackelte ihren Weg aus dem Bett und kippte in Richtung der Tür. Erste Station: die Küche. Maias Nase verzauberte an den köstlichen Gerüchen wafting aus dem Ofen. Sie erreichte auf ihren Zehenspitzen, um einen Keks aus der Theke zu greifen. "Mmm, Frühstück der Champions", dachte sie mit einem kichern, als sie auf ihrem Leckerling spritzte. Weiter machte sie ihren Weg zum Wohnzimmer, wo ihre Familie, Whiskers, war napping. Maia poked Whikers sanft, neugierig auf die Maiappy-Kette.
    

### Create the translation pipeline for English to Arabic


```python
translator = pipeline("translation_en_to_ar", 
                      model="Helsinki-NLP/opus-mt-en-ar")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]["translation_text"])
```

    في وقت من الأوقات، في منزل صغير على حافة مدينة أوتيكا في ولاية نيويورك، عاشت طفلة اسمها مايا. مايا لم تكن مجرد أي طفلة؛ كانت غريبة، ذكية، وكان لديها مُسْتَعِدّة للحصول على جميع أنواع من الرائحات المُضحكة. في صباح مشمس، استيقظت مايا بفكرة رائعة. قررت أنه اليوم المثالي لاستكشاف العالم خارج سريرها. مع نظرة حاسمة في عينيها المشرقة، كانت تهتز طريقها للخروج من السرير والتدحرج نحو الباب. أولاً، كانت:
    

## Text Generation

Now for the thing that transformer models have received a lot of hype for! Despite being made initially for translation services, their usage for generative language is famous. Let's use GPT2 which is open source. This model has only 124M parameters. 


```python
generator = pipeline("text-generation",
                    model="openai-community/gpt2")
response = "Maia gets ready for her next adventure. She finds a toy boat and"
prompt = text + "\n\n ---"+ response
```


```python
outputs = generator(prompt, max_length=500)
print(outputs[0]['generated_text'])
```

    Once upon a time, in a cozy little house at the edge of a bustling town Utica in New York State, there lived a baby girl named Maia. Maia wasn't just any baby girl; she was curious, smart, and had a knack for getting into all sorts of funny adventures.
    
    One sunny morning, Maia woke up with a brilliant idea. She decided it was the perfect day to explore the world beyond her crib. With a determined look in her bright eyes, she wiggled her way out of bed and toddled towards the door.
    
    First stop: the kitchen. Maia's nose twitched at the delicious smells wafting from the oven. She reached up on her tiptoes, managing to grab a cookie from the counter. "Mmm, breakfast of champions," she thought with a giggle as she munched on her treat.
    
    Next, she made her way to the living room, where her family's cat, Whiskers, was napping. Maia poked Whiskers gently, curious about the fluffy creature. Whiskers opened one eye, gave a lazy yawn, and decided that Maia's curiosity was too much for a morning nap. With a flick of his tail, he trotted off, leaving Maia to find her next adventure.
    
    The garden was full of wonders. Maia found flowers to smell, butterflies to chase, and even a friendly worm wriggling its way across the soil. She giggled with delight, her tiny hands clapping at the sight of every new discovery.
    
    She had seen so much, learned so many new things, and had so much fun. And though she was just a baby, Maia knew that the world was full of endless adventures waiting for her to discover.
    
     ---Maia gets ready for her next adventure. She finds a toy boat and travels out to explore the surrounding landscape to find it, not knowing what else to do in the nearby river valley. She decides to spend as little time on that boat as possible as she learns about the people living there. She will continue her quest, exploring and learning and learning and learning until she finally manages to reach Utica! As far as she knows, she has conquered all of the lands she's touched.
    
    I hope Maia continues her journey toward Utica.
    

Ok, things have advanced a lot since GPT2. Its clear to see the difference between GPT2's 124M parameters and GPT4s ~1.7T+ parameters. Let's try a more powerful 8b model. This is obviously at a major dissavantage to GPT4 regarding model size, but can be run locally on a GPU


```python
generator = pipeline("text-generation",
                    model="abacusai/Llama-3-Smaug-8B",
                    max_length=500)
response = "Maia gets ready for her next adventure. She finds a toy boat and"
prompt = text + "\n\n"+ response
```


    Downloading shards:   0%|          | 0/4 [00:00<?, ?it/s]



    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]



```python
outputs = generator(prompt)
print(outputs[0]['generated_text'])
```

    Once upon a time, in a cozy little house at the edge of a bustling town Utica in New York State, there lived a baby girl named Maia. Maia wasn't just any baby girl; she was curious, smart, and had a knack for getting into all sorts of funny adventures.
    
    One sunny morning, Maia woke up with a brilliant idea. She decided it was the perfect day to explore the world beyond her crib. With a determined look in her bright eyes, she wiggled her way out of bed and toddled towards the door.
    
    First stop: the kitchen. Maia's nose twitched at the delicious smells wafting from the oven. She reached up on her tiptoes, managing to grab a cookie from the counter. "Mmm, breakfast of champions," she thought with a giggle as she munched on her treat.
    
    Next, she made her way to the living room, where her family's cat, Whiskers, was napping. Maia poked Whiskers gently, curious about the fluffy creature. Whiskers opened one eye, gave a lazy yawn, and decided that Maia's curiosity was too much for a morning nap. With a flick of his tail, he trotted off, leaving Maia to find her next adventure.
    
    The garden was full of wonders. Maia found flowers to smell, butterflies to chase, and even a friendly worm wriggling its way across the soil. She giggled with delight, her tiny hands clapping at the sight of every new discovery.
    
    She had seen so much, learned so many new things, and had so much fun. And though she was just a baby, Maia knew that the world was full of endless adventures waiting for her to discover.
    
    Maia gets ready for her next adventure. She finds a toy boat and decides to sail it across the kitchen floor. She uses a spoon as a sail and a piece of cloth as a flag. She laughs with joy as the boat "sails" across the floor, making her family laugh too.
    
    Maia's adventures don't stop there. She decides to explore the backyard. She finds a small puddle and decides to make it her own "ocean." She uses a stick as a "paddle" and a leaf as a "lifeboat." She splashes and giggles, making the most of her "sailing" adventure.
    
    As the day comes to an end, Ma
    

Again not bad given that it is a local model running off of a single gpu!!


I hope this was helpful!

Best,
Dan Baissa
