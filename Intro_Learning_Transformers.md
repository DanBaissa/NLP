> # Using Transformers on Natural Language Data Locally!
>
> **by Daniel K Baissa**

ChatGPT has put Transformer models on the map. But there are actually many off-the-shelf transformer models to use. The goal here is to show some of the basics of using these models and what they can do. The attention mask that transformer models have really gives them an edge in time series data such as Natural Language Processing (NLP). So let's explore some use cases!

## Table of Contents

1.  [Set up your environment](#set-up-your-environment)
2.  [A Story to Practice NLP On](#a-story-to-practice-nlp-on)
3.  [Text Classification](#text-classification)
4.  [Named Entity Recognition](#named-entity-recognition)
5.  [Question Answering](#question-answering)
6.  [Summarization](#summarization)
7.  [Translation](#translation)
8.  [Text Generation](#text-generation)

## Set up your environment {#set-up-your-environment}

``` python
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

```         
CUDA is available. GPU is ready for use.
```

## A Story to Practice NLP On {#a-story-to-practice-nlp-on}

Let's use a story to have the models apply these functions. I used Chatgpt to throw together a quick story for us to use. It created this cute story about a baby girl named Maia:


``` python
text = """Once upon a time, in a cozy little house at the edge of a bustling town Utica in New York State, there lived a baby girl named Maia. Maia wasn't just any baby girl; she was curious, smart, and had a knack for getting into all sorts of funny adventures.

One sunny morning, Maia woke up with a brilliant idea. She decided it was the perfect day to explore the world beyond her crib. With a determined look in her bright eyes, she wiggled her way out of bed and toddled towards the door.

First stop: the kitchen. Maia's nose twitched at the delicious smells wafting from the oven. She reached up on her tiptoes, managing to grab a cookie from the counter. "Mmm, breakfast of champions," she thought with a giggle as she munched on her treat.

Next, she made her way to the living room, where her family's cat, Whiskers, was napping. Maia poked Whiskers gently, curious about the fluffy creature. Whiskers opened one eye, gave a lazy yawn, and decided that Maia's curiosity was too much for a morning nap. With a flick of his tail, he trotted off, leaving Maia to find her next adventure.

The garden was full of wonders. Maia found flowers to smell, butterflies to chase, and even a friendly worm wriggling its way across the soil. She giggled with delight, her tiny hands clapping at the sight of every new discovery.

She had seen so much, learned so many new things, and had so much fun. And though she was just a baby, Maia knew that the world was full of endless adventures waiting for her to discover."""
```

## Text Classification {#text-classification}

I am going to start with some basic sentiment analysis. I will use a transformer from Hugging Face🤗. Let's start by importing the transformer library.

``` python
from transformers import pipeline
import pandas as pd
```

To classify the text, I will use the `distilbert-base-uncased-finetuned-sst-2-english` model

``` python
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)
```

Now that we have the classifier, all we need to do is classify the text as negative or positive. Let's use pandas to make this into a table so it will look nice.

``` python
outputs = classifier(text)
pd.DataFrame(outputs)
```

<div>

```{=html}
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
```
+---+----------+----------+
|   | label    | score    |
+===+==========+==========+
| 0 | POSITIVE | 0.999142 |
+---+----------+----------+

</div>

Here, we can see that this is labeled as positive with an extremely high degree of confidence. These findings are not surprising given that the prompt given that this is a cute story about a baby!

## Named Entity Recognition {#named-entity-recognition}

We successfully estimated the story's sentiment, which can be very handy when reading survey or feedback responses. 

It is also nice to know what the text is actually talking about. For example, who or what is the text about? Here, we will use a named entity recognition model, `ner`. Given that this is a fast overview of these tasks, we will use the default model `dbmdz/bert-large-cased-finetuned-conll03-english`. That should be sufficient for this task


``` python
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
revision = "f2482bf"
ner_tagger = pipeline("ner", model=model_name, revision=revision, aggregation_strategy="simple")
```

``` python
outputs = ner_tagger(text)
pd.DataFrame(outputs)
```

<div>

```{=html}
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
```
+----+--------------+----------+----------------+-------+------+
|    | entity_group | score    | word           | start | end  |
+====+==============+==========+================+=======+======+
| 0  | LOC          | 0.988708 | Utica          | 72    | 77   |
+----+--------------+----------+----------------+-------+------+
| 1  | LOC          | 0.982091 | New York State | 81    | 95   |
+----+--------------+----------+----------------+-------+------+
| 2  | PER          | 0.987294 | Maia           | 127   | 131  |
+----+--------------+----------+----------------+-------+------+
| 3  | PER          | 0.990340 | Maia           | 133   | 137  |
+----+--------------+----------+----------------+-------+------+
| 4  | PER          | 0.982831 | Maia           | 273   | 277  |
+----+--------------+----------+----------------+-------+------+
| 5  | PER          | 0.973810 | Maia           | 512   | 516  |
+----+--------------+----------+----------------+-------+------+
| 6  | PER          | 0.954940 | Whiskers       | 809   | 817  |
+----+--------------+----------+----------------+-------+------+
| 7  | PER          | 0.965617 | Maia           | 832   | 836  |
+----+--------------+----------+----------------+-------+------+
| 8  | PER          | 0.918289 | Whiskers       | 843   | 851  |
+----+--------------+----------+----------------+-------+------+
| 9  | PER          | 0.941341 | Whiskers       | 895   | 903  |
+----+--------------+----------+----------------+-------+------+
| 10 | PER          | 0.972468 | Maia           | 955   | 959  |
+----+--------------+----------+----------------+-------+------+
| 11 | PER          | 0.945366 | Maia           | 1054  | 1058 |
+----+--------------+----------+----------------+-------+------+
| 12 | PER          | 0.962110 | Maia           | 1120  | 1124 |
+----+--------------+----------+----------------+-------+------+
| 13 | PER          | 0.971372 | Maia           | 1422  | 1426 |
+----+--------------+----------+----------------+-------+------+

</div>

Here, we see that Maia is the most common entity in the text. This is, of course, good because she is the main subject of most of the text. Whiskers the cat, Utica, and New York State also show up as named entities. We can also see that it groups the entities. Maia is listed as a person, and Utica and New York appear as Locations. Funnily enough, the model infers that Whiskers the cat is a person.

## Question Answering {#question-answering}

For question answering, this model will respond with a span of the corresponding text to answer the question that we give it. This is useful because you can identify sections of the data without asking the model for an opinion, so to speak. This should mitigate hallucinations because it does not force the model to be generative.

Let's start with an easy question and ask where Maia lives.

``` python
model_name = "distilbert/distilbert-base-cased-distilled-squad"
revision = "626af31"


reader = pipeline("question-answering", model=model_name, revision=revision)
question = "Where did Maia live?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
```

<div>

```{=html}
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
```
+---+----------+-------+-----+-------------------------+
|   | score    | start | end | answer                  |
+===+==========+=======+=====+=========================+
| 0 | 0.434803 | 72    | 95  | Utica in New York State |
+---+----------+-------+-----+-------------------------+

</div>

Here, we can see that it correctly identified Utica, New York, as the correct location. 

Let's see if it can handle a slightly more complicated question. We know the `dbmdz/bert-large-cased-finetuned-conll03-english` model identified Whiskers as a person. I wonder if it will know if Whiskers is Maia's cat.


``` python
reader = pipeline("question-answering", model=model_name, revision=revision)
question = "What is Maia's cat's name?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
```

<div>

```{=html}
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
```
+---+----------+-------+-----+----------+
|   | score    | start | end | answer   |
+===+==========+=======+=====+==========+
| 0 | 0.995755 | 809   | 817 | Whiskers |
+---+----------+-------+-----+----------+

</div>

Here we have a high degree of confidence that it is Whiskers, which is correct! Ironically it performed better at identifying the name of the cat than the location. Still these were easy questions to ask. We can push this model to the limit in the future.

## Summarization {#summarization}

Now, we will slowly step into generative text. Here, we will use `sshleifer/distilbart-cnn-12-6` to summarize the text. This model will actually have to read the text and then generate a text response. Let's summarize this story.

``` python
model_name = "sshleifer/distilbart-cnn-12-6"
revision = "a4f8f3e"
```

I will set max_length to 75 to ensure that this stays a short summary.

``` python
summarizer = pipeline("summarization", model=model_name, revision=revision)
outputs = summarizer(text, max_length=75, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
```

```         
 A baby girl named Maia was born in Utica, New York State. She was curious, smart, smart and had a knack for getting into funny adventures. One sunny morning, Maia woke up with a brilliant idea to explore the world beyond her crib. She found flowers to smell, butterflies to chase, and even a friendly worm in the garden.
```

Not bad at all for a quick model run locally! Ironically it might be a good kids story on its own.

## Translation {#translation}

Let's see if we can translate this text into multiple languages! Again, we are using basic models.

``` python
translator = pipeline("translation_en_to_de", 
                      model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]["translation_text"])
```

```         
Es war einmal, in einem gemütlichen kleinen Haus am Rande einer geschäftigen Stadt Utica in New York State, lebte ein Baby Mädchen namens Maia. Maia war nicht nur ein Baby Mädchen; sie war neugierig, klug, und hatte ein Kniff für die Aufnahme in alle Arten von lustigen Abenteuern. Eines sonnigen Morgens, erwachte Maia mit einer brillanten Idee. Sie entschied, es war der perfekte Tag, um die Welt jenseits ihrer Krippe zu erkunden. Mit einem entschlossenen Blick in ihren hellen Augen, sie wackelte ihren Weg aus dem Bett und kippte in Richtung der Tür. Erste Station: die Küche. Maias Nase verzauberte an den köstlichen Gerüchen wafting aus dem Ofen. Sie erreichte auf ihren Zehenspitzen, um einen Keks aus der Theke zu greifen. "Mmm, Frühstück der Champions", dachte sie mit einem kichern, als sie auf ihrem Leckerling spritzte. Weiter machte sie ihren Weg zum Wohnzimmer, wo ihre Familie, Whiskers, war napping. Maia poked Whikers sanft, neugierig auf die Maiappy-Kette.
```

### Create the translation pipeline for English to Arabic

``` python
translator = pipeline("translation_en_to_ar", 
                      model="Helsinki-NLP/opus-mt-en-ar")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]["translation_text"])
```

```         
في وقت من الأوقات، في منزل صغير على حافة مدينة أوتيكا في ولاية نيويورك، عاشت طفلة اسمها مايا. مايا لم تكن مجرد أي طفلة؛ كانت غريبة، ذكية، وكان لديها مُسْتَعِدّة للحصول على جميع أنواع من الرائحات المُضحكة. في صباح مشمس، استيقظت مايا بفكرة رائعة. قررت أنه اليوم المثالي لاستكشاف العالم خارج سريرها. مع نظرة حاسمة في عينيها المشرقة، كانت تهتز طريقها للخروج من السرير والتدحرج نحو الباب. أولاً، كانت:
```

## Text Generation {#text-generation}

Now for the thing that transformer models have received a lot of hype for! Despite being made initially for translation services, their usage for generative language is famous. Let's use GPT2, which is open-source. This model has only 124M parameters, so we should have too high of expectations. 

``` python
generator = pipeline("text-generation",
                    model="openai-community/gpt2")
response = "Maia gets ready for her next adventure. She finds a toy boat and"
prompt = text + "\n\n ---"+ response
```

``` python
outputs = generator(prompt, max_length=500)
print(outputs[0]['generated_text'])
```

```         
Once upon a time, in a cozy little house at the edge of a bustling town Utica in New York State, there lived a baby girl named Maia. Maia wasn't just any baby girl; she was curious, smart, and had a knack for getting into all sorts of funny adventures.

One sunny morning, Maia woke up with a brilliant idea. She decided it was the perfect day to explore the world beyond her crib. With a determined look in her bright eyes, she wiggled her way out of bed and toddled towards the door.

First stop: the kitchen. Maia's nose twitched at the delicious smells wafting from the oven. She reached up on her tiptoes, managing to grab a cookie from the counter. "Mmm, breakfast of champions," she thought with a giggle as she munched on her treat.

Next, she made her way to the living room, where her family's cat, Whiskers, was napping. Maia poked Whiskers gently, curious about the fluffy creature. Whiskers opened one eye, gave a lazy yawn, and decided that Maia's curiosity was too much for a morning nap. With a flick of his tail, he trotted off, leaving Maia to find her next adventure.

The garden was full of wonders. Maia found flowers to smell, butterflies to chase, and even a friendly worm wriggling its way across the soil. She giggled with delight, her tiny hands clapping at the sight of every new discovery.

She had seen so much, learned so many new things, and had so much fun. And though she was just a baby, Maia knew that the world was full of endless adventures waiting for her to discover.

 ---Maia gets ready for her next adventure. She finds a toy boat and travels out to explore the surrounding landscape to find it, not knowing what else to do in the nearby river valley. She decides to spend as little time on that boat as possible as she learns about the people living there. She will continue her quest, exploring and learning and learning and learning until she finally manages to reach Utica! As far as she knows, she has conquered all of the lands she's touched.

I hope Maia continues her journey toward Utica.
```

Ok, let's start by stating the obvious. Clearly, things have advanced considerably since GPT2. The difference between GPT2's 124M parameters and GPT4s \~1.7T+ parameters is extremely apparent. This is not to say that GPT4 is optimized, and in fact, the GPT4 model is likely bloated to some degree. But 124M in this configuration unmistakably underperforms. Let's try a more powerful 8b model. This is obviously at a significant disadvantage to GPT4 regarding model size, but it can be run locally on a GPU, and a local computer is an advantage on its own.

``` python
generator = pipeline("text-generation",
                    model="abacusai/Llama-3-Smaug-8B",
                    max_length=500)
response = "Maia gets ready for her next adventure. She finds a toy boat and"
prompt = text + "\n\n"+ response
```

```         
Downloading shards:   0%|          | 0/4 [00:00<?, ?it/s]



Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
```

``` python
outputs = generator(prompt)
print(outputs[0]['generated_text'])
```

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
```

Again not bad given that it is a local model running off of a single gpu!!

I hope this was helpful!

Best, Dan Baissa
