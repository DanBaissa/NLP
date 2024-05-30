# Diagnosing Autism by Reading Simulated Doctors' Notes

Autism is a complex medical condition which is described by certain traits in how the patient interacts with others. Doctors on the other hand document their interactions with patients. This implies that it might be possible to create a transformer model that can read the doctor's notes and classify if a patient has Autism. This of course will be imperfect in practice because the people write the notes are themselves imperfect reporters given that they are human. Nonetheless, this might be helpful. Given that medical notes are difficult and expensive to gain and will require IRB approval, here I will simulate doctor's notes using a GPT model, then train a model based on the synthetic data to see if it is possible to do.


```python
# Install necessary libraries if not already installed
libraries = ["openai", "transformers", "shap", "torch", "pandas"]

for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        !pip install {lib}
        __import__(lib)

# Import necessary libraries
import openai
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
import shap
import os
```


```python
openai.api_key = os.getenv("OPENAI_API_KEY")
```

Checking the availible models


```python
elist=openai.models.list()

import pandas as pd

model_data = []

# Iterate through each model in elist and collect the required information
for model in elist:
    model_info = {
        'id': model.id,
        'created': model.created,
        'object': model.object,
        'owned_by': model.owned_by
    }
    model_data.append(model_info)

# Create a DataFrame from the collected data
df = pd.DataFrame(model_data)

# Sort the DataFrame by the 'id' column
df_sorted = df.sort_values(by='id')

# Display the sorted DataFrame
df_sorted
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
      <th>id</th>
      <th>created</th>
      <th>object</th>
      <th>owned_by</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>babbage-002</td>
      <td>1692634615</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dall-e-2</td>
      <td>1698798177</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>0</th>
      <td>dall-e-3</td>
      <td>1698785189</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>3</th>
      <td>davinci-002</td>
      <td>1692634301</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>19</th>
      <td>gpt-3.5-turbo</td>
      <td>1677610602</td>
      <td>model</td>
      <td>openai</td>
    </tr>
    <tr>
      <th>18</th>
      <td>gpt-3.5-turbo-0125</td>
      <td>1706048358</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gpt-3.5-turbo-0301</td>
      <td>1677649963</td>
      <td>model</td>
      <td>openai</td>
    </tr>
    <tr>
      <th>25</th>
      <td>gpt-3.5-turbo-0613</td>
      <td>1686587434</td>
      <td>model</td>
      <td>openai</td>
    </tr>
    <tr>
      <th>13</th>
      <td>gpt-3.5-turbo-1106</td>
      <td>1698959748</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gpt-3.5-turbo-16k</td>
      <td>1683758102</td>
      <td>model</td>
      <td>openai-internal</td>
    </tr>
    <tr>
      <th>29</th>
      <td>gpt-3.5-turbo-16k-0613</td>
      <td>1685474247</td>
      <td>model</td>
      <td>openai</td>
    </tr>
    <tr>
      <th>15</th>
      <td>gpt-3.5-turbo-instruct</td>
      <td>1692901427</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>14</th>
      <td>gpt-3.5-turbo-instruct-0914</td>
      <td>1694122472</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>11</th>
      <td>gpt-4</td>
      <td>1687882411</td>
      <td>model</td>
      <td>openai</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gpt-4-0125-preview</td>
      <td>1706037612</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>12</th>
      <td>gpt-4-0613</td>
      <td>1686588896</td>
      <td>model</td>
      <td>openai</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gpt-4-1106-preview</td>
      <td>1698957206</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>27</th>
      <td>gpt-4-1106-vision-preview</td>
      <td>1711473033</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>23</th>
      <td>gpt-4-turbo</td>
      <td>1712361441</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>22</th>
      <td>gpt-4-turbo-2024-04-09</td>
      <td>1712601677</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gpt-4-turbo-preview</td>
      <td>1706037777</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>28</th>
      <td>gpt-4-vision-preview</td>
      <td>1698894917</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>30</th>
      <td>gpt-4o</td>
      <td>1715367049</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>31</th>
      <td>gpt-4o-2024-05-13</td>
      <td>1715368132</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>21</th>
      <td>text-embedding-3-large</td>
      <td>1705953180</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>24</th>
      <td>text-embedding-3-small</td>
      <td>1705948997</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>26</th>
      <td>text-embedding-ada-002</td>
      <td>1671217299</td>
      <td>model</td>
      <td>openai-internal</td>
    </tr>
    <tr>
      <th>16</th>
      <td>tts-1</td>
      <td>1681940951</td>
      <td>model</td>
      <td>openai-internal</td>
    </tr>
    <tr>
      <th>20</th>
      <td>tts-1-1106</td>
      <td>1699053241</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tts-1-hd</td>
      <td>1699046015</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tts-1-hd-1106</td>
      <td>1699053533</td>
      <td>model</td>
      <td>system</td>
    </tr>
    <tr>
      <th>2</th>
      <td>whisper-1</td>
      <td>1677532384</td>
      <td>model</td>
      <td>openai-internal</td>
    </tr>
  </tbody>
</table>
</div>



Ok, now let's get started simulated the sythetic data. I will use `gpt-3.5-turbo-0125` to generate the data because it is cheap and fast. With funding, GPT-4 would be a better choice. But again, real notes are ideal. 


```python
import openai
import openai
import pandas as pd
from tqdm import tqdm

client = openai.OpenAI()

# Function to generate notes using OpenAI API with chat completions
def generate_notes(content, num_notes):
    notes = []
    for _ in tqdm(range(num_notes), desc="Generating notes"):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": content
                },
                {
                    "role": "user",
                    "content": "Please generate a detailed doctor's note."
                }
            ],
            temperature=0.2,
            top_p=1
        )
        note = response.choices[0].message.content.strip()
        notes.append(note)
    return notes

# Generate autism notes
autism_content = "Write a detailed pediatrician note about a 2 year old patient diagnosed with autism."
autism_notes = generate_notes(autism_content, 1000)

# Generate non-autism notes
non_autism_content = "Write a detailed pediatrician note about a 2 year old patient."
non_autism_notes = generate_notes(non_autism_content, 1000)

# Prepare data for training
df = pd.DataFrame({
    'note': autism_notes + non_autism_notes,
    'label': [1] * len(autism_notes) + [0] * len(non_autism_notes)
})

```

    Generating notes: 100%|██████████████████████████████████████████████████████████| 1000/1000 [1:48:24<00:00,  6.50s/it]
    Generating notes: 100%|██████████████████████████████████████████████████████████| 1000/1000 [1:59:21<00:00,  7.16s/it]
    

Now the notes are saved in a dataset. This is equivilent to the notes being saved to an excel sheet with one column named `note` containing the doctors' notes, and another column named `label` which is 1 if autism and 0 otherwise. 

The next step is to split this into training, validation, and test datasets. The reason being is to have unused data for out-of-sample testing. We do not care how well our model predicts in sample. A good model should fit extremely well to the data it is trained on. The question is how well will it fit to data it has never seen before. If it can fit to data it is not trained on, then it should hopefully be reasonably accurite in practice. 


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, ClassLabel



# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Reset the index to avoid '__index_level_0__'
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Define the ClassLabel feature
class_label = ClassLabel(num_classes=2, names=['Not autism', 'Autism'])

# Convert pandas dataframes to Hugging Face datasets and set the ClassLabel feature
def convert_to_dataset(dataframe, class_label):
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.cast_column('label', class_label)
    return dataset

train_dataset = convert_to_dataset(train_df, class_label)
val_dataset = convert_to_dataset(val_df, class_label)
test_dataset = convert_to_dataset(test_df, class_label)


# Specify the directory where you want to save the DatasetDict
save_directory = './sim_notes_dataset'

# Save the DatasetDict to the specified directory
sim_notes.save_to_disk(save_directory)

print(f"DatasetDict saved to {save_directory}")
# Print the DatasetDict structure
print(sim_notes)
```


    Casting the dataset:   0%|          | 0/1400 [00:00<?, ? examples/s]



    Casting the dataset:   0%|          | 0/300 [00:00<?, ? examples/s]



    Casting the dataset:   0%|          | 0/300 [00:00<?, ? examples/s]


    DatasetDict({
        train: Dataset({
            features: ['note', 'label'],
            num_rows: 1400
        })
        validation: Dataset({
            features: ['note', 'label'],
            num_rows: 300
        })
        test: Dataset({
            features: ['note', 'label'],
            num_rows: 300
        })
    })
    

Above we can see that the training set is saved to train, validation, and test sets. The Training dataset has 1400 obs with the validation and test sets containing 300. This is pretty typical. It will take more data to fit the model than to test its accuracy. I also saved the data so we don't have to pay OpenAI again and again to generate the datasets. 


```python
from datasets import load_from_disk

save_directory = './sim_notes_dataset'
# Load the DatasetDict from the specified directory
sim_notes = load_from_disk(save_directory)

# Verify the loaded data
print(sim_notes)

```

    DatasetDict({
        train: Dataset({
            features: ['note', 'label'],
            num_rows: 1400
        })
        validation: Dataset({
            features: ['note', 'label'],
            num_rows: 300
        })
        test: Dataset({
            features: ['note', 'label'],
            num_rows: 300
        })
    })
    

## Pretrained Model

Here we will load the DistilBert checkpoint as a pretrained model and push it to the GPU if it is avaiable. This should happen automatically, but I want to be sure. 


```python
from transformers import AutoModel
import torch

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
```

    C:\Users\fa18d\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    

### Last hidden state

Next we will use the Last hidden state from the transformer model to train a simple classifier to predict autism or not. The hidden state data are basically the data sythesized by the transformer model and coverted into variables to be used in regression. In order to do that, however, we need to `tokenize` the words. This is basically turning the words or word stems into numbers so we can perform math on them. Tokenization is a really interesting idea in and of itself, turning words into numbers in a latent space is really fun but that is for a different time.  


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    texts = batch["note"]
    encoded = tokenizer(texts, padding=True, truncation=True)
    # Return the original text and labels along with the encoded texts
    return {**encoded, "label": batch["label"], "note": batch["note"]}

```


```python
sim_notes_encoded = sim_notes.map(tokenize, batched=True, batch_size=None)
```


    Map:   0%|          | 0/1400 [00:00<?, ? examples/s]



    Map:   0%|          | 0/300 [00:00<?, ? examples/s]



    Map:   0%|          | 0/300 [00:00<?, ? examples/s]



```python
print(sim_notes_encoded["train"].column_names)
```

    ['note', 'label', 'input_ids', 'attention_mask']
    

Now we have a dataset that contains the notes, labels, ids and attention mask, we can finally extract the hidden_states 


```python
def extract_hidden_states(batch):
    # Ensure all data expected to be tensors are actually tensors and then move to device
    inputs = {
        k: torch.tensor(v).to(device) if isinstance(v, list) else v.to(device)
        for k, v in batch.items() if k in tokenizer.model_input_names
    }
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}
```


```python
sim_notes_encoded.set_format("torch",
                            columns=["input_ids","attention_mask","label"])
```


```python
sim_notes_encoded_hidden = sim_notes_encoded.map(extract_hidden_states, batched=True)
```


    Map:   0%|          | 0/1400 [00:00<?, ? examples/s]



    Map:   0%|          | 0/300 [00:00<?, ? examples/s]



    Map:   0%|          | 0/300 [00:00<?, ? examples/s]


## Diagnosing Autism with Math

Now that we have access to the hidden_state, we have access to all of the data we need to train a classification model to determine of a Doctor's note contains language that is indicative of an Autism Diagnosis. To do this, I will just split the data into our X variables (independent variables) and y data, the outcome (dependant variable). 


```python
import numpy as np

X_train = np.array(sim_notes_encoded_hidden["train"]["hidden_state"])
X_valid = np.array(sim_notes_encoded_hidden["validation"]["hidden_state"])
y_train = np.array(sim_notes_encoded_hidden["train"]["label"]) 
y_valid = np.array(sim_notes_encoded_hidden["validation"]["label"])

X_train.shape, X_valid.shape
```




    ((1400, 768), (300, 768))



Here we can see that we have 768 variables across by 1400 observations for our training data. This means that each doctor's note is convered into a dataset that contains 768 variables (our independant variables). Now all that is left to do is fit the data. 

## Train a classifier


```python
from sklearn.linear_model import LogisticRegression 

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)
```




    1.0



We acheive a perfect fit on our first try. This is likely because of the GPT model or its settings. What this is saying is that it can predict with 100 percent accuracy if a patient has Autism or not. In practice I do not expect this number to be 100%.

### Comparying to dummy classifier


```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)
```




    0.49



The above is included to see how the model would perform against random "guessing". In this simulated case we had a balanced dataset so around 50/50 is expect. In practice the data will not be as perfect, and this will be more helpful.

The final step before finetuning is to see where the model is failing. To do that, I will use a confusion Matrix. This will tell us a breakdown of how the model does with false positives, true positives, false negatives, and true negatives. 


```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

labels = sim_notes["train"].features["label"].names

def plot_confusion_matrix(y_pred, y_true, labels):
    cm=confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()

y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)
```


    
![png](output_34_0.png)
    


Again, our model had a perfect fit on the data, so there is not much to see here. Still this is useful. On the y-axis (left hand side) we know the true label from the sythetic data. Autism or Not Autism. On the x-axis (bottem) we have the predictions made by the model. This is feeding the model a note, and having it pick between Autism and Not Autism based. Here we see it perfectly picked between Autism and Not Autism. In practice, I expect this will be much less accurate. Therefore, I expect we will need to finetune the model.

## Finetuning

Finetuning a model is basically starting with a good foundation and then building on top of it. In this case we will use the same model form above and just train it specifically on the data. Think of as going from a general pediatrian to getting a subspecialty. I will use accuracy and f1 statisics as a method for generating the metrics to see how well the model is performing. 


```python
from transformers import AutoModelForSequenceClassification

num_labels = 2
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1":f1}

```

    C:\Users\fa18d\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

I am using the transformers library again for this task. It is from [huggingface](https://huggingface.co/) which is one of the leaders in open source AI models and open research. To acess some of the features I will need to log into to huggingface using my api key. This will allow me to run gated models and features locally.


```python
import os
api_key = os.getenv("HF_HUB")

from huggingface_hub import login
login(token=api_key)
```

    The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.
    Token is valid (permission: read).
    Your token has been saved to C:\Users\fa18d\.cache\huggingface\token
    Login successful
    

Now that I am logged in all I have to do is finetune the model.


```python
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(sim_notes_encoded["train"])//batch_size
model_name = f"{model_ckpt}-finetuned-autism"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")
```


```python
from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=sim_notes_encoded["train"],
                  eval_dataset=sim_notes_encoded["validation"],
                  tokenizer=tokenizer)

trainer.train();

```



    <div>

      <progress value='44' max='44' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [44/44 20:33, Epoch 2/2]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.374500</td>
      <td>0.105375</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.070500</td>
      <td>0.039812</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table><p>


Here we again see perfect predictions happening. This is not too suprising given that is where we started, but it limits the model's ability to grow.








    
![png](output_45_1.png)
    



```python
model_name
```




    'distilbert-base-uncased-finetuned-autism'




```python
from transformers import Trainer, TrainingArguments


# Define the path where you want to save the model
save_path = f"A:\\ai\\models\\{model_name}"

# Save the fine-tuned model
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

```



    <div>

      <progress value='44' max='44' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [44/44 13:15, Epoch 2/2]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.021300</td>
      <td>0.008863</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.009200</td>
      <td>0.006813</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table><p>





    ('A:\\ai\\models\\distilbert-base-uncased-finetuned-autism\\tokenizer_config.json',
     'A:\\ai\\models\\distilbert-base-uncased-finetuned-autism\\special_tokens_map.json',
     'A:\\ai\\models\\distilbert-base-uncased-finetuned-autism\\vocab.txt',
     'A:\\ai\\models\\distilbert-base-uncased-finetuned-autism\\added_tokens.json',
     'A:\\ai\\models\\distilbert-base-uncased-finetuned-autism\\tokenizer.json')



## Using the Model

Now that we have created this model, let's use it to "diagnose" some simulated example patients. Here I will use a single sentance and use it to predict of the patient has Autism or not.


```python
sentence = "The patient shows signs of repetitive behavior and difficulty with social interactions."
```


```python
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the fine-tuned model and tokenizer from the specified path
model = AutoModelForSequenceClassification.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)

# Create a pipeline with the fine-tuned model
autism_pipeline = transformers.pipeline('text-classification', model=model, tokenizer=tokenizer)

# Analyze the classification of the input sentence
result = autism_pipeline(sentence)[0]
print(result)

```

    {'label': 'LABEL_1', 'score': 0.7688888311386108}
    

Here we see "Label_1" meaning it has indeed identified this text as a person with Autism. It has a score of 0.77, which says its pretty confident but not highly confident they have Autism.

But one might be interested in what words are leading to the decission of the model to classifiy it as "Autism".


```python
import shap
# explain the model on the input sentence
explainer = shap.Explainer(autism_pipeline) 
shap_values = explainer([sentence])

# visualize the first prediction's explanation for the predicted class
predicted_class = result['label']
shap.plots.text(shap_values[0, :, predicted_class])
```


      0%|          | 0/210 [00:00<?, ?it/s]



<svg width="100%" height="80px"><line x1="0" y1="33" x2="100%" y2="33" style="stroke:rgb(150,150,150);stroke-width:1" /><line x1="51.54171220337157%" y1="33" x2="51.54171220337157%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="51.54171220337157%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.4</text><line x1="41.63072039338849%" y1="33" x2="41.63072039338849%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="41.63072039338849%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.3</text><line x1="31.719728583405416%" y1="33" x2="31.719728583405416%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="31.719728583405416%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.2</text><line x1="21.808736773422346%" y1="33" x2="21.808736773422346%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="21.808736773422346%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.1</text><line x1="11.897744963439278%" y1="33" x2="11.897744963439278%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="11.897744963439278%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0</text><line x1="61.45270401335462%" y1="33" x2="61.45270401335462%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="61.45270401335462%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.5</text><line x1="71.3636958233377%" y1="33" x2="71.3636958233377%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="71.3636958233377%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.6</text><line x1="81.27468763332078%" y1="33" x2="81.27468763332078%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="81.27468763332078%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.7</text><line x1="91.18567944330384%" y1="33" x2="91.18567944330384%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="91.18567944330384%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0.8</text><line x1="11.897744963439278%" y1="33" x2="11.897744963439278%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="11.897744963439278%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">0</text><text x="11.897744963439278%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">0</text><text x="11.897744963439278%" y="10" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">base value</text><line x1="88.10225404546156%" y1="33" x2="88.10225404546156%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" /><text x="88.10225404546156%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" font-weight="bold" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">0.768889</text><text x="88.10225404546156%" y="27" font-size="13px" font-weight="bold" fill="rgb(0,0,0)" dominant-baseline="bottom" text-anchor="middle">0.768889</text><text x="88.10225404546156%" y="10" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">f<tspan baseline-shift="sub" font-size="8px">LABEL_1</tspan>(inputs)</text><rect x="8.333333250741738%" width="79.76892079471982%" y="40" height="18" style="fill:rgb(255.0, 0.0, 81.08083606031792); stroke-width:0; stroke:rgb(0,0,0)" /><line x1="70.4373953931114%" x2="88.10225404546156%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_11" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="79.26982471928648%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_11" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.178</text><svg x="70.4373953931114%" y="40" height="20" width="17.664858652350162%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">social</text>  </svg></svg><line x1="58.397153770293066%" x2="70.4373953931114%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_7" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="64.41727458170223%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_7" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.121</text><svg x="58.397153770293066%" y="40" height="20" width="12.040241622818336%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">behavior</text>  </svg></svg><line x1="47.03164265456885%" x2="58.397153770293066%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_12" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="52.71439821243096%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_12" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.115</text><svg x="47.03164265456885%" y="40" height="20" width="11.365511115724217%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">interactions</text>  </svg></svg><line x1="38.66241973775845%" x2="47.03164265456885%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_2" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="42.84703119616365%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_2" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.084</text><svg x="38.66241973775845%" y="40" height="20" width="8.3692229168104%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">patient</text>  </svg></svg><line x1="30.78372759281727%" x2="38.66241973775845%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_3" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="34.723073665287856%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_3" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.079</text><svg x="30.78372759281727%" y="40" height="20" width="7.878692144941176%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">shows</text>  </svg></svg><line x1="25.043811552435063%" x2="30.78372759281727%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_6" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="27.913769572626165%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_6" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.058</text><svg x="25.043811552435063%" y="40" height="20" width="5.739916040382209%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">repetitive</text>  </svg></svg><line x1="19.688514149884906%" x2="25.043811552435063%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_13" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="22.366162851159984%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_13" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.054</text><svg x="19.688514149884906%" y="40" height="20" width="5.355297402550157%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">.</text>  </svg></svg><line x1="15.741958635544702%" x2="19.688514149884906%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_5" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="17.715236392714804%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_5" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.04</text><svg x="15.741958635544702%" y="40" height="20" width="3.9465555143402042%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">of</text>  </svg></svg><line x1="12.938746426790706%" x2="15.741958635544702%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_1" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="14.340352531167703%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_1" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.028</text><svg x="12.938746426790706%" y="40" height="20" width="2.8032122087539957%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">The</text>  </svg></svg><line x1="10.829954127533847%" x2="12.938746426790706%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_9" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="11.884350277162277%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_9" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.021</text><svg x="10.829954127533847%" y="40" height="20" width="2.1087922992568586%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">difficulty</text>  </svg></svg><line x1="9.222159056404108%" x2="10.829954127533847%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_10" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="10.026056591968977%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_10" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.016</text><svg x="9.222159056404108%" y="40" height="20" width="1.6077950711297397%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">with</text>  </svg></svg><line x1="8.333333250741738%" x2="9.222159056404108%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_4" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2; opacity: 0"/><text x="8.777746153572924%" y="71" font-size="12px" id="_fs_asrngqcllywgnexwhqpd_ind_4" fill="rgb(255.0, 0.0, 81.08083606031792)" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">0.009</text><svg x="8.333333250741738%" y="40" height="20" width="0.8888258056623695%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">signs</text>  </svg></svg><g transform="translate(0,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(0,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(2,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(4,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(6,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-8,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-6,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-4,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><g transform="translate(-2,0)">  <svg x="9.222159056404108%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255.0, 0.0, 81.08083606031792);stroke-width:2" />  </svg></g><rect transform="translate(-8,0)" x="88.10225404546156%" y="40" width="8" height="18" style="fill:rgb(255.0, 0.0, 81.08083606031792)"/><g transform="translate(-11.5,0)">  <svg x="8.333333250741738%" y="40" height="18" overflow="visible" width="30">    <path d="M 10 -9 l 6 18 L 10 25 L 0 25 L 0 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />  </svg></g><g transform="translate(-1.5,0)">  <svg x="88.10225404546156%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="70.4373953931114%" y="40" height="20" width="17.664858652350162%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_11').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_11').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_11').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_11').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_11').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_11').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="70.4373953931114%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="58.397153770293066%" y="40" height="20" width="12.040241622818336%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_7').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_7').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_7').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_7').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_7').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_7').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="58.397153770293066%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="47.03164265456885%" y="40" height="20" width="11.365511115724217%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_12').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_12').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_12').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_12').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_12').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_12').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="47.03164265456885%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="38.66241973775845%" y="40" height="20" width="8.3692229168104%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_2').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_2').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_2').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_2').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_2').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_2').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="38.66241973775845%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="30.78372759281727%" y="40" height="20" width="7.878692144941176%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_3').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_3').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_3').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_3').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_3').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_3').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="30.78372759281727%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="25.043811552435063%" y="40" height="20" width="5.739916040382209%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_6').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_6').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_6').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_6').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_6').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_6').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="25.043811552435063%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="19.688514149884906%" y="40" height="20" width="5.355297402550157%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_13').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_13').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_13').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_13').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_13').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_13').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="19.688514149884906%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="15.741958635544702%" y="40" height="20" width="3.9465555143402042%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_5').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_5').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_5').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_5').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_5').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_5').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="15.741958635544702%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="12.938746426790706%" y="40" height="20" width="2.8032122087539957%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_1').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_1').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_1').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_1').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_1').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_1').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="12.938746426790706%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="10.829954127533847%" y="40" height="20" width="2.1087922992568586%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_9').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_9').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_9').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_9').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_9').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_9').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><g transform="translate(-1.5,0)">  <svg x="10.829954127533847%" y="40" height="18" overflow="visible" width="30">    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb(255, 195, 213);stroke-width:2" />  </svg></g><rect x="9.222159056404108%" y="40" height="20" width="1.6077950711297397%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_10').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_10').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_10').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_10').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_10').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_10').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><rect x="8.333333250741738%" y="40" height="20" width="0.8888258056623695%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_4').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_4').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_4').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_4').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_4').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_4').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /><rect x="88.10225404546156%" width="3.5644117126975394%" y="40" height="18" style="fill:rgb(0.0, 138.56128015770727, 250.76166088685727); stroke-width:0; stroke:rgb(0,0,0)" /><line x1="88.10225404546156%" x2="91.6666657581591%" y1="60" y2="60" id="_fb_asrngqcllywgnexwhqpd_ind_8" style="stroke:rgb(0.0, 138.56128015770727, 250.76166088685727);stroke-width:2; opacity: 0"/><text x="89.88445990181033%" y="71" font-size="12px" fill="rgb(0.0, 138.56128015770727, 250.76166088685727)" id="_fs_asrngqcllywgnexwhqpd_ind_8" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">-0.036</text><svg x="88.10225404546156%" y="40" height="20" width="3.5644117126975345%">  <svg x="0" y="0" width="100%" height="100%">    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">and</text>  </svg></svg><rect transform="translate(0,0)" x="88.10225404546156%" y="40" width="8" height="18" style="fill:rgb(0.0, 138.56128015770727, 250.76166088685727)"/><g transform="translate(-6.0,0)">  <svg x="91.6666657581591%" y="40" height="18" overflow="visible" width="30">    <path d="M 8 -9 l -6 18 L 8 25 L 20 25 L 20 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />  </svg></g><rect x="88.10225404546156%" y="40" height="20" width="3.5644117126975345%"      onmouseover="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_8').style.textDecoration = 'underline';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_8').style.opacity = 1;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_8').style.opacity = 1;"      onmouseout="document.getElementById('_tp_asrngqcllywgnexwhqpd_ind_8').style.textDecoration = 'none';document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_8').style.opacity = 0;document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_8').style.opacity = 0;" style="fill:rgb(0,0,0,0)" /></svg><div align='center'><div style="color: rgb(120,120,120); font-size: 12px; margin-top: -15px;">inputs</div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.0</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_0'
            style='display: inline; background: rgba(230.2941176470614, 26.505882352939775, 102.59215686274348, 0.0); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_0').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_0').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_0').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_0').style.opacity = 0;"
        ></div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.028</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_1'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.15654585066349747); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_1').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_1').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_1').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_1').style.opacity = 0;"
        >The </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.084</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_2'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.4718558130322837); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_2').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_2').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_2').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_2').style.opacity = 0;"
        >patient </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.079</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_3'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.44820756585462457); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_3').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_3').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_3').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_3').style.opacity = 0;"
        >shows </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.009</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_4'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.04618736383442265); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_4').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_4').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_4').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_4').style.opacity = 0;"
        >signs </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.04</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_5'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.219607843137255); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_5').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_5').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_5').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_5').style.opacity = 0;"
        >of </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.058</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_6'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.32208358090711037); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_6').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_6').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_6').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_6').style.opacity = 0;"
        >repetitive </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.121</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_7'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.6846900376312143); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_7').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_7').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_7').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_7').style.opacity = 0;"
        >behavior </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>-0.036</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_8'
            style='display: inline; background: rgba(30.0, 136.0, 229.0, 0.1959595959595959); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_8').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_8').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_8').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_8').style.opacity = 0;"
        >and </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.021</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_9'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.1171321053673995); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_9').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_9').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_9').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_9').style.opacity = 0;"
        >difficulty </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.016</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_10'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.08560110913052081); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_10').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_10').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_10').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_10').style.opacity = 0;"
        >with </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.178</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_11'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 1.0); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_11').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_11').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_11').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_11').style.opacity = 0;"
        >social </div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.115</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_12'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.6452762923351159); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_12').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_12').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_12').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_12').style.opacity = 0;"
        >interactions</div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.054</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_13'
            style='display: inline; background: rgba(255.0, 13.0, 87.0, 0.29843533372945136); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_13').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_13').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_13').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_13').style.opacity = 0;"
        >.</div></div><div style='display: inline; text-align: center;'
    ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'>0.0</div
        ><div id='_tp_asrngqcllywgnexwhqpd_ind_14'
            style='display: inline; background: rgba(230.2941176470614, 26.505882352939775, 102.59215686274348, 0.0); border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            } else {
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }"
            onmouseover="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_14').style.opacity = 1; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_14').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_asrngqcllywgnexwhqpd_ind_14').style.opacity = 0; document.getElementById('_fs_asrngqcllywgnexwhqpd_ind_14').style.opacity = 0;"
        ></div></div></div>


Here we can see the breakdown of the Shap values for the prediction. Shap is a concept from game thoery that basically describes the contibution of actors to forming a winning coaltion. Here the winning coalition are the words which make the model more likely to predict it as autism or not. We can see that social, interactions, and behavior are the words with the largest effect. 

In summary, this code is designed to help predict if a patient has Autism or Not simply by reading their notes. This is designed for future integration with real world data if they become available. All that would be needed to do is swap the `df` input from the GPT data to real world data and it should be possible to run the rest of this code to classify real patient data.

I hope this is helpful,

Dan Baissa
