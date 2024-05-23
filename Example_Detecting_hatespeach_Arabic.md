># Arabic Text Classification with Transformers
> **by Daniel K Baissa**


The transformer's attention layer gives it an advantage over older
models because it is better able to determine who is doing what action
and to whom. So the AraBERT should be very helpful at classifying hate
speech.

Let's demonstrate how to use a transformer model for text classification
using Hugging Face Datasets from Hugging Face hub.


```python
from datasets import load_dataset
```

Let's begin by loading a dataset from Arabic Machine Learning a group of researchers working on Arabic NLP. 


```python
arabic_hs = load_dataset("arbml/Arabic_Hate_Speech")

columns_to_remove = ["is_off", "is_vlg", "is_vio"]

# Use the remove_columns method to update each dataset in arabic_hs
for split in arabic_hs.keys():
    arabic_hs[split] = arabic_hs[split].remove_columns(columns_to_remove)

```


```python
from arabert.preprocess import ArabertPreprocessor

model_name="aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)
```

    [2024-05-20 20:49:32,961 - farasapy_logger - WARNING]: Be careful with large lines as they may break on interactive mode. You may switch to Standalone mode for such cases.
    


```python
def preprocess_function(preprocess):
    processed_text = arabert_prep.preprocess(preprocess["tweet"])
    return {"processed_tweets": processed_text}

# Apply the function to each example in the dataset
processed_dataset = arabic_hs.map(preprocess_function)

print(processed_dataset['train']['processed_tweets'][0])
```


    Map:   0%|          | 0/8557 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1266 [00:00<?, ? examples/s]


    [مستخدم] ردي +نا ع ال+ تطنز
    

### Set up the tokenizer


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

test = processed_dataset['train']['processed_tweets'][0] 
inputs = tokenizer(test, 
                   return_tensors="pt")
```

### Load the pretrained model


```python
from transformers import AutoModel
import torch

model_ckpt = "aubmindlab/bert-base-arabertv2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)

def tokenize(batch):
    tweets = batch["processed_tweets"]
    encoded = tokenizer(tweets, padding=True, truncation=True)
    return {**encoded, "is_hate": batch["is_hate"], "processed_tweets": batch["processed_tweets"]}

```


```python
processed_dataset_encoded = processed_dataset.map(tokenize, batched=True, batch_size=None)
```


    Map:   0%|          | 0/8557 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1266 [00:00<?, ? examples/s]



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

processed_dataset_encoded.set_format("torch",
                            columns=["id","attention_mask","is_hate"])

arabic_hs_hidden = processed_dataset_encoded.map(extract_hidden_states, batched=True)
```


    Map:   0%|          | 0/8557 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1266 [00:00<?, ? examples/s]


## Feature Matrix


```python
import numpy as np

X_train = np.array(arabic_hs_hidden["train"]["hidden_state"])
X_valid = np.array(arabic_hs_hidden["validation"]["hidden_state"])
y_train = np.array(arabic_hs_hidden["train"]["is_hate"]) 
y_valid = np.array(arabic_hs_hidden["validation"]["is_hate"])

X_train.shape, X_valid.shape
```




    ((8557, 768), (1266, 768))



As we can see that the hidden states generated over 700 dimentions for each tweet.

## Fitting a model to the Hidden States results

Here we will use a simple Logit to fit the data.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict(X_valid)
accuracy = lr_clf.score(X_valid, y_valid)
weighted_f1 = f1_score(y_valid, y_pred, average='weighted')


print("Accuracy:", accuracy)
print("Weighted F1 Score:", weighted_f1)

```

    Accuracy: 0.9249605055292259
    Weighted F1 Score: 0.9208651363629278
    

Here we can see that the fit is actually pretty decent with an accuracy and weighted F1 of .92


I hope this was helpful!

Best,
Dan Baissa
