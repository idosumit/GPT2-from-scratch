# GPT-2 From Scratch

## Introduction

I've always wanted to create a GPT-2 model from scratch, and this is my attempt at it. What I have in this repository is a fully-working GPT-2 model that can generate texts based on a given input and expected length of the text completion.

> [!IMPORTANT]
> As of Nov 30, 2024 (happy birthday, ChatGPT!), I've managed to initialize the model and although it generates output texts based on the input, they are utter gibberish. This is because the weights have not been trained yet. I'm currently working on the training process. Therefore, the model is not very useful at the moment, but it's a nice starting point as a base for further development. Also, I'm learning a lot here, so expect the code to change a bit in the future.

## Setup and installation

To start, use the terminal to go to the directory where you want to clone the repository, clone it, create a virtual environment, and install the dependencies:

```
git clone https://github.com/idosumit/GPT2-from-scratch.git # clone the repository
cd GPT2-from-scratch # navigate to the repository
python -m venv venv_gpt2 # create a virtual environment
source venv_gpt2/bin/activate # activate the virtual environment
pip install -r requirements.txt # install the dependencies
```

## Running the code

You can change the input text as well as the desired length of the output text in the [generatetext.py](./generatetext.py) file (lines 20 and 200 respectively). Based on the input text, the model will generate a completion of the text of the desired length.

To run the code, you can use the following command:

```python
python generatetext.py
```

This will generate a text completion based on the input text and the desired length of the output text. The output will be printed to the console.

It looks something like this at the moment(based on start_context="Happy birthday to" and max_new_tokens=200):

![generatedtext](./assets/gibberish.png)

---

## COMPLETED

- [x] Multi-head attention
- [x] Feed-forward network
- [x] Gelu activation function
- [x] Layer normalization
- [x] Transformer block
- [x] GPT-2 model (124M)
- [x] Text generation (simple gibberish)

---

## TODO

- [ ] Pretraining
  - [ ] Cross-entropy loss
  - [ ] Backpropagation
  - [ ] Update weights
- [ ] Fine-tuning
  - [ ] for classification
  - [ ] for following instructions
