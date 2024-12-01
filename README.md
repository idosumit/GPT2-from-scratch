# GPT-2 From Scratch

Table of contents:

- [INTRODUCTION](#introduction)
- [SETUP AND INSTALLATION](#setup-and-installation)
- [RUNNING THE CODE](#running-the-code)
- [COMPLETED UPDATES](#completed-updates)
- [TODO](#todo)
- [REFERENCES](#references)

---

## INTRODUCTION

I've always wanted to create a GPT-2 model from scratch, and this is my attempt at it. What I have in this repository is a fully-working GPT-2 model that can generate texts based on a given input and expected length of the text completion.

> [!IMPORTANT]
> **Update of Nov 30, 2024 (happy birthday, ChatGPT!):**\
> I've managed to initialize the model and although it generates output texts based on the input, they are utter gibberish. This is because the weights have not been trained yet. I'm currently working on the training process. Although the model is not very useful at the moment, it's a nice starting point as a base for further development. I'm also learning a lot here, so expect the code to change a bit in the future.

## SETUP AND INSTALLATION

Ensure you have `pip` installed in your computer before you begin.

[pip installation](https://pip.pypa.io/en/stable/installation/)

To start, use the terminal to go to the directory where you want to clone the repository and clone it:

```bash
git clone https://github.com/idosumit/GPT2-from-scratch.git # clone the repository

```

Navigate inside the repository:
```bash
cd GPT2-from-scratch # navigate to the repository
```

Create a virtual environment:

*for Windows and Linux:*
```bash
python -m venv venv_gpt2 # create a virtual environment
```

*for macOS:*
```zsh
python3 -m venv venv_gpt2
```

Activate the virtual environment:
```bash
source venv_gpt2/bin/activate # activate the virtual environment
```

Finally, install all the dependencies:
```bash
pip install -r requirements.txt # install the dependencies
```

And that's it! We have set up the enviroment.

## RUNNING THE CODE

You can change the input text as well as the desired length of the output text in the [generatetext.py](./generatetext.py) file (lines 20 and 200 respectively). Based on the input text, the model will generate a completion of the text of the desired length.

To run the code, you can use the following command:

```python
python generatetext.py
```

This will generate a text completion based on the input text and the desired length of the output text. The output will be printed to the console.

It looks something like the following at the moment (based on `start_context`="Happy birthday to" and `max_new_tokens`=200):

![generatedtext](./assets/gibberish.png)

---

## COMPLETED UPDATES

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

---

## REFERENCES

This has been made possible largely due to the book "[Build a Large Language Model from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)" by Sebastian Raschka. God bless the man.

Additional resources that I referred to:
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), the original GPT-2 paper
- [Let's build a GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=9s) by Andrej Karpathy
- [Attention is all you need](https://arxiv.org/abs/1706.03762), the original Transformer paper
- [Understanding LLMs: A Comprehensive Overview from Training to Inference](https://arxiv.org/abs/2401.02038)
- [The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) by Lilian Weng
