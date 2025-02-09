
## Chatbot DPO
## code running on DC 107 number 30.

## How to run the code:

* SFT model
    Python sftmodel.py

* reward model
    python rewardmodel.py

* dpo model
    python dpotrainer.py


## model inference output samples

    /model_inference
    Base model sample: Base 
    SFT model sample: SFT
    DPO model sample: DPO

## Code files

* sft model(which will be used as policy model in RLHF)

to train a SFT(supervised fine tuning) model use a pretrained base model

* sftperplexity

compare the perplexity of SFT model and the base model, expected result would be SFT model has less perplexity.
less perplexity means model is more confident for its prediction.

* reward model

to train a reward model from a based pretrained model, similar or exactly the same model which used to train the sft model

* reward evaluation

evaluate the performance of reward model, by predicting the score of chosen and rejected data column, 
the score of chosen will be slightly larger than score of rejected. 

* dpo trainer

to finetuning the SFT model with DPO, with or without reward model, depends on the datasets(whether including preference score or not)

* dpoeva

evaluating the model performance, using multiple ways, like perplexity, compare reward score and more.



## Model 
* base model
Qwen/Qwen2.5-0.5B

* SFT model
Qwen/Qwen2.5-0.5B-SFT/checkpoint-1975

* Reward model
Qwen2.5-0.5B-Reward/checkpoint-182

* DPO model
Qwen2.5-0.5B-DPO/checkpoint-5000




