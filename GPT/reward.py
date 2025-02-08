from trl import RewardTrainer, RewardConfig
from transformers import AutoTokenizer, AutoModel

# Suppose we have a dataset with columns: 
# [ "prompt", "completion_1", "completion_2", "chosen" (0 or 1) ]
pairwise_dataset = load_dataset("your_preference_dataset")["train"]  # just an example

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# For a reward model, you often start with a backbone that is pretrained or SFT'd
base_model = AutoModel.from_pretrained("gpt2")

# We define a RewardConfig that might specify a new linear head dimension of 1
reward_config = RewardConfig(
    model_name="gpt2",
    num_labels=1,
    # Additional reward config parameters if needed
)

def preprocess_reward_fn(examples):
    # We tokenize prompt+completion_1, prompt+completion_2
    texts_1 = [p + c for p, c in zip(examples["prompt"], examples["completion_1"])]
    texts_2 = [p + c for p, c in zip(examples["prompt"], examples["completion_2"])]

    batch_1 = tokenizer(texts_1, truncation=True, padding="longest", max_length=512)
    batch_2 = tokenizer(texts_2, truncation=True, padding="longest", max_length=512)

    # "chosen" indicates which completion is preferred: 0 means completion_1, 1 means completion_2
    labels = examples["chosen"]
    return {
        "input_ids_1": batch_1["input_ids"],
        "attention_mask_1": batch_1["attention_mask"],
        "input_ids_2": batch_2["input_ids"],
        "attention_mask_2": batch_2["attention_mask"],
        "labels": labels
    }

pairwise_dataset = pairwise_dataset.map(preprocess_reward_fn, batched=True)

reward_trainer = RewardTrainer(
    model=base_model, 
    args=reward_config,
    train_dataset=pairwise_dataset,
    eval_dataset=None,  # or a valid dataset if you have it
    tokenizer=tokenizer
)

reward_trainer.train()

reward_trainer.save_model("reward-model-checkpoint")
