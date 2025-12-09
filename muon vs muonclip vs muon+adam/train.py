import torch
from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = load_dataset(dataset_name, split="train")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager"
)

from trl import SFTConfig
from .optimizers import MuonClip, MuonPlusAdamW, Muon

optimizer_type = "muon+adam"  # Options: "muon+adam", "muonclip", "muon"

if optimizer_type == "muon+adam":
    optimizer = MuonPlusAdamW(model.parameters(), lr=5e-5)
elif optimizer_type == "muonclip":
    optimizer = MuonClip(model.parameters(), lr=5e-5)
    optimizer.set_model(model) 
elif optimizer_type == "muon":
    optimizer = Muon(model.parameters(), lr=5e-5)
else:
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")

lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=40)

training_args = SFTConfig(
    dataset_num_proc = 24,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 1,
    num_train_epochs = 1,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",
    output_dir="output",
    save_steps=100,
    save_total_limit=2,
    max_length=4096,
    push_to_hub=False
)

from trl import SFTTrainer 

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optimizer, lr_scheduler),
)

trainer.train()

trainer.save_model("output-final")
