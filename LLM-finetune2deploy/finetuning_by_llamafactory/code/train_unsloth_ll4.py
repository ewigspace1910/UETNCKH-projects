import logging, os, time, torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,          # prefer explicit TrainingArguments for clarity
    logging as hf_logging,
    TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ------------------------------------------------------------------
# 1 · LOGGING SET‑UP
# ------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),                 # stdout
        logging.FileHandler("logs/train.log")    # file
    ],
    level=logging.INFO
)
log = logging.getLogger("qlora")

# give Hugging Face/transformers the same verbosity
hf_logging.set_verbosity_info()

# ------------------------------------------------------------------
# 2 · DATA + MODEL
# ------------------------------------------------------------------
#model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model_name = "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"   # or "meta-llama/Meta-Llama-3-8B"
log.info(f"Loading dataset and tokenizer for {model_name}")

dataset = load_dataset("json", data_files="data/gra.json", split="train")
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token

log.info("Loading model in 4‑bit … this can take a minute")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="cpu"
)

peft_cfg = LoraConfig(
    r=32, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)
log.info("LoRA adapters added")

# ------------------------------------------------------------------
# 3 · OPTIONAL GPU‑MEM CALLBACK
# ------------------------------------------------------------------
class GPUStatsCallback(TrainerCallback):
    def on_log(self, args:TrainingArguments, state:TrainerState,
               control:TrainerControl, **kwargs):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserv = torch.cuda.memory_reserved() / 1024**3
            logging.info(f"GPU mem  alloc={alloc:.1f} GiB, reserved={reserv:.1f} GiB")

# ------------------------------------------------------------------
# 4 · TRAINER
# ------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    logging_dir="logs",
    report_to="none"        # disable wandb / tensorboard auto‑uploads
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=2048,
    tokenizer=tok,
    args=training_args,
    callbacks=[GPUStatsCallback()]
)

# ------------------------------------------------------------------
# 5 · GO!
# ------------------------------------------------------------------
log.info("Starting training …")
start = time.time()
trainer.train()
elapsed = (time.time() - start) / 60
log.info(f"Training finished in {elapsed:.1f} min")

model.save_pretrained("outputs/saves/qlora-llama4")
log.info("Model saved to outputs/saves/qlora-llama4")
