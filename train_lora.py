import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from typing import Any, Dict, List, Union

print("1. Loading processed dataset...")
dataset = load_from_disk("./quechua_whisper_ready")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="spanish", task="transcribe")

print("2. Loading Whisper Model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# Disable caching during training
model.config.use_cache = False 

print("3. Applying LoRA Adapters...")
# Target the attention mechanisms specifically
config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 4. Define Data Collator (Handles dynamic padding for audio/text sequences)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("5. Setting up Training Arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-quechua-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1, 
    learning_rate=1e-3, # LoRA requires higher learning rates than full fine-tuning
    warmup_steps=50,
    max_steps=500, # Adjust based on dataset size
    logging_steps=25,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    predict_with_generate=True,
    generation_max_length=225,
    remove_unused_columns=False, # Required for audio data
    label_names=["labels"],
    fp16=False, 
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset, # In a real scenario, split dataset into train/eval
    eval_dataset=dataset, 
    data_collator=data_collator,
    processing_class=processor.feature_extractor,
)

print("6. Starting Training...")
trainer.train()

print("7. Saving LoRA Adapters...")
model.save_pretrained("./whisper-quechua-lora-final")
print("Training complete! Adapter weights saved.")