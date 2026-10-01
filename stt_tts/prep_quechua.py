from datasets import load_dataset, Audio
from transformers import WhisperProcessor

print("1. Loading Whisper Processor...")
# The processor handles both audio feature extraction and text tokenization
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", 
    language="spanish", 
    task="transcribe"
)

print("2. Loading Dataset...")
# Example using an open-source Quechua ASR dataset (ctaguchi/killkan)
# If you have your own local folder of audio and a metadata.csv, use:
# dataset = load_dataset("audiofolder", data_dir="./my_quechua_data")
dataset = load_dataset("ctaguchi/killkan", split="train")

print("3. Resampling to 16kHz...")
# Whisper requires exactly 16000Hz. This cast applies on-the-fly.
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # 1. Load the resampled audio data
    audio = batch["audio"]

    # 2. Extract log-Mel features from the audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # 3. Encode the target text to label IDs
    # Note: Change "sentence" to whatever your dataset's text column is named
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch

print("4. Extracting features and tokenizing (this will take a while)...")
# Map the function across the dataset and remove raw columns to save memory
prepared_dataset = dataset.map(
    prepare_dataset, 
    remove_columns=dataset.column_names, 
    num_proc=4 # Uses 4 CPU cores to speed this up
)

print("5. Saving processed dataset to disk...")
prepared_dataset.save_to_disk("./quechua_whisper_ready")
print("Done! Dataset is ready for training.")