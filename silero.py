import torch
from glob import glob # for file paths 


# Device selection: Use CPU for inference
device = torch.device('cpu') 

# Model loading: Load the Silero STT model from torch hub
# repo_or_dir: The repository to load from
# model: The specific model name ('silero_stt')
# language: The language of the model ('en' for English)
# device: The device to run the model on
model, decoder, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_stt',
    language='en',
    device=device
)

# Utils loading: Unpack utility functions provided by the model
(read_batch, split_into_batches, read_audio, prepare_model_input) = utils

# File selection: Find the specific audio file to process
test_files = glob(r'C:\Users\ayham\Speech-Recognation-App\en_sample.wav')

# Batching: Split the files into batches of size 1
batches = split_into_batches(test_files, batch_size=1)

# Processing the first batch
# read_batch: Reads the audio files in the batch
audio = read_batch(batches[0])

# prepare_model_input: Converts audio to a tensor suitable for the model
input_tensor = prepare_model_input(audio, device=device)

# Inference: Run the model on the input tensor
output = model(input_tensor)
  
# Decoding: Convert the model output to text
for example in output:
    print(decoder(example.cpu()))
