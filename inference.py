import torch
from whisper import load_model
from functions import mass_transcribe
from yaml import safe_load
from munch import munchify
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("filename", help="Audio file name")
args = parser.parse_args()

with open("config.yaml", "rt") as f:
    config = safe_load(f)
    config = munchify(config)

# setting up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(config.whisper.model_name).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
checkpoint = torch.load(config.inference.checkpoint)
model.load_state_dict(checkpoint["model_state_dict"])
mass_transcribe([args.filename], model)
