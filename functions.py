import os
import torch
import pickle
import librosa
import whisper
from functools import partial
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import pad
from re import match
import csv

class Checkpointer:
    def __init__(self, run_id: str, path: str = "checkpoints", roll: int = 3) -> None:
        """
        Contructor for the Checkpoint class

        Parameters
        ----------
        run_id : str
            name of current run
        path : str
            checkpoint dir
        path : str
            masked path for saving files
        roll : int
            number of checkpoints to keep after each save, sorted by modification time (only newest kept)

        """
        assert roll > 0, "Roll number should be larger than zero"
        self.run_id = run_id
        self.path = path
        os.makedirs(os.path.dirname(os.path.join(path, "")), exist_ok=True)
        self.roll = roll

    def __call__(self, obj: object, id=None) -> bool:
        files = [
            f
            for f in os.listdir(self.path)
            if f.startswith(self.run_id) and f.endswith(".pt")
        ]
        if id is None:
            # look for already existing checkpoints
            ids = (
                [f[len(self.run_id) + 1 :].replace(".pt", "") for f in files]
                if len(files) > 0
                else []
            )
            ids = [int(i) for i in ids]
            id = max(ids + [-1]) + 1
        path = os.path.join(self.path, f"{self.run_id}_{id}.pt")
        if os.path.isfile(path):
            raise IOError(f"Checkpoint file with id='{id}' already exists")
        torch.save(obj, path)
        if len(files) >= self.roll:
            dates = [os.path.getmtime(os.path.join(self.path, f)) for f in files]
            ixs = [
                i
                for i, m in enumerate(dates)
                if m not in sorted(dates, reverse=True)[: self.roll - 1]
            ]
            for i, file in enumerate(files):
                if i in ixs:
                    os.remove(os.path.join(self.path, file))
        return True

    def save(self, obj: object, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir, index_filepath, sample_rate=16000, device=None
    ) -> None:
        """
        Loads list of phrases and path to audio. Index filepath should be made of separate lines,
        each line consists of path to audio, phrase and speaker_id, separated by |
        Enables caching of mel spectrograms to disk for speedier processing
        """

        super().__init__()
        with open(index_filepath, "rt") as f:
            reader = csv.reader(f, delimiter=";")
            self.data = {}
            for i, line in enumerate(reader):
                self.data[i] = {"path": line[0], "text": line[1], "speaker_id": line[2]}

        self.sr = sample_rate
        self.audio_root = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        # audio
        fpath = os.path.join(self.audio_root, os.path.basename(self.data[id]["path"]))
        audio, _ = librosa.load(path=fpath, sr=self.sr, mono=True)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        # text
        label = self.data[id]["text"]
        return {
            "mel": mel,
            "label": label,
        }


def decode(tokens, tokenizer, skip_special_tokens=True, keep_timestamps=False):
    if skip_special_tokens is True and keep_timestamps is False:
        sts = tokenizer.special_tokens.values()
    elif skip_special_tokens is True and keep_timestamps is True:
        sts = [
            v
            for k, v in tokenizer.special_tokens.items()
            if not match("<\|[\d.]+\|>", k)
        ]
    else:
        sts = []
    if isinstance(tokens[0], list):
        return [
            tokenizer.decode([st for st in tokens_row if st not in sts])
            for tokens_row in tokens
        ]
    else:
        return tokenizer.decode([st for st in tokens if st not in sts])


class Collator:
    def __init__(self, tokenizer, usetimestamps=False) -> None:
        self.tokenizer = tokenizer
        # we use sot_sequence because those are different:
        # for .en models it is just 50257 <sot>
        # for multilanguage it is 50258 <sot>, 50XXX <lang token>, 50359 <transcribe>
        # + 50263 <notimestamps> if notimestamps
        if usetimestamps is False:
            self.start_tokens = list(tokenizer.sot_sequence_including_notimestamps)
        else:
            self.start_tokens = list(tokenizer.sot_sequence)
        self.pad_token = tokenizer.eot

    def __call__(self, features):
        mels = torch.stack([el["mel"] for el in features])
        labels = [
            self.start_tokens
            + self.tokenizer.encode(
                el["label"]
            )  # labels should not start from 2nd token due to right shift of output
            + [self.pad_token]  # add extra eot at the end of the string
            for el in features
        ]
        maxlen = max([len(el) for el in labels])
        diffs = [maxlen - len(el) for el in labels]

        labels = torch.stack(
            [
                pad(
                    torch.LongTensor(el),
                    (0, diff),
                    mode="constant",
                    value=-100,
                )
                for el, diff in zip(labels, diffs)
            ]
        )

        dec_input_ids = torch.roll(labels, shifts=1, dims=-1)
        dec_input_ids[:, 0] = self.start_tokens[0]
        dec_input_ids[dec_input_ids == -100] = self.pad_token
        # decoder inputs should start with sot token

        return (
            labels,
            dec_input_ids,
            mels,
        )


def create_optimizer(model, weight_decay, lr, eps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=eps,
    )


# Linear schedule from huggingface transformers

def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# reference of shift_tokens_right from WhisperForConditionalGeneration forward pass
# not used, but good to know what's going on

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def mass_transcribe(filenames, model):
    """
        List of filenames to whisper.transcribe
    """
    with torch.no_grad():
        for filename in filenames:
            transcription = model.transcribe(filename)
            print(f"{os.path.basename(filename)}: {transcription['text']}")
