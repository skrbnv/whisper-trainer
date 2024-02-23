import comet_ml
import torch
import whisper
from tqdm import tqdm
import functions as _fn
from yaml import safe_load
from munch import munchify
import evaluate
from statistics import mean
from random import choice
from string import ascii_lowercase
from dotenv import load_dotenv
import os

with open("config.yaml", "rt") as f:
    config = safe_load(f)
    config = munchify(config)
    RESUME = config.training.resume
    TRAIN_DECODER_ONLY = config.training.decoder_only

initial_epoch = 0
if RESUME:
    train_checkpoint = torch.load(config.training.resume_checkpoint)
    run_id = train_checkpoint["run_id"]
    initial_epoch = train_checkpoint["epoch"] + 1
else:
    run_id = "".join(choice(ascii_lowercase) for _ in range(32))

if config.comet_ml.use is True:
    load_dotenv()
    api_key = os.environ.get("COMET_API_KEY")
    if RESUME:
        experiment = comet_ml.ExistingExperiment(
            api_key=api_key,
            experiment_key=run_id,
        )
    else:
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=config.comet_ml.project_name,
            experiment_key=run_id,
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model(config.whisper.model_name)
optimizer = _fn.create_optimizer(
    model,
    config.optimizer.weight_decay,
    config.optimizer.learning_rate,
    config.optimizer.epsilon,
)
if RESUME:
    model.load_state_dict(train_checkpoint["model_state_dict"])
    optimizer.load_state_dict(train_checkpoint["optimizer_state_dict"])


criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
tokenizer = whisper.tokenizer.get_tokenizer(
    multilingual=model.is_multilingual,
    num_languages=model.num_languages,
    language=config.whisper.target_language,
    task=config.whisper.task,
)
collator = _fn.Collator(tokenizer=tokenizer, usetimestamps=config.whisper.usetimestamps)
train_dataset = _fn.SpeechDataset(config.dataset.train_dir, config.dataset.train_csv)
val_dataset = _fn.SpeechDataset(config.dataset.val_dir, config.dataset.val_csv)
train_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    collate_fn=collator,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.evaluation.batch_size,
    num_workers=config.evaluation.num_workers,
    collate_fn=collator,
)

t_total = (
    (len(train_dataset) // config.training.batch_size)
    // config.training.gradient_accumulation_steps
    * float(config.training.epochs)
)

scheduler = _fn.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.scheduler.warmup_steps,
    num_training_steps=t_total,
)
metrics_wer = evaluate.load("wer")
metrics_cer = evaluate.load("cer")

# setting up checkpoint saving instance
# roll is number of latest checkpoints to keep
checkpointer = _fn.Checkpointer(
    run_id=run_id,
    path=config.checkpoints.training.dir,
    roll=config.checkpoints.training.roll,
)

if TRAIN_DECODER_ONLY:
    # disable training for encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

#print("Transcription state before training")
#_fn.mass_transcribe(
#    ["test.mp3", "test2.mp3", "test3.mp3", "test4.mp3", "test5.mp3"], model
#)

for epoch in range(initial_epoch, config.training.epochs):
    print(f"Epoch {epoch+1}/{config.training.epochs}")
    train_losses, cers, wers, val_losses = [], [], [], []
    # training
    for labels, dec_input_ids, mels in (pb := tqdm(train_dataloader)):
        optimizer.zero_grad()
        audio_features = model.encoder(mels.to(device))
        out = model.decoder(dec_input_ids.to(device), audio_features)
        loss = criterion(out.view(-1, out.size(-1)), labels.to(device).view(-1))
        train_losses.append(loss.item())
        pb.set_description_str(f"CEloss(train): {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        scheduler.step()

    # save checkpoint
    checkpointer(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "run_id": run_id,
        },
        id=epoch,
    )
    # validation
    for labels, dec_input_ids, mels in (pb := tqdm(val_dataloader)):
        with torch.no_grad():
            audio_features = model.encoder(mels.to(device))
            out = model.decoder(dec_input_ids.to(device), audio_features)
            loss = criterion(out.view(-1, out.size(-1)), labels.to(device).view(-1))
            labels[labels == -100] = tokenizer.eot
            out[out == -100] = tokenizer.eot
            texts_pred = _fn.decode(
                torch.argmax(out, dim=-1).tolist(), tokenizer, skip_special_tokens=True
            )
            texts_ref = _fn.decode(labels.tolist(), tokenizer, skip_special_tokens=True)
            cer = metrics_cer.compute(references=texts_ref, predictions=texts_pred)
            wer = metrics_wer.compute(references=texts_ref, predictions=texts_pred)
            pb.set_description_str(
                f"CEloss (val): {loss.item():.4f}, CER: {cer:.4f}, WER: {wer:.4f}"
            )
            cers.append(cer)
            wers.append(wer)
            val_losses.append(loss.item())

    if config.comet_ml.use is True:
        experiment.log_metrics(
            {"Validation loss": mean(val_losses), "CER": mean(cers), "WER": mean(wers)}
        )
    print(
        f"""Total: CEloss(train): {mean(train_losses):.4f}, CEloss(val): {mean(val_losses):.4f},
        CER: {mean(cers):.4f}, WER: {mean(wers):.4f}"""
    )
    #_fn.mass_transcribe(
    #    ["test.mp3", "test2.mp3", "test3.mp3", "test4.mp3", "test5.mp3"], model
    #)
