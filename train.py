import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np
import wandb
from pathlib import Path
from datetime import datetime

wandb.init(project=f"ddsp-pytorch", entity='auditory-grounding')


class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000


# Config related
args.parse_args()
with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

PATH_CHECKPOINTS = path.join(args.ROOT, args.NAME, "checkpoints")
Path(PATH_CHECKPOINTS).mkdir(parents=True, exist_ok=True)

# Model related
model = DDSP(**config["model"]).to(device)
if config["train"]["load_checkpoint"] and path.exists(path.join(PATH_CHECKPOINTS, "state-best.pth")):
    print(f"Loading the model from: {path.join(PATH_CHECKPOINTS, 'state-best.pth')}")
    model.load_state_dict(torch.load(path.join(PATH_CHECKPOINTS, "state-best.pth")))

wandb.watch(model, log='all', log_freq=1000, log_graph=True)

# Data related
dataset = Dataset(config["preprocess"]["out_dir"])
dataloader = torch.utils.data.DataLoader(
    dataset,
    args.BATCH,
    True,
    drop_last=True,
)
print(f"Total steps: {args.STEPS}")
print(f"len(dataloader): {len(dataloader)}")

mean_loudness, std_loudness = mean_std_loudness(dataloader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))
print(f"Epochs: {epochs}")

for e in tqdm(range(epochs)):
    start = datetime.now()
    for s, p, l in dataloader:
        s = s.to(device)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)

        l = (l - mean_loudness) / std_loudness

        y, harmonic, noise, harmonic_plus_noise = model(p, l)
        y = y.squeeze(-1)
        harmonic = harmonic.squeeze(-1)
        noise = noise.squeeze(-1)
        harmonic_plus_noise = harmonic_plus_noise.squeeze(-1)

        ori_stft = multiscale_fft(
            s,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)
        wandb.log({"loss": loss.item(),
                   "step": step
                   })
        step += 1

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
    wandb.log({"seconds_per_epoch": (datetime.now() - start).total_seconds()})

    if (e+1 == epochs) or (not e % config["train"]["eval_per_n_epochs"]):
        writer.add_scalar("lr", schedule(e), e)
        writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
        writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
        # scheduler.step()
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(PATH_CHECKPOINTS, f"state-{e}.pth"),
            )
            torch.save(
                model.state_dict(),
                path.join(PATH_CHECKPOINTS, f"state-best.pth"),
            )

            artifact = wandb.Artifact(f"{args.NAME}-{e}", type='model')
            artifact.add_file(path.join(PATH_CHECKPOINTS, f"state-{e}.pth"))
            artifact.add_file(path.join(PATH_CHECKPOINTS, f"state-best.pth"))
            wandb.log_artifact(artifact)

        mean_loss = 0
        n_element = 0

        for b in range(3):
            audio = torch.cat([y[b, :], s[b, :], harmonic[b, :], noise[b, :], harmonic_plus_noise[b, :]], -1).reshape(-1).detach().cpu().numpy()
            audio_path = path.join(args.ROOT, args.NAME, f"eval_{e:06d}-{b:02d}.wav")
            sf.write(
                audio_path,
                audio,
                config["preprocess"]["sampling_rate"],
            )
            wandb.log(
                {f"eval_{e:06d}-{b:02d}": wandb.Audio(audio, caption=f"Resynth - original - harmonic - noise - harmonic_plus_noise", sample_rate=config["preprocess"]["sampling_rate"])})
