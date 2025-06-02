import torch
import random


def time_mask(spec, max_width=20):
    t = spec.shape[-1]
    width = random.randint(0, max_width)
    start = random.randint(0, t - width) if t - width > 0 else 0
    spec[..., start:start+width] = 0
    return spec


def freq_mask(spec, max_width=10):
    f = spec.shape[-2]
    width = random.randint(0, max_width)
    start = random.randint(0, f - width) if f - width > 0 else 0
    spec[..., start:start+width, :] = 0
    return spec


def time_shift(spec, max_shift=10):
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(spec, shifts=shift, dims=-1)


def spec_augment(mix, vocals, instruments):
    for fn in [time_mask, freq_mask, time_shift]:
        if random.random() < 0.5:
            mix = fn(mix.clone())
            vocals = fn(vocals.clone())
            instruments = fn(instruments.clone())
    return mix, vocals, instruments
