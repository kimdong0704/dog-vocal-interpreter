import numpy as np

import torch
 
from woofdecoder.branches.audio import load_audio_encoder
 
encoder = load_audio_encoder("beats_iter3_plus_as2m")

print("1. device, layers, dim:", encoder.device, encoder.num_layers, encoder.dim)
 
# 2. The hooks see the same last layer that BEATs itself returns

audio = torch.randn(1, 16000, device=encoder.device) * 0.1

mask = torch.zeros(1, 16000, dtype=torch.bool, device=encoder.device)

features, _ = encoder.layer_features(audio, mask)

with torch.no_grad():

    final, _ = encoder.model.extract_features(audio, padding_mask=mask)

print("2. last layer matches BEATs output:", torch.allclose(features[-1], final, atol=1e-4))
 
# 3. Short, normal and long audio all give (12, 768)

rng = np.random.default_rng(0)

for seconds in [0.2, 1.0, 10.0, 25.0]:

    wave = (rng.standard_normal(int(seconds * 16000)) * 0.1).astype(np.float32)

    embedding = encoder.embed_waveforms([wave])

    print(f"3. {seconds:>5} s ->", embedding.shape, "finite:", np.isfinite(embedding).all())
 
# 4. Batched vs one at a time (training batches, the app embeds one segment at a time)

waves = [(rng.standard_normal(n) * 0.1).astype(np.float32) for n in (4000, 16000, 40000)]

together = encoder.embed_waveforms(waves)

alone = np.stack([encoder.embed_waveforms([wave])[0] for wave in waves])

difference = np.abs(together - alone).max() / np.abs(alone).max()

print(f"4. batched vs alone, largest relative difference: {difference:.10f}")
 
from woofdecoder.branches.audio import encoder_from_config

from woofdecoder.config import load_config
 
encoder = encoder_from_config(load_config())

print("5. from config:", encoder.name, encoder.settings)
 