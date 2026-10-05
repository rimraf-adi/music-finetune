import torch
import numpy as np
import os
from pathlib import Path

from configs.config import Config
from models.cp_transformer import CPTransformer
from models.reference_encoder import ReferenceEncoder
from data.tokenizer import CPTokenizer
from features.mft import MFTExtractor

def main():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Running generation on device: {device}")
    
    # 1. Initialize Tokenizer
    tokenizer = CPTokenizer(config.vocab, config.data)
    
    # 2. Load trained Policy Model
    policy = CPTransformer(config).to(device)
    ckpt_path = "checkpoints/grpo/step_0002950/model.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return
        
    print(f"Loading trained policy from {ckpt_path}...")
    policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    policy.eval()
    
    # 3. Create a Controlled Reference Trajectory (8 bars)
    # y_ref shape: (batch, num_bars, 4) -> [note_density, pitch_centroid, tonal_tension, rhythmic_complexity]
    num_bars = 8
    y_ref = torch.zeros((1, num_bars, 4), device=device)
    
    # Scenario: Crescendo & rising pitch contour over 8 bars
    for b in range(num_bars):
        y_ref[0, b, 0] = 3.0 + b * 2.0         # Note density: from 3 notes/bar to 17 notes/bar
        y_ref[0, b, 1] = 40.0 + b * 6.0        # Pitch centroid: ascending from E2 (40) to G#5 (82)
        y_ref[0, b, 2] = 0.1 + b * 0.1         # Tonal tension: rising
        y_ref[0, b, 3] = 0.2 + b * 0.08        # Rhythmic complexity: increasing
        
    print(f"Generated target trajectory across {num_bars} bars:")
    print(f"  Note density range: {y_ref[0, 0, 0].item():.1f} -> {y_ref[0, -1, 0].item():.1f} notes/bar")
    print(f"  Pitch centroid range: {y_ref[0, 0, 1].item():.1f} -> {y_ref[0, -1, 1].item():.1f} MIDI key")
    
    # 4. Encode Reference Trajectory
    ref_encoder = ReferenceEncoder(config).to(device)
    encoded_ref = ref_encoder(y_ref) # (1, num_bars, d_model)
    
    # 5. Initialize Prompt (Bar 1 start token)
    prompt_bar = torch.tensor([[1]], device=device)     # NEW_BAR = 1
    prompt_pos = torch.tensor([[1]], device=device)     # Position 1
    prompt_pitch = torch.tensor([[40]], device=device)   # Pitch ~ 60 (Middle C)
    prompt_dur = torch.tensor([[4]], device=device)     # Quarter note
    
    print("\nGenerating conditioned music tokens with CPTransformer...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            completions = policy.generate(
                prompt_bar=prompt_bar,
                prompt_pos=prompt_pos,
                prompt_pitch=prompt_pitch,
                prompt_dur=prompt_dur,
                max_new_tokens=128,
                temperature=0.9,
                top_p=0.92,
                ref_embeddings=encoded_ref
            )
            
    # Combine outputs into (seq_len, 4)
    tokens_np = np.stack([
        completions['bar'][0].cpu().numpy(),
        completions['position'][0].cpu().numpy(),
        completions['pitch'][0].cpu().numpy(),
        completions['duration'][0].cpu().numpy()
    ], axis=-1)
    
    print(f"Generated sequence of {len(tokens_np)} CP tokens.")
    
    # 6. Extract MFT to verify tracking
    mft_extractor = MFTExtractor()
    feats = mft_extractor.extract(tokens_np[:, 0], tokens_np[:, 1], tokens_np[:, 2], tokens_np[:, 3])
    print(f"\nExtracted MFT over {feats.shape[0]} generated bars:")
    for b_idx in range(min(len(feats), num_bars)):
        print(f"  Bar {b_idx + 1}: ND={feats[b_idx, 0]:.1f} (target {y_ref[0, b_idx, 0]:.1f}) | PC={feats[b_idx, 1]:.1f} (target {y_ref[0, b_idx, 1]:.1f})")
        
    # 7. Decode to MIDI
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = output_dir / "generated_reference_tracking.mid"
    
    midi_obj = tokenizer.decode(tokens_np)
    midi_obj.write(str(midi_path))
    print(f"\nSuccessfully wrote MIDI file to: {midi_path.resolve()}")

if __name__ == "__main__":
    main()
