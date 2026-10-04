import numpy as np
import pretty_midi
from typing import List, Tuple
from pathlib import Path
import logging

from configs.config import Config, DataConfig, VocabConfig

logger = logging.getLogger(__name__)

class CPTokenizer:
    """Tokenizer for converting MIDI to Compound Word (CP) tokens."""
    
    def __init__(self, vocab_config: VocabConfig, data_config: DataConfig):
        self.vocab = vocab_config
        self.data = data_config
        
        self.PAD = self.vocab.pad_id
        self.BAR_NEW = 1
        self.BAR_CONT = 2
        
        # We assume specific EOS token values if not strictly provided as attributes,
        # but using the given ones:
        self.bar_eos = 3
        self.pos_eos = 17
        self.pitch_eos = 89
        self.dur_eos = 65

    def encode(self, midi_path: str) -> np.ndarray:
        """
        Convert a MIDI file to a sequence of CP tokens.
        
        Args:
            midi_path: Path to the MIDI file.
            
        Returns:
            numpy array of shape (N, 4) with tokenized sequence.
        """
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            logger.debug(f"Failed to parse {midi_path}: {e}")
            return np.empty((0, 4), dtype=np.int32)
            
        notes = []
        # Extract piano notes
        for instrument in midi.instruments:
            if not instrument.is_drum and (0 <= instrument.program <= 7):
                notes.extend(instrument.notes)
                
        if not notes:
            return np.empty((0, 4), dtype=np.int32)
            
        # Sort notes by start time, then pitch
        notes.sort(key=lambda x: (x.start, x.pitch))
        
        # Quantize to 16th note grid
        tempo_changes = midi.get_tempo_changes()
        tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
        sec_per_beat = 60.0 / tempo
        sec_per_16th = sec_per_beat / 4.0
        
        tokens = []
        current_bar = -1
        
        for note in notes:
            start_tick = int(round(note.start / sec_per_16th))
            dur_tick = int(round((note.end - note.start) / sec_per_16th))
            
            # Quantize duration (clamp 1-64)
            dur_tick = max(1, min(dur_tick, self.data.max_duration_steps))
            
            bar_idx = start_tick // self.data.resolution
            pos_idx = start_tick % self.data.resolution
            
            if bar_idx > self.data.max_bars_per_sequence:
                continue
                
            pitch = note.pitch
            if pitch < self.data.min_pitch or pitch > self.data.max_pitch:
                continue
                
            pitch_token = pitch - self.data.min_pitch + 1
            pos_token = pos_idx + 1
            dur_token = dur_tick
            
            bar_token = self.BAR_NEW if bar_idx != current_bar else self.BAR_CONT
            current_bar = bar_idx
            
            tokens.append((bar_token, pos_token, pitch_token, dur_token))
            
        if not tokens:
            return np.empty((0, 4), dtype=np.int32)
            
        # Append EOS tokens at the end
        tokens.append((self.bar_eos, self.pos_eos, self.pitch_eos, self.dur_eos))
        
        return np.array(tokens, dtype=np.int32)

    def decode(self, tokens: np.ndarray) -> pretty_midi.PrettyMIDI:
        """
        Convert CP tokens back to a MIDI file.
        
        Args:
            tokens: numpy array of shape (N, 4).
            
        Returns:
            pretty_midi.PrettyMIDI object.
        """
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        current_bar = 0
        tempo = 120.0
        sec_per_beat = 60.0 / tempo
        sec_per_16th = sec_per_beat / 4.0
        
        for t in tokens:
            bar, pos, pitch, dur = t
            if bar == self.PAD or bar == self.bar_eos:
                break
                
            if bar == self.BAR_NEW:
                current_bar += 1
                
            # If sequence didn't start with BAR_NEW, assume bar 1
            if current_bar == 0:
                current_bar = 1
                
            bar_offset = (current_bar - 1) * self.data.resolution
            pos_idx = pos - 1
            pitch_val = pitch - 1 + self.data.min_pitch
            dur_val = dur
            
            start_time = (bar_offset + pos_idx) * sec_per_16th
            end_time = start_time + (dur_val * sec_per_16th)
            
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch_val,
                start=start_time,
                end=end_time
            )
            piano.notes.append(note)
            
        midi.instruments.append(piano)
        return midi

    def encode_corpus(self, midi_paths: List[str], max_seq_len: int) -> List[np.ndarray]:
        """
        Process a whole dataset.
        
        Args:
            midi_paths: List of MIDI file paths.
            max_seq_len: Maximum sequence length (truncate if longer).
            
        Returns:
            List of tokenized sequences.
        """
        corpus = []
        for path in midi_paths:
            tokens = self.encode(path)
            if len(tokens) > 0:
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                    # Ensure EOS token at the end after truncation
                    tokens[-1] = (self.bar_eos, self.pos_eos, self.pitch_eos, self.dur_eos)
                corpus.append(tokens)
        return corpus
