"""
Generate Preference Dataset from MAESTRO MIDI files for MIDI-BERT Reward Model.

This script:
1. Tokenizes all MAESTRO MIDI files into CP (Compound Word) representation
   compatible with MIDI-BERT: each token = [Bar, Position, Pitch, Duration]
2. Corrupts each tokenized file using one of several strategies
3. Generates n^2 preference pairs (every original vs every corrupted)
4. Saves as .npy arrays + metadata CSV

Usage:
    python generate_preference_dataset.py \
        --midi_dir maestro-v3.0.0 \
        --dict_file data_creation/prepare_data/dict/CP.pkl \
        --output_dir preference_data \
        --max_seq_len 512 \
        --max_files 0          # 0 = all files
"""

import os
import sys
import glob
import copy
import random
import pickle
import argparse
import csv
from pathlib import Path
from itertools import product
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

import numpy as np
from tqdm import tqdm

# ─── CP Tokenization (vendored from MIDI-BERT) ────────────────────────────────

try:
    import miditoolkit
except ImportError:
    logging.error("miditoolkit not installed. Run: pip install miditoolkit")
    sys.exit(1)

# ── Constants from MIDI-BERT utils.py ──
DEFAULT_VELOCITY_BINS = np.array([0, 32, 48, 64, 80, 96, 128])
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
DEFAULT_RESOLUTION = 480


class Item:
    def __init__(self, name, start, end, velocity, pitch, Type):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.Type = Type


class Event:
    def __init__(self, name, time, value, text, Type):
        self.name = name
        self.time = time
        self.value = value
        self.text = text
        self.Type = Type


def read_items(file_path):
    """Read notes and tempo changes from a MIDI file."""
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    note_items = []
    for i, instr in enumerate(midi_obj.instruments):
        notes = sorted(instr.notes, key=lambda x: (x.start, x.pitch))
        for note in notes:
            note_items.append(Item(
                name='Note', start=note.start, end=note.end,
                velocity=note.velocity, pitch=note.pitch, Type=i
            ))
    note_items.sort(key=lambda x: x.start)

    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo', start=tempo.time, end=None,
            velocity=None, pitch=int(tempo.tempo), Type=-1
        ))
    tempo_items.sort(key=lambda x: x.start)

    if not tempo_items:
        # Default tempo if none specified
        tempo_items = [Item('Tempo', 0, None, None, 120, -1)]

    # Expand tempo to all beats
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item('Tempo', tick, None, None, existing_ticks[tick], -1))
        else:
            if output:
                output.append(Item('Tempo', tick, None, None, output[-1].pitch, -1))
            else:
                output.append(Item('Tempo', tick, None, None, 120, -1))
    tempo_items = output
    return note_items, tempo_items


def quantize_items(items, ticks=120):
    if not items:
        return items
    grids = np.arange(0, items[-1].start + ticks, ticks, dtype=int)
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items


def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = [item for item in items if db1 <= item.start < db2]
        groups.append([db1] + insiders + [db2])
    return groups


def item2event(groups):
    """Convert groups of items to CP events (no task-specific labeling)."""
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        new_bar = True

        for item in groups[i][1:-1]:
            if item.name != 'Note':
                continue
            note_tuple = []

            # Bar
            bar_value = 'New' if new_bar else 'Continue'
            new_bar = False
            note_tuple.append(Event('Bar', None, bar_value, str(n_downbeat), -1))

            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = int(np.argmin(abs(flags - item.start)))
            note_tuple.append(Event(
                'Position', item.start,
                f'{index + 1}/{DEFAULT_FRACTION}',
                str(item.start), -1
            ))

            # Pitch
            note_tuple.append(Event(
                'Pitch', item.start, item.pitch,
                str(item.pitch), -1
            ))

            # Duration
            duration = item.end - item.start
            dur_index = int(np.argmin(abs(DEFAULT_DURATION_BINS - duration)))
            note_tuple.append(Event(
                'Duration', item.start, dur_index,
                f'{duration}/{DEFAULT_DURATION_BINS[dur_index]}', -1
            ))

            events.append(note_tuple)
    return events


class CPTokenizer:
    """Tokenize MIDI files into CP (Compound Word) representation."""

    def __init__(self, dict_path=None, e2w=None, w2e=None):
        if dict_path is not None:
            with open(dict_path, 'rb') as f:
                self.e2w, self.w2e = pickle.load(f)
        elif e2w is not None and w2e is not None:
            self.e2w = e2w
            self.w2e = w2e
        else:
            # Build default dictionary matching MIDI-BERT's CP.pkl structure
            self.e2w, self.w2e = self._build_default_dict()

        self.pad_word = [self.e2w[etype][f'{etype} <PAD>']
                         for etype in self.e2w]
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']

    def _build_default_dict(self):
        """Build the default CP dictionary matching MIDI-BERT."""
        e2w = {}
        w2e = {}

        # Bar: New, Continue, <PAD>, <MASK>
        e2w['Bar'] = {'Bar New': 0, 'Bar Continue': 1, 'Bar <PAD>': 2, 'Bar <MASK>': 3}

        # Position: 1/16 through 16/16, <PAD>, <MASK>
        e2w['Position'] = {}
        for i in range(DEFAULT_FRACTION):
            e2w['Position'][f'Position {i + 1}/{DEFAULT_FRACTION}'] = i
        e2w['Position']['Position <PAD>'] = DEFAULT_FRACTION
        e2w['Position']['Position <MASK>'] = DEFAULT_FRACTION + 1

        # Pitch: 0-127, <PAD>, <MASK>  (but MIDI-BERT uses 22-107 → 88 pitches)
        # Checkpoint expects exactly 88 tokens for Pitch.
        # We'll use 0-85 for active pitches (22-107), 86 for PAD, 87 for MASK.
        e2w['Pitch'] = {}
        for i in range(22, 108):  # 86 piano keys
            e2w['Pitch'][f'Pitch {i}'] = i - 22
        e2w['Pitch']['Pitch <PAD>'] = 86
        e2w['Pitch']['Pitch <MASK>'] = 87

        # Duration: 0-63, <PAD>, <MASK>
        e2w['Duration'] = {}
        for i in range(len(DEFAULT_DURATION_BINS)):
            e2w['Duration'][f'Duration {i}'] = i
        e2w['Duration']['Duration <PAD>'] = len(DEFAULT_DURATION_BINS)
        e2w['Duration']['Duration <MASK>'] = len(DEFAULT_DURATION_BINS) + 1

        # Build reverse mapping
        for etype in e2w:
            w2e[etype] = {v: k for k, v in e2w[etype].items()}

        return e2w, w2e

    def extract_events(self, midi_path):
        """Extract CP events from a MIDI file."""
        try:
            note_items, tempo_items = read_items(midi_path)
            if len(note_items) == 0:
                return None
            note_items = quantize_items(note_items)
            max_time = note_items[-1].end
            items = tempo_items + note_items
            groups = group_items(items, max_time)
            events = item2event(groups)
            return events
        except Exception as e:
            logging.warning(f"Failed to process {midi_path}: {e}")
            return None

    def events_to_words(self, events):
        """Convert events to integer word indices."""
        words = []
        for note_tuple in events:
            nts = []
            for e in note_tuple:
                e_text = f'{e.name} {e.value}'
                if e_text in self.e2w[e.name]:
                    nts.append(self.e2w[e.name][e_text])
                else:
                    # Handle out-of-vocabulary (e.g., pitch outside piano range)
                    # Map to closest valid token
                    if e.name == 'Pitch':
                        pitch = max(22, min(107, e.value))
                        nts.append(self.e2w['Pitch'][f'Pitch {pitch}'])
                    else:
                        nts.append(0)  # fallback
            if len(nts) == 4:
                words.append(nts)
        return words

    def tokenize_file(self, midi_path, max_seq_len=512):
        """
        Tokenize a MIDI file into CP chunks.

        Returns:
            list of np.array, each shape (max_seq_len, 4)
        """
        events = self.extract_events(midi_path)
        if events is None or len(events) == 0:
            return []

        words = self.events_to_words(events)
        if len(words) == 0:
            return []

        # Slice into chunks of max_seq_len
        chunks = []
        for i in range(0, len(words), max_seq_len):
            chunk = words[i:i + max_seq_len]
            # Pad if necessary
            while len(chunk) < max_seq_len:
                chunk.append(self.pad_word)
            chunks.append(np.array(chunk, dtype=np.int64))

        return chunks


# ─── Corruption Strategies ─────────────────────────────────────────────────────

def corrupt_shuffle_bars(tokens: np.ndarray, e2w: dict) -> np.ndarray:
    """Shuffle bar-level blocks. Splits at 'Bar New' tokens and shuffles."""
    tokens = tokens.copy()
    # Bar dimension is index 0; 'Bar New' = 0 typically
    bar_new_id = e2w['Bar'].get('Bar New', 0)

    # Find bar boundaries
    bar_positions = [i for i in range(len(tokens)) if tokens[i, 0] == bar_new_id]

    if len(bar_positions) < 2:
        return tokens  # Can't shuffle with < 2 bars

    # Build bar blocks
    bars = []
    for idx in range(len(bar_positions)):
        start = bar_positions[idx]
        end = bar_positions[idx + 1] if idx + 1 < len(bar_positions) else len(tokens)
        # Skip padding
        if np.all(tokens[start] == tokens[-1]):  # likely pad
            break
        bars.append(tokens[start:end].copy())

    if len(bars) < 2:
        return tokens

    random.shuffle(bars)
    result = np.concatenate(bars, axis=0)

    # Pad/truncate to original length
    if len(result) < len(tokens):
        pad = np.tile(tokens[-1], (len(tokens) - len(result), 1))
        result = np.concatenate([result, pad], axis=0)
    elif len(result) > len(tokens):
        result = result[:len(tokens)]

    return result


def corrupt_randomize_pitch(tokens: np.ndarray, e2w: dict) -> np.ndarray:
    """Randomize the Pitch dimension (index 2) for all non-padding tokens."""
    tokens = tokens.copy()
    pad_pitch = e2w['Pitch'].get('Pitch <PAD>', 86)
    n_pitches = pad_pitch  # valid pitch range is [0, pad_pitch)

    for i in range(len(tokens)):
        if tokens[i, 2] != pad_pitch:  # not padding
            tokens[i, 2] = random.randint(0, n_pitches - 1)
    return tokens


def corrupt_randomize_position(tokens: np.ndarray, e2w: dict) -> np.ndarray:
    """Randomize the Position dimension (index 1) for non-padding tokens."""
    tokens = tokens.copy()
    pad_pos = e2w['Position'].get('Position <PAD>', DEFAULT_FRACTION)
    n_positions = pad_pos  # valid range [0, pad_pos)

    for i in range(len(tokens)):
        if tokens[i, 1] != pad_pos:
            tokens[i, 1] = random.randint(0, n_positions - 1)
    return tokens


def corrupt_note_dropout(tokens: np.ndarray, e2w: dict, drop_rate=0.4) -> np.ndarray:
    """Drop ~40% of note tokens and re-pack with padding."""
    pad_word = np.array([
        e2w['Bar'].get('Bar <PAD>', 2),
        e2w['Position'].get('Position <PAD>', DEFAULT_FRACTION),
        e2w['Pitch'].get('Pitch <PAD>', 86),
        e2w['Duration'].get('Duration <PAD>', len(DEFAULT_DURATION_BINS)),
    ])

    kept = []
    for i in range(len(tokens)):
        if np.array_equal(tokens[i], pad_word):
            break  # reached padding
        if random.random() > drop_rate:
            kept.append(tokens[i].copy())

    # Re-pad to original length
    result = np.array(kept) if kept else np.empty((0, 4), dtype=np.int64)
    if len(result) < len(tokens):
        pad = np.tile(pad_word, (len(tokens) - len(result), 1))
        result = np.concatenate([result, pad], axis=0) if len(result) > 0 else pad

    return result[:len(tokens)]


def corrupt_randomize_duration(tokens: np.ndarray, e2w: dict) -> np.ndarray:
    """Randomize the Duration dimension (index 3) for non-padding tokens."""
    tokens = tokens.copy()
    pad_dur = e2w['Duration'].get('Duration <PAD>', len(DEFAULT_DURATION_BINS))
    n_durations = pad_dur  # valid range [0, pad_dur)

    for i in range(len(tokens)):
        if tokens[i, 3] != pad_dur:
            tokens[i, 3] = random.randint(0, n_durations - 1)
    return tokens


CORRUPTION_STRATEGIES = {
    'shuffle_bars': corrupt_shuffle_bars,
    'randomize_pitch': corrupt_randomize_pitch,
    'randomize_position': corrupt_randomize_position,
    'note_dropout': corrupt_note_dropout,
    'randomize_duration': corrupt_randomize_duration,
}


def corrupt_tokens(tokens: np.ndarray, e2w: dict, strategy: str = None) -> tuple:
    """
    Apply a corruption strategy to a CP token array.

    Args:
        tokens: shape (seq_len, 4) — CP tokens
        e2w: event-to-word dictionary
        strategy: specific strategy name, or None for random

    Returns:
        (corrupted_tokens, strategy_name)
    """
    if strategy is None:
        strategy = random.choice(list(CORRUPTION_STRATEGIES.keys()))

    corrupt_fn = CORRUPTION_STRATEGIES[strategy]
    corrupted = corrupt_fn(tokens, e2w)
    return corrupted, strategy


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def find_midi_files(midi_dir: str) -> list:
    """Find all .mid and .midi files recursively."""
    patterns = ['**/*.mid', '**/*.midi', '**/*.MID', '**/*.MIDI']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(midi_dir, pattern), recursive=True))
    return sorted(set(files))


def generate_dataset(args):
    """Main dataset generation pipeline."""
    logging.info("=" * 60)
    logging.info("MIDI-BERT Preference Dataset Generator Started")
    logging.info("=" * 60)

    # ── 1. Initialize tokenizer ──
    logging.info("[1/4] Initializing CP tokenizer...")
    if args.dict_file and os.path.exists(args.dict_file):
        logging.info(f"Loading dictionary from: {args.dict_file}")
        tokenizer = CPTokenizer(dict_path=args.dict_file)
    elif args.ckpt_file and os.path.exists(args.ckpt_file):
        logging.info(f"Extracting dictionary from checkpoint: {args.ckpt_file}")
        import torch
        ckpt = torch.load(args.ckpt_file, map_location='cpu')
        if 'e2w' in ckpt and 'w2e' in ckpt:
            tokenizer = CPTokenizer(e2w=ckpt['e2w'], w2e=ckpt['w2e'])
        else:
            logging.warning("Checkpoint does not contain e2w/w2e. Using default dict.")
            tokenizer = CPTokenizer()
    else:
        logging.info("Using default built-in CP dictionary")
        tokenizer = CPTokenizer()

    e2w = tokenizer.e2w
    logging.info(f"Dictionary sizes: Bar={len(e2w['Bar'])}, Position={len(e2w['Position'])}, "
                 f"Pitch={len(e2w['Pitch'])}, Duration={len(e2w['Duration'])}")

    # ── 2. Find and tokenize MIDI files ──
    logging.info(f"[2/4] Finding MIDI files in: {args.midi_dir}")
    midi_files = find_midi_files(args.midi_dir)
    if args.max_files > 0:
        midi_files = midi_files[:args.max_files]
    logging.info(f"Found {len(midi_files)} MIDI files")

    if len(midi_files) == 0:
        logging.error("No MIDI files found!")
        sys.exit(1)

    # Tokenize all files — keep first chunk from each file
    logging.info(f"Tokenizing (max_seq_len={args.max_seq_len})...")
    originals = []       # list of np.array (max_seq_len, 4)
    file_names = []      # corresponding file paths

    for i, midi_path in enumerate(tqdm(midi_files, desc="Tokenizing")):
        if i % 100 == 0:
            logging.info(f"Tokenizing file {i}/{len(midi_files)}: {midi_path}")
        
        chunks = tokenizer.tokenize_file(midi_path, max_seq_len=args.max_seq_len)
        if chunks:
            # Use the first chunk as representative of this file
            if args.all_chunks:
                for ci, chunk in enumerate(chunks):
                    originals.append(chunk)
                    file_names.append(f"{midi_path}::chunk{ci}")
            else:
                originals.append(chunks[0])
                file_names.append(midi_path)
        else:
            logging.warning(f"No tokens generated for {midi_path}")

    n = len(originals)
    logging.info(f"Successfully tokenized: {n} segments from {len(midi_files)} files")

    if n == 0:
        logging.error("No files could be tokenized!")
        sys.exit(1)

    # ── 3. Generate corruptions ──
    logging.info(f"[3/4] Generating {args.num_corruptions} corrupted versions per original...")
    final_originals = []
    final_corrupted = []
    corruption_types = []
    final_file_names = []

    for i in tqdm(range(n), desc="Corrupting"):
        for _ in range(args.num_corruptions):
            corrupted, strategy = corrupt_tokens(originals[i], e2w)
            final_originals.append(originals[i])
            final_corrupted.append(corrupted)
            corruption_types.append(strategy)
            final_file_names.append(file_names[i])

    total_pairs = len(final_originals)

    # ── 4. Save dataset components ──
    logging.info(f"[4/4] Saving to disk: {total_pairs} paired originals and corrupted variants...")
    os.makedirs(args.output_dir, exist_ok=True)

    metadata_rows = []
    for i in range(total_pairs):
        metadata_rows.append({
            'source_idx': i // args.num_corruptions,
            'file_name': final_file_names[i],
            'corruption_type': corruption_types[i],
        })

    # Save arrays
    originals_path = os.path.join(args.output_dir, 'originals.npy')
    corrupted_path = os.path.join(args.output_dir, 'corrupted.npy')
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')

    np.save(originals_path, np.array(final_originals))
    np.save(corrupted_path, np.array(final_corrupted))

    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Save tokenizer dict for the finetuning script
    dict_path = os.path.join(args.output_dir, 'CP_dict.pkl')
    with open(dict_path, 'wb') as f:
        pickle.dump((tokenizer.e2w, tokenizer.w2e), f)

    logging.info("=" * 60)
    logging.info("Dataset generation complete!")
    logging.info("=" * 60)
    logging.info(f"Total files tokenized:  {n}")
    logging.info(f"Pairs represented:      {total_pairs}")
    logging.info("Corruption breakdown:")
    from collections import Counter
    for strat, count in Counter(corruption_types).items():
        logging.info(f"  {strat}: {count}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("  originals.npy          - all original tokenized segments")
    logging.info("  corrupted.npy          - all corrupted tokenized segments")
    logging.info("  metadata.csv           - metadata & labels")
    logging.info("  CP_dict.pkl            - CP dictionary for finetuning")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate preference dataset from MAESTRO MIDI for MIDI-BERT'
    )
    parser.add_argument('--midi_dir', type=str, default='maestro-v3.0.0',
                        help='Path to MAESTRO dataset directory')
    parser.add_argument('--dict_file', type=str, default='',
                        help='Path to CP.pkl dictionary (from MIDI-BERT repo)')
    parser.add_argument('--ckpt_file', type=str, default='pretrain_model.ckpt',
                        help='Path to MIDI-BERT pretrain checkpoint (to extract dict)')
    parser.add_argument('--output_dir', type=str, default='preference_data',
                        help='Output directory for dataset')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length per chunk')
    parser.add_argument('--max_files', type=int, default=0,
                        help='Max MIDI files to process (0 = all)')
    parser.add_argument('--num_corruptions', type=int, default=5,
                        help='Number of corrupted versions to generate per original chunk')
    parser.add_argument('--all_chunks', action='store_true',
                        help='Use all chunks per file (default: first chunk only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    generate_dataset(args)
