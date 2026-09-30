import argparse
import hashlib
from functools import reduce
from operator import or_
from pathlib import Path

import slider

from osunator.config import load_norm_stats, ROOT
from osrparse.utils import Mod

MODEL_PATH = ROOT / "best_model.keras"
MOD_MAP = {'NM': Mod.NoMod, 'NF': Mod.NoFail, 'SO': Mod.SpunOut, 'HD': Mod.Hidden}

def process_replay(filename, mods, output, temperature):
    # Tensorflow is slow to import, so we do it here
    from osunator.generate import generate_replay, result_to_replay
    from tensorflow import keras

    stats = load_norm_stats()
    beatmap = slider.beatmap.Beatmap.from_path(filename)
    with open(filename, 'rb') as f:
        beatmap_hash = hashlib.md5(f.read()).hexdigest()
    print(f"generating replay for {filename}...")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    prediction = generate_replay(model=model, beatmap=beatmap, stats=stats, temperature=temperature)

    replay = result_to_replay(result=prediction, beatmap_hash=beatmap_hash, mods=mods)
    replay.write_path(output / f'{filename.stem}.osr')
    print(f"wrote {output / f'{filename.stem}.osr'}")

def valid_file(path_str):
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"The file {path} does not exist")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"The file {path} is not a file")
    if not path.suffix == '.osu':
        raise argparse.ArgumentTypeError(f"The file {path} is not a .osu file")
    return path

def valid_dir(path_str):
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"The directory {path} does not exist")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"The directory {path} is not a directory")
    return path

def valid_temp(temperature):
    t = float(temperature)
    if t < 0:
        raise argparse.ArgumentTypeError(f"Temperature must be positive, got {t}")
    return t

def main():
    parser = argparse.ArgumentParser(
        prog='osunator',
        description='osu! AI replay generator'
    )

    parser.add_argument('filename', type=valid_file, help='Osu! beatmap file (.osu)')
    parser.add_argument('--mods', default=['SO'], type=str, choices=MOD_MAP.keys(), nargs='+', help='Mods to apply to the replay, seperated by a space')
    parser.add_argument('--output', '-o', type=valid_dir, help='Output path', default='./')
    parser.add_argument('--temperature', '-t', type=valid_temp, default=0.0, help='Amount of noise to add to the cursor (recommended [0-0.3])')

    args = parser.parse_args()
    mods = reduce(or_, (MOD_MAP[mod] for mod in args.mods), Mod.NoMod)

    process_replay(args.filename, mods, args.output, args.temperature)

    #try:
    #    process_replay(args.filename, mod, args.output, args.temperature)
    #except Exception as e:
    #    print(f"error: {e}", file=sys.stderr)
    #    sys.exit(1)


if __name__ == '__main__':
    main()
