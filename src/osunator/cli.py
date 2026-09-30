import hashlib
from argparse import ArgumentParser, ArgumentTypeError
from functools import reduce
from operator import or_
from pathlib import Path

from slider.beatmap import Beatmap

from osunator.config import load_norm_stats, ROOT
from osrparse.utils import Mod

MODEL_PATH = ROOT / "best_model.keras"
MOD_MAP = {
    "NM": Mod.NoMod,
    "NF": Mod.NoFail,
    "HD": Mod.Hidden,
    "SD": Mod.SuddenDeath,
    "DT": Mod.DoubleTime,
    "RX": Mod.Relax,
    "HT": Mod.HalfTime,
    "NC": Mod.Nightcore,
    "FL": Mod.Flashlight,
    "SO": Mod.SpunOut,
    "PF": Mod.Perfect,
}

def process_replay(filename: str, mods: Mod, username: str, output: Path, temperature: float) -> None:
    # Tensorflow is slow to import, so we do it here
    from osunator.generate import generate_replay, result_to_replay
    from tensorflow import keras

    stats = load_norm_stats()
    beatmap = Beatmap.from_path(filename)

    with open(filename, "rb") as file:
        beatmap_hash = hashlib.md5(file.read()).hexdigest()

    print(f"generating replay for {filename}...")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    prediction = generate_replay(model, beatmap, stats, temperature)
    replay = result_to_replay(prediction, beatmap_hash, username, mods)

    replay.write_path(output / f"{filename.stem}.osr")
    print(f"wrote {output / f"{filename.stem}.osr"}")

def valid_file(arg: str) -> Path:
    path = Path(arg)

    if not path.exists():
        raise ArgumentTypeError(f"The file {path} does not exist")
    elif not path.is_file():
        raise ArgumentTypeError(f"The file {path} is not a file")
    elif not path.suffix == ".osu":
        raise ArgumentTypeError(f"The file {path} is not a .osu file")

    return path

def valid_dir(arg: str) -> Path:
    path = Path(arg)

    if not path.exists():
        raise ArgumentTypeError(f"The directory {path} does not exist")
    elif not path.is_dir():
        raise ArgumentTypeError(f"The directory {path} is not a directory")

    return path

def valid_temp(arg: str) -> float:
    temperature = float(arg)

    if temperature < 0:
        raise ArgumentTypeError(f"Temperature must be positive, got {temperature}")

    return temperature

def main() -> None:
    parser = ArgumentParser(prog="osunator", description="osu! AI deteministic replay generator")

    parser.add_argument("filename", type=valid_file, help="Path to the .osu beatmap file.")
    parser.add_argument("--mods", default=["SO"], type=str, choices=MOD_MAP.keys(), nargs="+", help="Mods to apply to the replay, seperated by a space.")
    parser.add_argument("--username", "-u", type=str, default="Osunator", help="Username of the player in the replay.")
    parser.add_argument("--output", "-o", type=valid_dir, help="Output directory for generated replays. Defaults to the current directory.", default="./")
    parser.add_argument("--temperature", "-t", type=valid_temp, default=0.0, help="Amount of noise to add to the cursor. Recommended range is 0.0-0.3.")

    args = parser.parse_args()
    mods = reduce(or_, (MOD_MAP[mod] for mod in args.mods), Mod.NoMod)

    process_replay(args.filename, mods, args.username, args.output, args.temperature)

if __name__ == "__main__":
    main()
