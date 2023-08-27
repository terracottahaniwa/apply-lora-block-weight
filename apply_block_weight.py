import os
import re
import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from networks import convert_diffusers_name_to_compvis
from train_util import precalculate_safetensors_hashes as safetensors_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to input safetensors")
    parser.add_argument("output", help="path to output safetensors")
    parser.add_argument("ratios", help="the LBW numeric array")
    args = parser.parse_args()

    LOAD_PATH = args.input
    SAVE_PATH = args.output
    RATIOS = [float(x) for x in args.ratios.split(",")]

    BLOCKID17 = [
        "BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "M00",
        "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]
    BLOCKID26 = [
        "BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "IN09", "IN10", "IN11", "M00",
        "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]

    LAYERS = len(RATIOS)
    assert LAYERS in [17, 26]

    if LAYERS == 17:
        RATIO_OF_ = dict(zip(BLOCKID17, RATIOS))
    if LAYERS == 26:
        RATIO_OF_ = dict(zip(BLOCKID26, RATIOS))
    print(RATIO_OF_)

    def compvis_name_to_blockid(compvis_name):
        PATTERNS = [
            r"^transformer_text_model_(encoder)_layers_(\d+)_.*",
            r"^diffusion_model_(in)put_blocks_(\d+)_.*",
            r"^diffusion_model_(middle)_block_(\d+)_.*",
            r"^diffusion_model_(out)put_blocks_(\d+)_.*"]

        def replacement(match):
            g1 = str(match.group(1))  # encoder, in, middle, out
            g2 = int(match.group(2))  # number
            assert g1 in ["encoder", "in", "middle", "out"]
            assert isinstance(g2, int)

            if g1 == "encoder":
                return "BASE"
            if g1 == "middle":
                return "M00"
            return f"{str.upper(g1)}{g2:02}"

        strings = compvis_name
        for pattern in PATTERNS:
            strings = re.sub(pattern, replacement, strings)
        blockid = strings

        if LAYERS == 17:
            assert blockid in BLOCKID17
        if LAYERS == 26:
            assert blockid in BLOCKID26
        return blockid

    with safe_open(LOAD_PATH, framework="pt", device="cpu") as f:
        tensors = {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
            compvis_name = convert_diffusers_name_to_compvis(key, is_sd2=False)
            blockid = compvis_name_to_blockid(compvis_name)
            if compvis_name.endswith("lora_up.weight"):
                tensors[key] *= RATIO_OF_[blockid]
                print(f"({blockid}) {compvis_name}"
                      f"updated with factor {RATIO_OF_[blockid]}")

        metadata = f.metadata()
        model_hash, legacy_hash = safetensors_hashes(tensors, metadata)
        output_name = os.path.splitext(os.path.basename(SAVE_PATH))[0]
        metadata['ss_output_name'] = output_name
        metadata['sshs_model_hash'] = model_hash
        metadata['sshs_legacy_hash'] = legacy_hash

        save_file(tensors, SAVE_PATH, metadata)


if __name__ == "__main__":
    main()
