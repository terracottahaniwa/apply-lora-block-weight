import os
import re
import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from networks import convert_diffusers_name_to_compvis
from train_util import precalculate_safetensors_hashes


# argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="path to input safetensors")
parser.add_argument("output", help="path to output safetensors")
parser.add_argument("ratios", help = "the lbw numeric array")
args = parser.parse_args()


def main():
    BLOCKID17 = [
        "BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00",
        "OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
    BLOCKID26 = [
        "BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00",
        "OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]

    def compvis_name_to_blockid(compvis_name):
        patterns = [
            r"^transformer_text_model_(encoder)_layers_(\d+)_.*",
            r"^diffusion_model_(in)put_blocks_(\d+)_.*",
            r"^diffusion_model_(middle)_block_(\d+)_.*",
            r"^diffusion_model_(out)put_blocks_(\d+)_.*"]
        
        def replacement(match):
            g1 = str(match.group(1)) # encoder, in, middle, out
            g2 = int(match.group(2)) # number
            assert g1 in ["encoder", "in", "middle", "out"]
            assert isinstance(g2, int)
            if g1 == "encoder":
                return "BASE"
            if g1 == "middle":
                return "M00"
            return f"{str.upper(g1)}{g2:02}"
            
        strings = compvis_name
        for pattern in patterns:
            strings = re.sub(pattern, replacement, strings)
        blockid = strings
        return blockid

    load_path = args.input
    save_path = args.output
    ratios = [float(x) for x in args.ratios.split(",")]

    LAYERS = len(ratios)
    assert LAYERS in [17, 26]
    if LAYERS == 17: ratio_of_ = dict(zip(BLOCKID17, ratios))
    if LAYERS == 26: ratio_of_ = dict(zip(BLOCKID26, ratios))
    print(ratio_of_)
    tensors = {}
    with safe_open(load_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
            compvis_name = convert_diffusers_name_to_compvis(key, is_sd2=False)
            blockid = compvis_name_to_blockid(compvis_name)
            if LAYERS == 17: assert blockid in BLOCKID17
            if LAYERS == 26: assert blockid in BLOCKID26
            if compvis_name.endswith("lora_up.weight"):
                tensors[key] *= ratio_of_[blockid]
                print(f"({blockid}) {compvis_name} updated with factor {ratio_of_[blockid]}")
        metadata = f.metadata()
        model_hash, legacy_hash = precalculate_safetensors_hashes(tensors, metadata)
        metadata['ss_output_name'] = os.path.splitext(os.path.basename(save_path))[0]
        metadata['sshs_model_hash'] = model_hash
        metadata['sshs_legacy_hash'] = legacy_hash
        save_file(tensors, save_path, metadata)


if __name__ == "__main__":
    main()
