import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from networks import convert_diffusers_name_to_compvis
from train_util import precalculate_safetensors_hashes


# settings

load_path = r"path_to_input_model.safetensors"
save_path = r"path_to_output_model.safetensors"
ratios = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


def main():
    BLOCKID17 = [
        "BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00",
        "OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
    
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
        assert blockid in BLOCKID17, blockid
        return blockid

    ratio_of_ = dict(zip(BLOCKID17, ratios))
    tensors = {}
    with safe_open(load_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
            compvis_name = convert_diffusers_name_to_compvis(key, False)
            blockid = compvis_name_to_blockid(compvis_name)
            if compvis_name.endswith("lora_up.weight"):
                tensors[key] *= ratio_of_[blockid]
            print(f"{blockid}:{ratio_of_[blockid]}\t{key}\n\t{compvis_name}")
        metadata = f.metadata()
        model_hash, legacy_hash = precalculate_safetensors_hashes(tensors, metadata)
        metadata['ss_output_name'] = os.path.splitext(os.path.basename(save_path))[0]
        metadata['sshs_model_hash'] = model_hash
        metadata['sshs_legacy_hash'] = legacy_hash
        save_file(tensors, save_path, metadata)


if __name__ == "__main__":
    main()
