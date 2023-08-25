# apply_block_weight
## What is this?
This is a utility to apply LoRA Block Weight (lbw) to LoRA for sd1.5 and export to a new safetensors.
## install
```
python -m venv venv
venv\Sctipt\activate
pip install -r requirements.txt
```
## How to use
Edit the line in apply_block_weight.py  
Ratios is the lbw numbers.  
Currently only LoRA (17 layers) for sd1.5 created in sd-scripts is supported.
```
# settings

load_path = r"path_to_input_model.safetensors"
save_path = r"path_to_output_model.safetensors"
ratios = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
```
To run
```
python apply_block_weight.py
```
## license
stable-diffusion-webui: AGPL-3.0 license  
sd-webui-lora-block-weight: ?  
ss-scripts: Apache-2.0 license  
