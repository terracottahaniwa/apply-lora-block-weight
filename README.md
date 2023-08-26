# apply_block_weight
## What is this?
This is a utility to apply LoRA Block Weight (lbw) to LoRA for sd1.5 and export to a new safetensors.  
Currently only 17 or 26 layers are supported.
## Install
```
python -m venv venv
venv\Sctipt\activate
pip install -r requirements.txt
```
## How to use
To run
```
python apply_block_weight.py [-h] input output ratios

positional arguments:
  input       path to input safetensors
  output      path to output safetensors
  ratios      the lbw numeric array
```
If you want the BASE value to be negative, the LBW array is considered a command line option.  
To avoid this, insert -- before the array as follows  
```
python apply_block_weight.py input output -- -1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```
## License
stable-diffusion-webui: AGPL-3.0 license  
sd-webui-lora-block-weight: ?  
ss-scripts: Apache-2.0 license  
