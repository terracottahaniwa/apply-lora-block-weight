# apply_block_weight

[English] [[日本語](./README_ja.md)]

## What is this?

This is a utility to apply LoRA Block Weight (lbw) to LoRA for sd1.5 and export to a new safetensors.
Currently only 17 or 26 layers are supported.

~~※This is a standalone utility, not an extension of StableDiffusion's functionalities.~~


## Install

1. Download the zip file and extract it to a location of your choice.
2. Launch PowerShell.
3. Execute the following command.

```
> cd {extract folder}
> python -m venv venv
> venv\Sctipt\activate
> pip install -r requirements.txt
```

### Troubleshooting

#### Error with `venv\Scripts\activate`

Please **launch PowerShell as an administrator** and execute the following command

```
> Set-ExecutionPolicy RemoteSigned
```

It will ask whether to change the policy, so input **[Y]**.


#### Error Occurs with `pip install`

Please install the libraries listed in `requirements.txt` individually.

```
> pip install numpy
> pip install torch
> pip install safetensors
```


## How to use

```
> python apply_block_weight.py {input} {output} {block weight}
```

### Example 1

- input file: e:¥foo.safetensor
- output file: e:¥foo_lbw.safetensor (Name is optional)
- BlockWeight: 0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1

```
> python apply_block_weight.py e:¥foo.safetensor e:¥foo_lbw.safetensor 0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1
```

### Example 2

If you want to make the BASE value negative, please insert -- before BlockWeight.

```
> python apply_block_weight.py e:¥foo.safetensor e:¥foo_lbw.safetensor -- -1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```


## License

stable-diffusion-webui: AGPL-3.0 license
sd-webui-lora-block-weight: ?
ss-scripts: Apache-2.0 license
