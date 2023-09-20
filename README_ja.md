# apply_block_weight
[[English](./README.md)] [日本語]

## What is this?
これは、LoRA Block Weight（lbw）をsd1.5向けのLoRAに適用し、新しい .safetensors ファイルを出力するためのユーティリティです。  
現在、17層または26層のみがサポートされています。  

~~※StableDiffusion の機能拡張ではなく単独で動作するユーティリティです。~~  
**a1111 webuiに対応しました。"Extensions > Install from URL"タブからインストールできます。**  
txt2img/img2imgタブ内のScriptの中から"Apply LBW"を選択してください。

## 単独のツールとしてインストールするには
1. zip ファイルをダウンロードして任意の場所に解凍する
2. PowerShell を起動する
3. 下記のコマンドを実行する
```
> cd {解凍したフォルダ}
> python -m venv venv
> venv\Sctipt\activate
> pip install -r requirements.txt
```
### インストール時のトラブル
#### `venv\Sctipt\activate` でエラーが発生
PowerShell を**管理者権限で起動**して、下記のコマンドを実行してください。
```
> Set-ExecutionPolicy RemoteSigned
```
ポリシーを変更するか質問してくるので **[Y]** を入力します。
#### `pip install` でエラーが発生
`requirements.txt` に記載されたライブラリを個別にインストールしてください。
```
> pip install numpy
> pip install torch
> pip install safetensors
```

## 使い方
```
> python apply_lora_block_weight.py {input} {output} {block weight}
```
### 使用例1

- 入力ファイル: e:¥foo.safetensor
- 出力ファイル: e:¥foo_lbw.safetensor（名前は任意）
- BlockWeight: 0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1
```
> python apply_lora_block_weight.py e:¥foo.safetensor e:¥foo_lbw.safetensor 0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1
```
### 使用例2
もしBASE値をマイナスにしたい場合は、BlockWeight の前に `--` を挿入してください
```
> python apply_lora_block_weight.py e:¥foo.safetensor e:¥foo_lbw.safetensor -- -1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

## License
stable-diffusion-webui: AGPL-3.0 license  
sd-webui-lora-block-weight: ?  
ss-scripts: Apache-2.0 license
