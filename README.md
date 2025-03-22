# Gemma-EasyLM For 2b

This document outlines the integration of the Gemma model into the EasyLM framework, including instructions for training, converting the model format, and serving the model with Gradio.

## Training: Integrating HF Flax Weights into EasyLM

You are in 2B version now.
(view also [7B version](https://github.com/deveworld/Gemma-EasyLM/))

### Step 1: Consolidate Flax Weights from Hugging Face

> You can skip this step with downloading https://huggingface.co/beomi/gemma-ko-2b/resolve/flax-init/flax_model.msgpack

Firstly, sharded Flax Gemma model weights available at: [Hugging Face - Gemma 2B](https://huggingface.co/google/gemma-2b/tree/flax).

Use the following example code to accomplish this:

```python
from transformers import FlaxAutoModelForCausalLM

model = FlaxAutoModelForCausalLM.from_pretrained("google/gemma-2b", revision="flax")
model.save_pretrained("./flax-concatted", max_shard_size="99GB")
```

This script generates a `flax-concatted/flax_model.msgpack` file. We will utilize this `.msgpack` file during the training process.

### Step 2: Upload the .msgpack File to Google Cloud Storage (GCS)

Execute the following command to rename and upload the generated `.msgpack` file to your GCS repository:

```bash
mv ./flax-concatted/flax_model.msgpack ./flax-concatted/flax_2b_model.msgpack 
gsutil cp ./flax-concatted/flax_2b_model.msgpack gs://YOUR_GCS_REPO_NAME
```

### Step 3: Modify the `train.sh` Script

Adjust the paths for `load_checkpoint`, `train_dataset.json_dataset.path`, and `logger.output_dir` within the `train.sh` script to match your setup.

The provided example `train.sh` script assumes training will be conducted on a TPUv4-64 pod slice.

### Step 4: Initiate Training

Execute the training script to start the training process:

```
./train.sh
```

## Conversion: From EasyLM to Hugging Face Format

### Step 1: Retrieve the `streaming_train_state` File

Download the `streaming_train_state` file from your GCS repository using the following command:

```
gsutil cp gs://YOUR_GCS_REPO_NAME/.../streaming_train_state_STEPMO .
```

Note: The file name will either be `streaming_train_state` or `streaming_train_state_STEPNO`.

### Step 2: Execute the Conversion Script

Run the conversion script to convert the EasyLM model format to Hugging Face's format:

```
python convert_easylm_stream_to_hf_safetensors.py ./streaming_train_state_STEPNO
```

### Step 3: Verify the Output Files

Check the generated output files in the `./gemma-ko-2b-dev` directory.

> The Flax-version of the weight file can be found in the `./flax-gemma-ko-2b` folder.

## Serving the Model with Gradio

To serve the model using Gradio, follow these steps:

```
cd EasyLM/models/gemma
pip install -r serving_requirements.txt
./serve_test.sh <path of flax_model.msgpack>
```

where flax_model.msgpack is in ./flax-gemma-ko-2b/

## Original EasyLM Reference
If you found EasyLM useful in your research or applications, please cite using the following BibTeX:
```
@software{geng2023easylm,
  author = {Geng, Xinyang},
  title = {EasyLM: A Simple And Scalable Training Framework for Large Language Models},
  month = March,
  year = 2023,
  url = {https://github.com/young-geng/EasyLM}
}
```

## Credits
* The LLaMA implementation is from [JAX_llama](https://github.com/Sea-Snell/JAX_llama)
* The JAX/Flax GPT-J and RoBERTa implementation are from [transformers](https://huggingface.co/docs/transformers/main/en/index)
* Most of the JAX utilities are from [mlxu](https://github.com/young-geng/mlxu)
* The codebase is heavily inspired by [JAXSeq](https://github.com/Sea-Snell/JAXSeq)
