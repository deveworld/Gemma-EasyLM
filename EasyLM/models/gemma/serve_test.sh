export PYTHONPATH=$(realpath ../../../):$PYTHONPATH
export CUDA_VISIBLE_DEVICES='1'
python gemma_serve.py \
--load_checkpoint="flax_params::$1" \
--load_gemma_config='7b' \
--input_length=30 \
--seq_length=8192 \
--add_bos_token 
