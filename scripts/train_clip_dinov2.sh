now=$(date +"%Y%m%d_%H%M%S")
port=$(($RANDOM + 10000))

method=$1
config=$2
n_gpus=$3

echo 'Start training on port' $port
python -m torch.distributed.launch \
    --nproc_per_node=$n_gpus \
    --master_addr=localhost \
    --master_port=$port \
    $method.py \
    --config=$config --port $port --model_2 mmseg.dual_model_dinov2 2>&1