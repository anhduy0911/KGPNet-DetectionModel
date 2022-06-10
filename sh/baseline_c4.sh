if [ $1 -eq 1 ]
then
    echo "Parallel"
    TORCH_DISTRIBUTED_DEBUG=INFO torchrun main.py --name baseline_c4 --mode train --batch_size 8 --n_gpus 2
else
    echo "Sequential"
    CUDA_VISIBLE_DEVICES=1 python main.py --name baseline_c4 --mode train --batch_size 8 --n_gpus 1
fi