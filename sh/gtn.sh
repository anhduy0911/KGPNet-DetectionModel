if [ $1 -eq 1 ]
then
    echo "Parallel"
    TORCH_DISTRIBUTED_DEBUG=INFO torchrun main.py --name KGP_e2e_gtn --mode train --batch_size 4 --n_gpus 2 --resume True
else
    echo "Sequential"
    CUDA_VISIBLE_DEVICES=1 python main.py --name KGP_e2e_gtn --mode train --batch_size 2 --n_gpus 1
fi