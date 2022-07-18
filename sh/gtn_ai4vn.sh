if [ $1 -eq 1 ]
then
    echo "Parallel"
    TORCH_DISTRIBUTED_DEBUG=INFO torchrun main.py --name KGP_e2e_gtn_ai4vn --mode test --batch_size 16 --n_gpus 4
else
    echo "Sequential"
    CUDA_VISIBLE_DEVICES=0 python main.py --name KGP_e2e_gtn_ai4vn --mode test --batch_size 2 --n_gpus 1
fi