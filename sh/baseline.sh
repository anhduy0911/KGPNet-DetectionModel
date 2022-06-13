if [ $1 -eq 1 ]
then
    echo "Parallel"
    TORCH_DISTRIBUTED_DEBUG=INFO torchrun main.py --name baseline --mode train --batch_size 8 --n_gpus 2
else
    echo "Sequential"
    CUDA_VISIBLE_DEVICES=1 python main.py --name baseline --mode test_pres --batch_size 4 --n_gpus 1 --pres_id 20210504_225719445934
fi