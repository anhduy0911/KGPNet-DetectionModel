if [ $1 -eq 1 ]
then
    echo "Parallel"
    torchrun main.py --name KGP_e2e_gtn --mode train --batch_size 4 --n_gpus 2
else
    echo "Sequential"
    CUDA_VISIBLE_DEVICES=1 python main.py --name KGP_e2e_gtn --mode test --batch_size 4 --n_gpus 1 --resume_path 0004999.pth
fi