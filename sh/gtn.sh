if [ $1 -eq 1 ]
then
    echo "Parallel"
    CUDA_VISIBLE_DEVICES=1,2 torchrun main.py --name KGP_e2e_gtn --mode test --batch_size 4 --n_gpus 2 --resume_path model_best.pth
else
    echo "Sequential"
    CUDA_VISIBLE_DEVICES=2 python main.py --name KGP_e2e_gtn --mode train --batch_size 4 --n_gpus 1
fi