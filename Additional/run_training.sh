CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 --master_port 25047\
    run.py --seed 152\
    --original 0\
    --data-pth ./data\
    --log-pth ./save/test_lr\
    --model resnet56\
    --resume 250 --resume-pth ./save/postlocal/phase1_common.pt\
    --batch-size 128 --total-batch-size 2048 --steps-per-epoch 24\
    --debug 1 --wandb-save 0\
    --epochs 150\
    --start-lr 0.0625 --max-lr -1 --wd 0.0005\
    --momentum 0 --nesterov 0 --warm-up 0 --warmup-epochs 50\
    --bn 1 --bn-batches 100 --group-weight 1 --n-groups -1\
    --decay1 25000 --decay2 25000\
    --round_values 1 2 5 10 20 50 100 400 600 1200 1800 3600\
    --eval-freq1 1000\
    --save-freq1 0\
    --eval-on-start 0\
    --replacement 0 --aug 1\
    --label-noise 0 --noise-p 0
    # --wandb disabled
# --seed $(($RANDOM * 32768 + $RANDOM))