mkdir exps
mkdir exps/voc
mkdir exps/voc/base_train
mkdir exps/voc/seed01_1shot
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./tools/run_dist_launch.sh 6 python3.6 -u main.py \
    --dataset_file voc_base1 \
    --backbone resnet101 \
    --num_feature_levels 1 \
    --enc_layers 6 \
    --dec_layers 6 \
    --hidden_dim 256 \
    --num_queries 300 \
    --batch_size 4 \
    --category_codes_cls_loss \
    --resume ./exps/voc/base_train/voc1-checkpoint.pth \
    --fewshot_finetune \
    --fewshot_seed 01 \
    --num_shots 1 \
    --epoch 500 \
    --lr_drop_milestones 300 450 \
    --warmup_epochs 50 \
    --save_every_epoch 50 \
    --eval_every_epoch 1 \
    --output_dir exps/voc/seed01_1shot \
2>&1 | tee exps/voc/seed01_1shot/log.txt

