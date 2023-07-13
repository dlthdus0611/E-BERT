CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=4 --file_name='df_8class_0414' --ids_entity='ids_entity_0414_token' --seed=20 \
                                      --arch=bert --scheduler=linear --weight_decay=1e-2 --lr=5e-6 --epochs=30 --max_len=512 \
                                      --data=all --ver=base+etc --ratio=0.2 --Kfold=5 --server=lsy