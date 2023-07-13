CUDA_VISIBLE_DEVICES=0 python main.py --file_name='df_8class_0414' --ids_entity='ids_entity_0414_token' \
                                      --batch_size=4 --scheduler=linear --weight_decay=1e-2 --lr=5e-6 --epochs=30 \
                                      --ratio=0.2 --max_len=512 --seed=20