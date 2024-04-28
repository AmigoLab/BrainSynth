python3 /project/run_transformer.py run \
  --training_subjects='/path/to/training_samples.csv' \
  --validation_subjects='/path/to/validation_samples.csv' \
  --conditioning_path='/path/to/conditioning.tsv' \
  --conditionings='(\"age\", \"PTGENDER\", \"pathology\")' \
  --project_directory='/results/' \
  --experiment_name='adni_transformer' \
  --mode='training' \
  --deterministic=False \
  --cuda_benchmark=True \
  --device='ddp' \
  --seed=4 \
  --epochs=500 \
  --learning_rate=0.0005 \
  --gamma='auto' \
  --log_every=1 \
  --checkpoint_every=1 \
  --eval_every=1 \
  --batch_size=3 \
  --eval_batch_size=3 \
  --num_workers=16 \
  --prefetch_factor=16 \
  --vqvae_checkpoint='/path/to/adni/vqvae/file.pt' \
  --vqvae_aug_conditionings='continuous' \
  --vqvae_aug_load_nii_canonical=False \
  --vqvae_aug_augmentation_probability=0.20 \
  --vqvae_aug_augmentation_strength=0.0 \
  --vqvae_aug_normalize=True \
  --vqvae_aug_roi='((16,176), (16,240),(96,256))' \
  --vqvae_network='baseline_vqvae' \
  --vqvae_net_level=0 \
  --vqvae_net_use_subpixel_conv=False \
  --vqvae_net_use_slim_residual=True \
  --vqvae_net_no_levels=4 \
  --vqvae_net_downsample_parameters='((4,2,1,1),(4,2,1,1),(4,2,1,1),(4,2,1,1))' \
  --vqvae_net_upsample_parameters='((4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1))' \
  --vqvae_net_no_res_layers=3 \
  --vqvae_net_no_channels=256 \
  --vqvae_net_codebook_type='ema' \
  --vqvae_net_num_embeddings='(2048,)' \
  --vqvae_net_embedding_dim='(32,)' \
  --vqvae_net_embedding_init='(\"normal\",)' \
  --vqvae_net_commitment_cost='(0.25, )' \
  --vqvae_net_decay='(0.5,)' \
  --vqvae_net_dropout=0.0 \
  --vqvae_net_act='RELU'\
  --starting_epoch=0 \
  --ordering_type='raster_scan' \
  --transpositions_axes='((2, 0, 1),)' \
  --rot90_axes='((0, 1),)' \
  --transformation_order='(\"rotate_90\", \"transpose\")' \
  --network='xtransformer' \
  --vocab_size=2048 \
  --n_embd=512 \
  --n_layers=24 \
  --n_head=16 \
  --tie_embedding=False \
  --ff_glu=False \
  --emb_dropout=0.1 \
  --ff_dropout=0.1 \
  --attn_dropout=0.1 \
  --use_rezero=False \
  --position_emb='rotary' \
  --conditioning_type='cross_attend' \
  --use_continuous_conditioning='(True, True, True)' \
  --local_attn_heads=8 \
  --local_window_size=420 \
  --feature_redraw_interval=1 \
  --generalized_attention=False \
  --use_rmsnorm=True \
  --attn_talking_heads=False \
  --attn_on_attn=False \
  --attn_gate_values=True \
  --sandwich_norm=False \
  --rel_pos_bias=False \
  --use_qk_norm_attn=False \
  --spatial_rel_pos_bias=True \
  --bucket_values=False \
  --shift_mem_down=1
