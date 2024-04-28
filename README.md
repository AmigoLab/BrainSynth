# BrainSynth [![DOI](https://zenodo.org/badge/706209780.svg)](https://zenodo.org/doi/10.5281/zenodo.10014960)
Official implementation of "Realistic Morphology-preserving Generative Modelling of the Brain"

# Pretrained models

Experiments scripts have been provided in the experiments folder and are named based on the models they have trained.

Bellow you have some toy examples and how to use pretrained models.

To use the pretrained models you need to do the following:

0) Create a docker container based on the Dockerfile and requirements file found in the dcoker folder
1) Create a folder similar with the following structure where you replace 'experiment_name' with the name of your experiment and you chose either baseline_vqvae or performer depending on which weights you want to use:
```
<<experiment_name>>
├── baseline_vqvae/performer
    ├── checkpoints 
    ├── logs
    └── outputs
```
2) Download the weights of the desired model from [here](https://drive.google.com/drive/folders/1KoOfM3NvJ_SGazWmA-Mp0V_n9q2Amj6V?usp=sharing) and put it the checkpoints folder:
3) Rename the file to 'checkpoint_epoch=0.pt'
4) Use the corresponding script from the examples bellow and remember to:
* Replace the training/validation subjects with paths towards either folder filled with .nii.gz files or towards csv/tsv files that have a path column with the full paths towards the files.
* Replace the conditioning files with the correct one for the transformer training.
* Replace the project_directory with the path were you created the folder from point 1
* Replace the experiment_name with the name of the experiment you created from point 1
5) Properly mount the paths towards the files and results folders and launch your docker container
6) Use the appropriate script for the model from bellow and change the mode to the desired one

# VQ-VAE

To extract the quantized latent representations of the images you need to run the same command as you used for training and replace the `--mode=Training` parameter with `--mode=extracting`. For decoding, you need to replace it with `--mode=decoding`.

Training script example for VQ-VAE.
```bash
python /project/run_vqvae.py run \
    --training_subjects="/path/to/training/data/tsv/" \
    --validation_subjects="/path/to/validation/data/tsv/" \
    --load_nii_canonical=False \
    --project_directory="/results/" \
    --experiment_name="example_run" \
    --mode='training' \
    --device='ddp' \
    --distributed_port=29500 \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=4 \
    --epochs=500 \
    --learning_rate=0.000165 \
    --gamma=0.99999 \
    --log_every=1 \
    --checkpoint_every=1 \
    --eval_every=1 \
    --loss='jukebox_perceptual' \
    --adversarial_component=True \
    --discriminator_network='baseline_discriminator' \
    --discriminator_learning_rate=5e-05 \
    --discriminator_loss='least_square' \
    --generator_loss='least_square' \
    --initial_factor_value=0 \
    --initial_factor_steps=25 \
    --max_factor_steps=50 \
    --max_factor_value=5 \
    --batch_size=8 \
    --normalize=True \
    --roi='((16,176), (16,240),(96,256))' \
    --eval_batch_size=8 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=172 \
    --network='baseline_vqvae' \
    --use_subpixel_conv=False \
    --use_slim_residual=True \
    --no_levels=4 \
    --downsample_parameters='((4,2,1,1),(4,2,1,1),(4,2,1,1),(4,2,1,1))' \
    --upsample_parameters='((4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1))' \
    --no_res_layers=3 \
    --no_channels=256 \
    --codebook_type='ema' \
    --num_embeddings='(2048,)' \
    --embedding_dim='(32,)' \
    --decay='(0.5,)' \
    --commitment_cost='(0.25,)' \
    --max_decay_epochs=100 \
    --dropout=0.0 \
    --act='RELU'
```

# Transformer

To sample new images from the trained model you need to run the same command as you used for training and replace the `--mode=training` parameter with `--mode=inference`.

Best performance was found by equalising normalised continuous conditioning variables.

Training script example for Transformer based on the UKB one.
```bash
python3 /project/run_transformer.py run \
    --training_subjects='/path/to/training/data/tsv/' \
    --validation_subjects='/path/to/validation/data/tsv/' \
    --conditioning_path='/path/to/continuous/equalised/tsv/' \
    --conditionings='(\"used\", \"conditioning\", \"columns\")' \
    --project_directory='/results/' \
    --experiment_name='example_run' \
    --mode='training' \
    --deterministic=False \
    --cuda_benchmark=False \
    --cuda_enable=True \
    --use_zero=True \
    --device='ddp' \
    --seed=4 \
    --epochs=500 \
    --learning_rate=0.0005 \
    --gamma='auto' \
    --log_every=1 \
    --checkpoint_every=1 \
    --eval_every=0 \
    --weighted_sampling=True \
    --batch_size=2 \
    --eval_batch_size=2 \
    --num_workers=16 \
    --prefetch_factor=16 \
    --vqvae_checkpoint='/path/to/vqvae/checkpoint/' \
    --vqvae_aug_conditionings='none' \
    --vqvae_aug_load_nii_canonical=False \
    --vqvae_aug_augmentation_probability=0.00 \
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
    --n_embd=1024 \
    --n_layers=36 \
    --n_head=16 \
    --tie_embedding=False \
    --ff_glu=False \
    --emb_dropout=0.001 \
    --ff_dropout=0.001 \
    --attn_dropout=0.001 \
    --use_rezero=False \
    --position_emb='rotary' \
    --conditioning_type='cross_attend' \
    --use_continuous_conditioning='(True, True, True, True)' \
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
```

# Acknowledgements

Work done through the collaboration between NVIDIA and KCL.

The models in this work were trained on [NVIDIA Cambridge-1](https://www.nvidia.com/en-us/industries/healthcare-life-sciences/cambridge-1/), the UK’s largest supercomputer, aimed at accelerating digital biology.

# Funding
- Jointly with UCL - Wellcome Flagship Programme (WT213038/Z/18/Z)
- Wellcome/EPSRC Centre for Medical Engineering (WT203148/Z/16/Z)
- EPSRC Research Council DTP (EP/R513064/1)
- The London AI Center for Value-Based Healthcare
- GE Healthcare
- Intramural Research Program of the NIMH (ZIC-MH002960 and ZIC-MH002968).
- European Union’s HORIZON 2020 Research 
- Innovation Programme under the Marie Sklodowska-Curie Grant Agreement No 814302
- UCLH NIHR Biomedical Research Centre.
