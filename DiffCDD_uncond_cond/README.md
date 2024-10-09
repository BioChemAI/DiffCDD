# DiffCDD-uncond and DiffCDD-cond

# Usage

This section is about sampling with DiffCDD-uncond and DiffCDD-cond

# DiffCDD-uncond
Unconditional generation, first sample and save latent vectors,this generates 5000 latent vectors
```
python scripts/pl_sample_t_uncond  --model_path ./Models/uncond_model004200.pt --save_path wherever/you/like.pkl --diffusion_steps 200 --noise_schedule linear --rescale_timesteps False
```
Decode latent vectors into molecules
```
python scripts/decode_save_single.py --sampled_vec saved/path/ --save_path_10k path/to/valid/SMILES --save_path_full path/to/all/SMILES --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

# DiffCDD-cond

## Sample for a single protein target
Input the protein sequence instead.
```
python scripts/pl_sample_singleseq.py  --model_path ../Models/model009000.pt  --save_path path/to/a/pklfile --num_samples amount/to/sample\\
       --protein_seq "Sequenc you want to sample with"  --diffusion_steps 300 --noise_schedule linear --rescale_timesteps False \\
```
Decode latent vectors into molecules
```
python scripts/decode_save_single.py --sampled_vec saved/path/ --save_path_10k path/to/valid/SMILES --save_path_full path/to/all/SMILES --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

## Sample for all pockets in the crossdocked2020 testset
This will sample 200 latent vectors for each pocket in the testset, the processed Crossdocked2020 test set is provided in Models Folder.
```
python scripts/pl_sample_full.py --model_path ./Models/cond_model009000.pt --save_path path/to/a/directory! --diffusion_steps 300 --noise_schedule linear --rescale_timesteps False 
```
Decode them.
```
python scripts/decode_save.py --sampled_vec path/of/latent vectors --save_path_10k path/for/100valid/molecules/for/each/pocket --save_path_full path/for/all/decoded/molecules --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

