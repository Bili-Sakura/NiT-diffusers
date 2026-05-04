# Native-Resolution Image Synthesis for Diffusers

This repository contains a Diffusers-style implementation of Native-resolution
diffusion Transformer (NiT). The legacy standalone training, preprocessing,
sampling, YAML config, and utility codepaths have been removed so the remaining
tree mirrors the package boundaries used by `huggingface/diffusers`.

## Package layout

- `src/diffusers/models/transformers/transformer_nit.py`:
  `NiTTransformer2DModel`, a `ModelMixin`/`ConfigMixin` class-conditional
  transformer.
- `src/diffusers/schedulers/scheduling_flow_match_nit.py`:
  `NiTFlowMatchScheduler`, including ODE and SDE flow-matching updates.
- `src/diffusers/pipelines/nit/pipeline_nit.py`:
  `NiTPipeline`, a Diffusers pipeline with classifier-free guidance, VAE
  decoding, and native-resolution latent sampling.
- `scripts/convert_nit_to_diffusers.py`:
  converts original NiT checkpoints to a Diffusers pipeline directory.
- `scripts/sample_nit.py`:
  samples from a converted pipeline.

## Convert a checkpoint

```bash
python scripts/convert_nit_to_diffusers.py \
  --checkpoint checkpoints/nit_xl_model_1000K.safetensors \
  --output nit-xl-1000k-diffusers \
  --model-size nit-xl \
  --mode ode
```

The converted directory contains `model_index.json`, transformer weights and
config, scheduler config, and a VAE reference. Use `--copy-vae /path/to/vae` to
copy a local VAE into the output directory.

## Sample

```bash
python scripts/sample_nit.py \
  --model nit-xl-1000k-diffusers \
  --class-label 207 \
  --height 512 \
  --width 512 \
  --mode sde \
  --num-inference-steps 250 \
  --guidance-scale 2.05 \
  --guidance-low 0.0 \
  --guidance-high 0.7
```

## Upstreaming to Diffusers

Copy the files under `src/diffusers` into the matching locations in the
`huggingface/diffusers` repository and add the classes to Diffusers' lazy import
tables. The module names and save/load artifacts are already aligned with the
Diffusers package conventions.

## Citation

```bibtex
@article{wang2025native,
  title={Native-Resolution Image Synthesis},
  author={Wang, Zidong and Bai, Lei and Yue, Xiangyu and Ouyang, Wanli and Zhang, Yiyuan},
  year={2025},
  eprint={2506.03131},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

This project is licensed under the Apache-2.0 license.
