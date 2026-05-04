NiT Diffusers integration
=========================

This repository now mirrors the structure expected for upstream Diffusers
integration. The core implementation lives under `src/diffusers`:

- `models/transformers/transformer_nit.py` provides `NiTTransformer2DModel`
  as a `ModelMixin`/`ConfigMixin` transformer.
- `schedulers/scheduling_flow_match_nit.py` provides `NiTFlowMatchScheduler`
  with ODE and SDE sampling modes.
- `pipelines/nit/pipeline_nit.py` provides `NiTPipeline` for class-conditional
  image generation.
- `scripts/convert_nit_to_diffusers.py` converts original NiT checkpoints into
  a Diffusers pipeline directory.

Convert a checkpoint
--------------------

```bash
python scripts/convert_nit_to_diffusers.py \
  --checkpoint checkpoints/nit_xl_model_1000K.safetensors \
  --output nit-xl-1000k-diffusers \
  --model-size nit-xl \
  --mode ode
```

The output directory contains:

```text
model_index.json
scheduler/scheduler_config.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
vae_pretrained_model_name_or_path.txt
```

Use `--copy-vae /path/to/vae` to vendor a local VAE into `output/vae`.

Sample from a converted checkpoint
----------------------------------

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

For direct upstreaming to `huggingface/diffusers`, copy the files under
`src/diffusers` into the corresponding Diffusers package locations and add the
classes to Diffusers' lazy import tables.
