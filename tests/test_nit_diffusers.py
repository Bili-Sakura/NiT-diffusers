import pytest

torch = pytest.importorskip("torch")

from diffusers.models.transformers import NiTTransformer2DModel
from diffusers.schedulers import NiTFlowMatchScheduler


def test_nit_transformer_forward_image_latents():
    model = NiTTransformer2DModel(
        patch_size=1,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        encoder_depth=1,
        projector_dim=16,
        z_dim=8,
        num_classes=10,
    )
    latents = torch.randn(2, 4, 4, 4)
    timesteps = torch.tensor([1.0, 0.5])
    class_labels = torch.tensor([1, 2])

    output = model(latents, timesteps, class_labels)

    assert output.sample.shape == latents.shape


def test_scheduler_ode_step_matches_velocity_update():
    scheduler = NiTFlowMatchScheduler(mode="ode")
    sample = torch.ones(1, 4, 2, 2)
    velocity = torch.full_like(sample, 2.0)

    output = scheduler.step(velocity, torch.tensor([1.0]), sample, torch.tensor([0.75]))

    assert torch.allclose(output.prev_sample, torch.full_like(sample, 0.5))


def test_scheduler_sde_final_step_is_deterministic():
    scheduler = NiTFlowMatchScheduler(mode="sde")
    sample = torch.randn(2, 4, 1, 1)
    velocity = torch.zeros_like(sample)
    image_sizes = torch.tensor([[1, 1], [1, 1]])

    output = scheduler.step(
        velocity,
        torch.tensor([0.04]),
        sample,
        torch.tensor([0.0]),
        image_sizes=image_sizes,
        final_step=True,
    )

    assert output.prev_sample.shape == sample.shape
