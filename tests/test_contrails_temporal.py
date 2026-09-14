import pytest
import torch

from sciona.contrails_temporal import create_four_panels, create_four_panels_tta


@pytest.mark.parametrize('rotation', range(4))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_temporal_quadrants_stay_fixed_under_spatial_augmentation(rotation, dtype):
    x = torch.arange(2 * 4 * 3 * 3 * 3, dtype=dtype).reshape(2, 4, 3, 3, 3)
    result = create_four_panels_tta(x, rotation)
    assert result.shape == (4, 3, 6, 6) and result.dtype == torch.float32
    for batch in range(2):
        for frame, row, col in [(3, 0, 0), (0, 0, 3), (1, 3, 0), (2, 3, 3)]:
            expected = torch.rot90(x[batch, frame], rotation, (-2, -1)).float()
            torch.testing.assert_close(result[batch, :, row:row+3, col:col+3], expected)
            torch.testing.assert_close(result[batch+2, :, row:row+3, col:col+3], expected.flip(-1))
    if rotation:
        assert not torch.equal(result[:2], torch.rot90(create_four_panels(x), rotation, (-2, -1)))


def test_panel_assembly_preserves_training_gradient():
    x = torch.zeros(1, 4, 3, 2, 2, requires_grad=True)
    panel = create_four_panels(x)
    (panel[:, :, :2, :2].sum() * 7 + panel[:, :, 2:, 2:].sum() * 3).backward()
    assert torch.all(x.grad[:, 3] == 7)
    assert torch.all(x.grad[:, 2] == 3)
    assert torch.all(x.grad[:, :2] == 0)


@pytest.mark.parametrize('shape', [(1, 3, 3, 2, 2), (1, 4, 2, 2, 2), (0, 4, 3, 2, 2), (1, 4, 3, 2, 3)])
def test_invalid_temporal_shape(shape):
    with pytest.raises(ValueError):
        create_four_panels(torch.zeros(shape))


@pytest.mark.parametrize('rotation', [-1, 4, True, 1.5])
def test_invalid_rotation(rotation):
    with pytest.raises(ValueError):
        create_four_panels_tta(torch.zeros(1, 4, 3, 2, 2), rotation)
