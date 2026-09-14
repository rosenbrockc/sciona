"""Check actual tensor transport, separately from worker-side hash replay."""
import json
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticTensors(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        return torch.full((3, 32, 32), float(index)), index


def main():
    torch.set_num_threads(1)
    loader = DataLoader(SyntheticTensors(), batch_size=2, num_workers=16,
                        multiprocessing_context='spawn', timeout=30)
    count = 0
    for images, indices in loader:
        for image, index in zip(images, indices):
            assert torch.equal(image, torch.full_like(image, float(index)))
            count += 1
    assert count == 32
    print(json.dumps(dict(passed=True, samples=count, workers=16,
                          tensor_transport_exact=True, approved=False)))


if __name__ == '__main__':
    main()
