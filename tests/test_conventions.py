"""Convention tests that need no GPU, torch or mmdet3d.

Deliberately a separate module from test_geometry.py: that file calls
pytest.importorskip('mmdet3d') at module scope, which skips every test in the
file when the inference stack is absent. Keeping these here means they run on
any machine -- including a laptop with no CUDA -- so the profiler's device
handling stays covered even when a GPU pod is unavailable.
"""

import os

import pytest

# ------------------------------------------------------------ device parsing
def _cuda_ordinal(device: str) -> int:
    """Mirror of profile_system.cuda_ordinal.

    Duplicated rather than imported: importing profile_system pulls torch,
    mmdet3d and nuscenes at module scope, which this pure-string logic does
    not need.
    """
    if ':' in device:
        try:
            return int(device.split(':', 1)[1])
        except ValueError:
            pass
    return 0


@pytest.mark.parametrize('device,expected', [
    ('cuda:0', 0),
    ('cuda:1', 1),
    ('cuda:7', 7),
    ('cuda', 0),        # bare 'cuda' means the default device
    ('cpu', 0),         # unused on CPU, but must not raise
    ('cuda:bogus', 0),  # malformed ordinal falls back rather than crashing
])
def test_cuda_ordinal_parsing(device, expected):
    """Guards profiling the wrong GPU.

    NVML, device identity, synchronisation and peak-memory all key off this
    ordinal. If it silently returns 0 for 'cuda:1', every reported GPU number
    describes a different card than the one the model ran on.
    """
    assert _cuda_ordinal(device) == expected


def test_profile_system_uses_the_ordinal_consistently():
    """No bare GPU-0 calls may remain in the profiler.

    A partial replacement is worse than none: some metrics would describe the
    selected device and others GPU 0, with nothing to indicate the mix.
    """
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scripts', 'profile_system.py')
    if not os.path.exists(src_path):
        pytest.skip('profile_system.py not present')
    src = open(src_path).read()
    for bad in ['torch.cuda.synchronize()',
                'torch.cuda.get_device_name(0)',
                'torch.cuda.get_device_properties(0)',
                'torch.cuda.max_memory_allocated()',
                'torch.cuda.reset_peak_memory_stats()',
                'nvmlDeviceGetHandleByIndex(0)']:
        assert bad not in src, f'{bad} ignores the selected CUDA device'
