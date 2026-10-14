"""Test local GPU math, actual ball model and a public-video frame."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'pingpang'
sys.path.insert(0, str(TASK))


def main():
    import torch
    import av
    from core.types import RunContext
    from participant.solution_local import Solution

    print('Python:', sys.version.split()[0])
    print('PyTorch:', torch.__version__)
    assert torch.cuda.is_available(), 'CUDA GPU unavailable'
    print('GPU:', torch.cuda.get_device_name(0))
    a = torch.ones((32, 32), device='cuda')
    assert (a @ a).sum().item() == 32768
    output = ROOT / 'out/local_env_check'
    output.mkdir(parents=True, exist_ok=True)
    context = RunContext(input_dir=str(TASK/'public_data'), output_dir=str(output), decoder='cpu')
    Solution().prepare(context)
    print('Actual model:', json.dumps(context.info, ensure_ascii=False))
    data = (TASK / 'public_data').resolve()
    if os.name == 'nt':
        data = Path('\\\\?\\' + str(data))
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    with av.open(str(data / manifest['videos'][0]['path'])) as container:
        frame = next(container.decode(video=0))
        print('Video frame:', frame.width, frame.height)
    print('LOCAL ENV CHECK PASSED')


if __name__ == '__main__':
    main()
