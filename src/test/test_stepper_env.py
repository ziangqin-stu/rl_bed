import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上一级目录
sys.path.append(config_path)

from func.env.stepper_env import *


def test_stepper_env_basic_function():
    env = StepperDiscreteEnvironment(None)

    env.render()
    env.render(animate=True)

    obs, info = env.reset()
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert info == {'status': 'reset', 'action': None}

    action = torch.tensor(4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(4.0, dtype=torch.float32)
    assert reward == float(-1.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(4)}

    action = torch.tensor(2, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(6.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(2)}

    action = torch.tensor(3, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(9.0, dtype=torch.float32)
    assert reward == float(1.0)
    assert done == bool(True)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(3)}

    obs, info = env.reset()
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert info == {'status': 'reset', 'action': None}


def test_stepper_env_corner_case():
    pass


def test_stepper_env_efficiency():
    pass
