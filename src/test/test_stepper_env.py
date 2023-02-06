import os
from ..func.utils.configs import *
from ..func.utils.utils import load_param
from ..func.env.stepper_env import *


def test_stepper_env_basic_function():
    # load parameters
    conf_path_name = os.path.join(TEST_PATH, "test_stepper_env.yml")
    conf = load_param(conf_path_name)

    env = StepperDiscreteEnvironment(conf)
    # render
    env.render()
    env.render(animate=True)
    # reset
    obs, info = env.reset()
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert info == {'status': 'reset', 'action': None}
    # step-1
    action = torch.tensor(4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(4.0, dtype=torch.float32)
    assert reward == float(-1.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(4)}
    # step-2
    action = torch.tensor(2, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(6.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(2)}
    # step-3
    action = torch.tensor(3, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(9.0, dtype=torch.float32)
    assert reward == float(1.0)
    assert done == bool(True)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(3)}
    # re-reset
    obs, info = env.reset()
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert info == {'status': 'reset', 'action': None}


def test_stepper_env_corner_case():
    # 撞墙
    # 反复撞墙
    # 左右震荡
    # 冲线
    pass


def test_stepper_env_efficiency():
    # 初始化速度
    # 执行速度
    # 可视化速度
    pass
