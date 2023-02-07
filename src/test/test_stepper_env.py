from ..func.utils.configs import *
from ..func.utils.utils import *
from ..func.env.stepper_env import *


def test_stepper_env_basic_function():
    # build environment
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
    # build environment
    conf_path_name = os.path.join(TEST_PATH, "test_stepper_env.yml")
    conf = load_param(conf_path_name)
    env = StepperDiscreteEnvironment(conf)

    # 撞墙
    obs, info = env.reset()
    # forward-4
    action = torch.tensor(4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(4.0, dtype=torch.float32)
    assert reward == float(-1.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(4)}
    # backward-4
    action = torch.tensor(-4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(-4)}
    # forward-4
    action = torch.tensor(4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(4.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(4)}
    # backward-5
    action = torch.tensor(-5, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(-5)}
    # forward-2
    action = torch.tensor(2, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(2.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(2)}
    # backward-4
    action = torch.tensor(-4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(-4)}
    # backward-100
    action = torch.tensor(-100, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(-100)}

    # 左右震荡
    obs, info = env.reset()
    # forward-3
    action = torch.tensor(3, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(3.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(3)}
    # backward-2
    action = torch.tensor(-2, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(1.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(-2)}
    # forward-3
    action = torch.tensor(3, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(4.0, dtype=torch.float32)
    assert reward == float(-1.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(3)}
    # backward-4
    action = torch.tensor(-4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(-4)}

    # 冲线-直接
    obs, info = env.reset()
    # forward-100
    action = torch.tensor(100, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(9.0, dtype=torch.float32)
    assert reward == float(1.0)
    assert done == bool(True)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(100)}
    # 冲线-分段
    obs, info = env.reset()
    # forward-3
    action = torch.tensor(3, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(3.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(3)}
    # forward-10
    action = torch.tensor(10, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(9.0, dtype=torch.float32)
    assert reward == float(1.0)
    assert done == bool(True)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(10)}
    # 冲线-折返
    obs, info = env.reset()
    # backward-3
    action = torch.tensor(-3, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(0.0, dtype=torch.float32)
    assert reward == float(0.0)
    assert done == bool(False)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(-3)}
    # forward-10
    action = torch.tensor(10, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert obs == torch.tensor(9.0, dtype=torch.float32)
    assert reward == float(1.0)
    assert done == bool(True)
    assert truncated == bool(False)
    assert info == {'status': 'running', 'action': int(10)}


def test_stepper_env_efficiency():
    conf_path_name = os.path.join(TEST_PATH, "test_stepper_env.yml")
    conf = load_param(conf_path_name)
    # 初始化速度
    start_t = time.time()
    for i in list(range(100)):
        env = StepperDiscreteEnvironment(conf)
    init_speed = 100 / (time.time() - start_t)
    # 执行速度
    start_t = time.time()
    for i in list(range(50)):
        env.step(torch.tensor(1, dtype=torch.float32))
        env.step(torch.tensor(-1, dtype=torch.float32))
    step_speed = 100 / (time.time() - start_t)
    # 可视化速度
    start_t = time.time()
    for i in list(range(50)):
        env.render(animate=True, frame_interval=0.0)
    render_speed_1 = 50 / (time.time() - start_t)
    for i in list(range(50)):
        env.render(animate=False)
    render_speed_2 = 50 / (time.time() - start_t)
    print(f"\nSpeed of "
          f"\n  - init:  {init_speed:.1f}its/s"
          f"\n  - step:  {step_speed:.1f}its/s"
          f"\n  - render:{render_speed_1:.1f}its/s, {render_speed_2:.1f}its/s")