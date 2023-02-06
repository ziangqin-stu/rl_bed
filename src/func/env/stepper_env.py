import torch
import time
import yaml
# from base_env import BaseEnvironment


class StepperDiscreteEnvironment:
    def __init__(self, conf):
        self.conf = conf
        self.name = "StepperEnvironment"
        self._build_sandbox()

    def _build_sandbox(self):
        self.sandbox = {
            "bonus_map": self.conf['bonus_map'],
            "position": int(self.conf['init_position']),
            "length": int(len(self.conf['bonus_map'])),
        }
        return self.sandbox

    def _build_observation(self) -> torch.Tensor:
        observation = torch.tensor(self.sandbox['position'], dtype=torch.float32)
        return observation

    def reset(self):
        self._build_sandbox()
        _info = {
            'status': 'reset',
            'action': None
        }
        observation = self._build_observation()
        return observation, _info

    def step(self, _action: torch.Tensor):
        # pretreatment
        action_value = int(_action.item())
        # update sandbox
        new_position = self.sandbox['position'] + action_value  # compute new position w.r.t incoming action
        new_position = max(0, min(self.sandbox['length'] - 1, new_position))  # clip new position into valid range
        self.sandbox['position'] = new_position
        _reward = self.sandbox['bonus_map'][self.sandbox['position']]  # consume reward
        self.sandbox['bonus_map'][self.sandbox['position']] = 0.0
        # compute done sate
        _done = self.sandbox['position'] == self.sandbox['length'] - 1
        _truncated = False
        # return
        observation = self._build_observation()
        _info = {
            'status': 'running',
            'action': action_value
        }
        return observation, _reward, _done, _truncated, _info

    def render(self, animate=False, frame_interval=1.0):
        if animate:
            print('\r', end='')
        for i, e in enumerate(self.sandbox['bonus_map']):
            if i == self.sandbox['position']:
                print('\033[1;4m' + f'{e:.1f}' + '\033[0m' + ' | ', end='')
            else:
                print(f'{e:.1f} | ', end='')
        if animate:
            time.sleep(frame_interval)
        else:
            print('')


if __name__ == "__main__":


    env = StepperDiscreteEnvironment(None)
    obs, _ = env.reset()
    env.render(animate=True)

    action = torch.tensor(4, dtype=torch.float32)
    obs, reward, done, truncated, info = env.step(action)
    env.render(animate=True)
