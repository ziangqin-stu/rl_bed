class BaseEnvironment:
    def __init__(self):
        self.name = "BaseEnvironment"
        self.body = self.build_body()

    def _build_body(self):
        return None

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass

