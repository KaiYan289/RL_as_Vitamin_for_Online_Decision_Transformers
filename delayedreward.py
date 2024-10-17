import gym
import d4rl

class CumulativeRewardWrapper(gym.Wrapper):
    def __init__(self, env, wraplen):
        super().__init__(env)
        self.cumulative_reward = 0
        self.wraplen = int(wraplen)
        self.step_count = 0

    def reset(self, **kwargs):
        self.cumulative_reward = 0
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.cumulative_reward += reward
        self.step_count += 1
        info['single_reward'] = reward
        # Check if it's time to report the cumulative reward
        if self.step_count % self.wraplen == 0 or done:
            if done:
                # Reset cumulative reward if the episode is done
                self.step_count = 0
            reward = self.cumulative_reward
            self.cumulative_reward = 0

        else: reward = 0

        return observation, reward, done, info
        
if __name__ == "__main__":
    env = CumulativeRewardWrapper(gym.make('halfcheetah-random-v2'), 5)
    
    for _ in range(1):
        done = False
        i = 0
        o = env.reset()
        while not done:
            a = env.action_space.sample()
            o, r, done, info = env.step(a)
            print("i:", i, "reward:", r, "done:", done, "info:", info)
            i += 1