import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]


class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:
            dp_results = np.load('./result/dp.npz')
            self.values = dp_results['V']
            self.policy = dp_results['PI']
        else:
            self.values = np.zeros((HEIGHT, WIDTH))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)




    def policy_evaluation(self, iter, env, policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        new_state_values = np.zeros((HEIGHT, WIDTH))
        iteration = 0
        # threshold 값을 1e-3으로 설정
        theta = 1e-3
        # 기존 state value function에 해당하는 old_state_values 행렬을 self.values에 저장된 행렬로 초기화
        old_state_values = self.values.copy()
        # Policy Evaluation Loop 를 설정
        while True:
            # 모든 state에 대하여 value function을 계산 (full backup)
            for height in range(HEIGHT):
                for width in range(WIDTH):
                    if [height, width] in env.obstacles:    # 현재 state가 obstacle의 위치에 해당된다면
                        new_state_values[height, width] = 0 # state value 값은 0으로 설정한다
                    else:   # 현재 state가 obstacle의 위치가 아닌 경우
                        # 각 action에 따른 future return의 기댓값을 저장하는 행렬 action_sum을 초기화
                        action_sum = np.zeros(4) 
                        for action in range(len(ACTIONS)):  # 각 action에 대해 for loop 반복
                            # 현재 state, action이 주어졌을 때 next state, return을 얻는다
                            [next_height, next_width], r = env.interaction([height, width], ACTIONS[action])
                            # 현재 action에 해당하는 policy(a|s) * action value function 값을 action_sum 행렬에 저장
                            action_sum[action] = policy[height, width, action] * 1.0 *\
                                (r + (discount * old_state_values[next_height, next_width]))
                        # 새로운 state value function을 앞서 구한 특정 state에서의 future return 기댓값들의 합으로 구함
                        new_state_values[height, width] = np.sum(action_sum)
            # delta 값을 모든 state에서의 value function 변화량 중 가장 큰 값으로 설정
            delta = abs(old_state_values - new_state_values).max()
            if delta < theta:   # delta 값이 threshold 값보다 작을 경우 Policy Evaluation Loop 반복 종료
                break
            iteration += 1  # iteration flag 값은 1만큼 증가
            # 기존 state value function에 해당하는 old_state_values를 new_state_values 값들로 update
            old_state_values = new_state_values.copy()
        print(f"policy iteration #: {iter}, policy evaluation iteration #: {iteration}")
        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration





    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()
        # policy가 더 이상 업데이트되지 않는지를 판별하는 flag 변수인 policy_stable 값을 True로 초기화
        policy_stable = True
        # 모든 state에 대해 policy improvement 수행
        for height in range(HEIGHT):
            for width in range(WIDTH):
                # action에 따른 future return의 값을 저장하기 위한 행렬 action_value를 초기화
                action_value = np.zeros(4)
                # 현재 state에서 취할 수 있는 모든 action들에 대해 반복
                for action in range(len(ACTIONS)):
                    # 현재 state, action이 주어졌을 때 next state, return을 얻는다
                    [next_height, next_width], r = env.interaction([height, width], ACTIONS[action])
                    # 현재 state, action에서의 action value function 값을 action_value 행렬에 저장
                    action_value[action] = 1.0 * (r + discount*state_values[next_height, next_width])
                # action value function이 최대로 나타나는 action들을 찾아냄
                indices = np.argwhere(action_value == np.amax(action_value))
                # 현재 state에서 각 action에 대하여 greedy하게 improve된 policy를 선언해 준다
                for action in range(len(ACTIONS)):
                    # 현재 action이 argmax에 해당되는 액션인 경우
                    if action in indices:
                        # 확률값 policy(a|s)를 1/(argmax action들의 개수)로 부여한다
                        policy[height, width, action] = 1./len(indices)
                    # 현재 action이 argmax에 해당되는 액션이 아닌 경우
                    else: 
                        # 확률값 policy(a|s)를 0으로 부여한다
                        policy[height, width, action] = 0
        # 만약 policy의 개선이 있다면
        if np.all(old_policy == policy) == False: 
            # policy_stable flag값을 False로 설정
            policy_stable = False
        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        return policy, policy_stable

    def policy_iteration(self):
        iter = 1
        while (True):
            self.values, iteration = self.policy_evaluation(iter, env=self.env, policy=self.policy)
            self.policy, policy_stable = self.policy_improvement(iter, env=self.env, state_values=self.values,
                                                       old_policy=self.policy, discount=1.0)
            iter += 1
            if policy_stable == True:
                break
        np.savez('./result/dp.npz', V=self.values, PI=self.policy)
        return self.values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

