import numpy as np
from visualize_train import draw_value_image, draw_policy_image
# 시간 측정을 위해 time library를 import 했습니다. 
import time
# 경과 시간과 Q value의 평균값을 plot하기 위해 pyplot library를 import 했습니다. 
import matplotlib.pyplot as plt

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]

TRAINING_EPISODE_NUM = 800000

class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:   # Test
            mcc_results = np.load('./result/mcc.npz')
            self.V_values = mcc_results['V']
            self.Q_values = mcc_results['Q']
            self.policy = mcc_results['PI']
        else:          # For training
            self.V_values = np.zeros((HEIGHT, WIDTH))
            self.Q_values = np.zeros((HEIGHT, WIDTH, len(self.ACTIONS)))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)



    def initialize_episode(self):
        HEIGHT, WIDTH = self.env.size()
        while True:
            i = np.random.randint(HEIGHT)
            j = np.random.randint(WIDTH)
            state = [i, j]
            if (state in self.env.goal) or (state in self.env.obstacles):
                continue
            break
            # if (state not in self.env.goal) and (state not in self.env.obstacles):
            #     break
        return state



    def Monte_Carlo_Control(self, discount=1.0, alpha=0.01, max_seq_len=500,
                            epsilon=0.3, decay_period=20000, decay_rate=0.9):
        # env 객체의 size method를 통해 grid world의 크기를 불러온다
        HEIGHT, WIDTH = self.env.size()
        # 경과한 시간을 plot하기 위해 list와 경과시간의 평균을 저장하기 위한 변수 생성
        elapsed_time = []
        avg_time = 0.0
        # Q value의 평균값을 plot하기 위해 list변수 생성
        Q_avg = []
        # 현재 episode에 대한 계산 시작 시점을 저장하는 변수 생성
        start = time.time()
        for episode in range(TRAINING_EPISODE_NUM):
            state = self.initialize_episode()

            done = False
            timeout = False
            seq_len = 0
            history = []

            # 이전 time step에서 해당 state를 방문했는지를 기록하는 array를 생성
            visited = np.zeros((HEIGHT, WIDTH))

            # Sequence generation
            # done, timeout이라는 flag 값들을 이용하여 while문 반복을 제어
            while (done == False) and (timeout == False):
                # get_action 함수를 통해 policy에 따른 현재 state의 action을 불러옴
                action_index = self.get_action(state)
                # action_index에 해당하는 action을 action이란 변수로 지정
                action = ACTIONS[action_index]
                # env 객체의 interaction method를 이용하여 현재 state, action에 대한 next_state, reward를 구함
                next_state, reward = self.env.interaction(state, action)
                # history list에 (s, a, s', r)를 저장한다
                history.append((state, action_index, next_state, reward))
                # 현재 방문한 state를 visited array의 해당 state 자리에 count를 1만큼 늘려줌
                visited[state[0], state[1]] += 1

                # done이 true이면 agent가 terminal state에 도달했다는 것을 의미
                if (next_state in self.env.goal):
                    done = True
                
                # timeout이 true이면 generated sequence의 길이가 max_seq_len에 도달했다는 것을 의미
                if (seq_len == max_seq_len):
                    timeout = True
                # timeout이 발생하지 않았을 경우 
                else:
                    # seq_len을 1만큼 늘려주고
                    seq_len += 1
                    # 현재 state를 next_state값으로 update 함
                    state = next_state

            # timeout이 발생한 경우엔 생성한 episode를 policy update에 사용하지 않기 위해 조건문을 지정
            if timeout == False:

                # cum_reward는 현재 sequence에서 누적된 reward값을 의미
                cum_reward = 0
                # Q Value and policy update
                # sequence의 길이만큼 for문을 반복한다.
                for index in range(seq_len):
                    # history list에 저장되어 있는 (s, a, s', r) pair를 불러옴
                    ([i, j], a, _, r) = history[(seq_len-1) - index]
                    # 누적된 reward에 discount rate을 적용하여 현재 time step에서의 reward도 더해줌
                    cum_reward = discount * cum_reward + r
                    # first-visit MC를 적용
                    # 만약 현재 time step에서 해당 state를 방문한 게 처음이 아니면
                    if (visited[i, j] > 1):
                        # visited 행렬에 저장되어 있는 값에서 1만큼을 빼줌
                        visited[i, j] -= 1
                    # 만약 현재 time step에서 해당 state를 방문한 게 처음이라면
                    else: 
                        # 해당 state, action에 해당하는 action value function(Q)을 constant alpha MC로 update 해줌
                        self.Q_values[i][j][a] += alpha * (cum_reward - self.Q_values[i][j][a])
                        # epsilon-greedy하게 policy를 update하기 위해 Q값이 최대가 되게 하는 action index들을 구함
                        indices = np.argmax(self.Q_values[i][j])
                        # policy를 epsilon-greedy하게 update를 시켜주기 위해 각 action에 접근하도록 for loop을 돌림
                        for k, actions in enumerate(self.ACTIONS):
                            # 현재 action index가 greedy하게 찾은 action index에 해당된다면
                            if (k == indices):
                                # 현재 state, action의 policy를 epsilon-greedy하게 update 해줌
                                self.policy[i, j, k] = (((1 - epsilon)) + (epsilon / len(self.ACTIONS)))
                            # 현재 action index가 greedy하게 찾은 action index에 해당되지 않는다면
                            else:
                                # 현재 state, action의 policy를 epsilon-greedy하게 update 해줌
                                self.policy[i, j, k] = epsilon / len(self.ACTIONS)
                    # computation이 끝나는 시점의 시간을 기록
                    end = time.time()
                    # 20000번씩의 episode마다 경과 시간의 평균값을 구하기 위해 각각의 경과시간을 합함
                    avg_time += (end - start) * 1000 / 20000
                    # 현재 종료 시점의 시간을 다음 시작 시점의 시간으로 설정
                    start = end
                # 매번 decay_period의 배수만큼 episode에 대한 계산이 진행될 때마다
                if (episode % decay_period == 0) and (episode != 0):
                    # optimal policy로 수렴시키기 위해 epsilon을 decay_rate만큼 곱하여 감소
                    epsilon *= decay_rate
                    # 현재 episode가 끝난 시점에 대한 시간 값을 저장
                    end = time.time()
                    # decay_period의 배수만큼 반복될때마다 Q function의 평균값을 저장
                    Q_avg.append(np.average(self.Q_values))
                    # decay_period의 배수만큼 반복될때마다 현재 episode에 대한 연산에서 경과된 시간의 평균을 저장
                    elapsed_time.append(avg_time)
                    avg_time = 0  # 20000번째 episode마다 average time을 0으로 초기화

        self.V_values = np.max(self.Q_values, axis=2)
        draw_value_image(1, np.round(self.V_values, decimals=2), env=self.env)
        draw_policy_image(1, np.round(self.policy, decimals=2), env=self.env)
        np.savez('./result/mcc.npz', Q=self.Q_values, V=self.V_values, PI=self.policy)
        
        plt.figure()
        # 20000 episode마다 computation 시간을 plot
        plt.plot(elapsed_time)
        plt.show()
        # 20000 episode마다 Q value의 평균값을 plot
        plt.plot(Q_avg)
        plt.show()
        return self.Q_values, self.V_values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

