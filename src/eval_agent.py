import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from climber_env import Climber

matplotlib.rcParams.update({'font.size': 16})


def plot_return(data):
    ax = plt.axes()
    x = np.linspace(0, len(data), len(data))
    ax.plot(x, data)
    plt.show()


def mov_avg(data):
    mean_score = []
    for i in range(len(data)):
        if i<100:
            mean_score.append(np.mean(data[0:i]))
        else:
            mean_score.append(np.mean(data[i-100:i]))

    # plot_return(mean_score)
    return mean_score


def evalute_multiple_runs(run1, run2, run3):

    num_eps = 330

    mean = np.zeros(num_eps)
    std = np.zeros(num_eps)
    e = np.arange(0, num_eps, 1)

    for i in range(num_eps):
        mean[i] = np.mean([run1[i], run2[i], run3[i]])
        std[i] = np.std([run1[i], run2[i], run3[i]])

    # plot it!
    ax = plt.axes()
    ax.plot(e, mean, lw=2, color='blue')
    ax.fill_between(e, mean + std, mean - std, facecolor='blue', alpha=0.3)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average score')
    ax.set_xlim(0, 330)
    ax.set_ylim(0, 7500)

    plt.show()


def evaluate_one_episode(episode_data):

    # state
    height = [h[0] for h in episode_data['States']]

    # actions
    left_hip_y = [a[0] for a in episode_data['Actions']]
    left_knee = [a[1] for a in episode_data['Actions']]
    left_shoulder1 = [a[2] for a in episode_data['Actions']]
    left_shoulder2 = [a[3] for a in episode_data['Actions']]
    left_elbow = [a[4] for a in episode_data['Actions']]
    # scale with moments
    left_hip_y = [a * 800 for a in left_hip_y]
    left_knee = [a * 500 for a in left_knee]
    left_shoulder1 = [a * 200 for a in left_shoulder1]
    left_shoulder2 = [a * 200 for a in left_shoulder2]
    left_elbow = [a * 400 for a in left_elbow]

    # plot it!
    frameskip = 5
    dt = 0.005
    x = np.linspace(0, len(height) * dt * frameskip, num=len(height))

    ax = plt.axes()
    ax.plot(x, height, lw=2, color='blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Ascended height (m)')

    plt.show()

    ax = plt.axes()
    ax.plot(x[:100], left_hip_y[:100], label="left hip", lw=2)
    ax.plot(x[:100], left_knee[:100], label="left knee", lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint moments (Nm)')
    plt.legend(loc='upper left')
    plt.show()

    ax = plt.axes()
    ax.plot(x[:100], left_shoulder1[:100], label="left shoulder 1", lw=2)
    ax.plot(x[:100], left_shoulder2[:100], label="left shoulder 2", lw=2)
    ax.plot(x[:100], left_elbow[:100], label="left elbow", lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint moments (Nm)')
    plt.legend(loc='upper left')
    plt.show()


def run_episode(env, agent, rendering=True):
    state = env.reset()
    done = False
    episode_reward = 0

    episode_data = {
        'States': [],
        'Actions': [],
        'Rewards': [],
        'Cumulated_reward': 0
    }

    i = 0
    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)  # Step
        episode_reward += reward

        episode_data['States'].append(state)
        episode_data['Actions'].append(action)
        episode_data['Rewards'].append(reward)

        if rendering is True:
            env.render()
        # print(state[0], action, reward)
        state = next_state
        i = i + 1
        print(i)
    episode_data['Cumulated_reward'] = episode_reward

    return episode_data


def main():

    RENDERING = True

    training_data = pickle.load(open('ClimbingAgent_0606_22-56-50_good.p', 'rb'))
    run1 = mov_avg(pickle.load(open('ClimbingAgent_0602_06-04-48_good.p', 'rb'))[4])
    run2 = mov_avg(pickle.load(open('ClimbingAgent_0603_18-51-10_good.p', 'rb'))[4])
    run3 = mov_avg(pickle.load(open('ClimbingAgent_0606_22-56-50_good.p', 'rb'))[4])
    env = Climber()
    agent = training_data[0]

    num_episodes = 1
    episode_scores = []

    for e in range(num_episodes):

        ep_data = run_episode(env, agent, RENDERING)
        episode_scores.append(ep_data['Cumulated_reward'])

    evalute_multiple_runs(run1, run2, run3)
    evaluate_one_episode(ep_data)


if __name__ == '__main__':
    main()