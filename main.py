import Core
import Job
import queue
import random
from utils import *
from Configs import *
from threading import Thread
from Agent import Agent
import time

new_job_queue = queue.Queue()
rewards_queue = queue.Queue()
job_stats = 3

# History maintained for RL to learn from every epoch
job_counter = 0
rew_counter = 0
Rewards = {}
History = {}

# RL object
my_agent = Agent(NUM_CORES * 2, NUM_CORES, SPE, MODEL_DIR, from_saved=LOAD_RL)


# a function to create job DAGS (right now just single node jobs)
def dag_builder(ID):
    return Job.Job(ID, 0.5)


# a function to simulate incoming jobs places one job a second with unit normal gaussian randomness)
def input_simulator(num_jobs, job_ID_start):
    incr = job_ID_start
    time.sleep(np.random.rand(1)[0])
    for job in range(num_jobs):
        new_job = dag_builder(incr)
        new_job_queue.put(new_job)

        sleep_time = gaussian_rand(0.5)
        time.sleep(sleep_time)


def reward_function(job_stats):
    return job_stats[0] - job_stats[1]


def reward_collector():
    global Rewards
    while True:
        if not rewards_queue.empty():
            a_rew = rewards_queue.get()
            Rewards[a_rew[0]] = (a_rew[1])
        else:
            time.sleep(0.25)


# Thread to process rewards and train model after jobs are completed every epoch
def rewarder():
    global rew_counter, my_agent, Rewards, History
    episode_length = 0
    episode_return = 0

    while True:
        if rew_counter in Rewards:
            # get history to process
            hist = History.pop(rew_counter)
            job_stats = Rewards.pop(rew_counter)
            obs = hist[2].reshape(1, -1)
            logits = hist[0]
            action = hist[1]

            # store observations internally to PPO agent
            reward = reward_function(job_stats)
            episode_return += reward
            episode_length += 1
            my_agent.store_step(obs, logits, action, reward)

            # after each epoch, train the RL models
            if episode_length == SPE:
                print(episode_return)
                last_value = 0
                my_agent.critic(obs.reshape(1, -1))
                my_agent.buffer.finish_trajectory(last_value)
                my_agent.train_self()
                episode_length = 0
                episode_return = 0

            rew_counter += 1
        else:
            time.sleep(0.25)


if __name__ == '__main__':
    # initialize cores here (executors in paper)
    cores = []
    for ID in range(NUM_CORES):
        c = Core.Core(ID, queue.Queue(), rewards_queue)
        cores.append(c)
        c.start_core()

    # start up threads to receive rewards when jobs are finished
    Thread(target=reward_collector, args=[]).start()
    Thread(target=rewarder, args=[]).start()

    # start up job creators
    counter = 0
    for job_maker in range(NUM_JOB_CREATORS):
        Thread(target=input_simulator, args=[JOBS_PER_CREATOR, counter]).start()
        counter += JOBS_PER_CREATOR

    end_time = time.time() + 10
    while True:
        if not new_job_queue.empty():
            end_time = time.time() + 10
            a_job = new_job_queue.get()
            a_job.ID2 = job_counter

            core_obs = []
            for core in cores:
                core_obs.append(core.in_q.qsize())
                if core.busy:
                    core_obs.append(1)
                else:
                    core_obs.append(0)

            # from session information (state) determine action
            observation = np.array(core_obs, dtype=np.float32).reshape(1, -1)
            logits, act = my_agent.sample_action(observation)
            core_index = act[0].numpy()
            History[job_counter] = (logits, act, observation)
            job_counter += 1

            index = random.randint(0, NUM_CORES - 1)

            if USE_RL:
                cores[core_index].push_job(a_job)
            else:
                cores[index].push_job(a_job)
        elif time.time() > end_time:
            break
        else:
            time.sleep(0.05)

    for core in cores:
        time.sleep(0.1)
        core.kill_core()
    my_agent.save()
