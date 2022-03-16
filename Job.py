import time
from utils import gaussian_rand


# a simulated task. This basically just records how long the task took from its creation to completion.
# After the job has been run, itd data will be used to train the RL
class Job:
    def __init__(self, ID, est_len):
        self.ID = ID
        self.est_len = est_len  # est_len is the estimated run time of the job
        self.actual_time = gaussian_rand(self.est_len)

        self.start_time = time.time()
        self.finish_time = 0

    # simulate operation by just sleeping
    def sim(self):
        time.sleep(self.actual_time)
        self.finish_time = time.time()

    def get_runtime(self):
        return self.actual_time, self.finish_time - self.start_time
