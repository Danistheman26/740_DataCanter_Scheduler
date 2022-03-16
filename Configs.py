# if training, don't start with RL, if testing, load the saved RL model
LOAD_RL = True
# can either use random scheduler, or RL based scheduler
USE_RL = True

NUM_CORES = 5               # number of simulated executors. Make sure to keep this value larger than num_job_creators
NUM_JOB_CREATORS = 2        # each job creator will create two jobs every second, each job averages 0.5 seconds
JOBS_PER_CREATOR = 100      # what name says, totoal jobs run will be this x num_jobs_creators

SIM_TIME = JOBS_PER_CREATOR * 1.5   # each job created is around 1 second, so leave room for error

MODEL_DIR = './MODEL/'
SPE = 20
