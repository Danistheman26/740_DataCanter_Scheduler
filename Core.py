from threading import Thread
import time

# a class the simulates a core. It has a job queue that simulates processing, it also has an output queue that
# lets the RL know when a job has finished running
class Core:
    def __init__(self, ID, input_q, output_q):
        self.busy = False
        self.ID = ID
        self.in_q = input_q
        self.out_q = output_q

        self.internal_thread = Thread(target=self.run, args=[])
        self.wrap_up = False

        self.metrics = [0, 0, 0, 0]   # jobs run, job runtime, job idle time, total_time

    # the cores main functionality is to sleep, so we run a separate thread for this
    def start_core(self):
        print(f"Started up core {self.ID}")
        self.internal_thread.start()

    def kill_core(self):
        self.wrap_up = True

    def run(self):
        start_time = time.time()

        while True:
            if not self.in_q.empty():
                # run simulated job (will just sleep)
                job = self.in_q.get()
                self.busy = True
                job.sim()
                self.busy = False

                # get job metrics after simulating (such as time to run, queue times)
                job_times = job.get_runtime()
                self.metrics[1] += job_times[0]
                self.metrics[2] += job_times[1] - job_times[0]
                self.out_q.put((job.ID2, job_times))

            elif self.wrap_up:
                self.metrics[3] += time.time() - start_time
                self.print_self()
                break

            else:
                time.sleep(0.01)

    def push_job(self, job):
        self.in_q.put(job)
        self.metrics[0] += 1

    def get_state(self):
        return [self.ID, self.in_q.qsize(), self.busy]

    def print_self(self):
        print(f"ID: {self.ID}   jobs_run: {self.metrics[0]}    run_time: {self.metrics[1]}    time jobs idle: "
              f"{self.metrics[2]}    overall_time: {self.metrics[3]}")
