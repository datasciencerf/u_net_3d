from typing import Iterator, Sequence
import time
import descarteslabs.workflows as wf


def as_completed(jobs: Sequence[wf.Job], interval_sec: int = 5) -> Iterator[wf.Job]:
    """
    Iterator over Jobs that yields each Job when it completes.

    Parameters
    ----------
    jobs: Sequence[wf.Job]
        Jobs to wait for
    interval_sec: int, optional, default 5
        Wait at least this many seconds between polling for job updates.

    Yields
    ------
    job: wf.Job
        A completed job (either succeeded or failed).
    """
    jobs = list(jobs)
    while len(jobs) > 0:
        loop_start = time.perf_counter()

        i = 0
        while i < len(jobs):
            job = jobs[i]
            if not job.done:  # in case it's already loaded
                try:
                    job.refresh()
                except Exception as e:
                    print(e)
                    continue  # be resilient to transient errors for now

            if job.done:
                yield job
                del jobs[i]  # "advances" i
            else:
                i += 1

        loop_duration = time.perf_counter() - loop_start
        if len(jobs) > 0 and loop_duration < interval_sec:
            time.sleep(interval_sec - loop_duration)
