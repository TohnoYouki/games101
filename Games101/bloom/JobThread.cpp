#include "JobThread.hpp"

JobThread::JobThread(int num)
{
    control_payload.thread_num = num;
    for (int i = 0; i < num; i++) {
        threads.emplace_back(&JobThread::thread_job, this, i);
    }
}

JobThread::~JobThread() {
    control_payload.exit = true;
    control_payload.cv.notify_all();
    for (int i = 0; i < control_payload.thread_num; i++) {
        threads[i].join();
    }
}

void JobThread::wake_thread_job(const ThreadTaskPayload& payload)
{
    task_payload = &(payload);
    std::unique_lock<std::mutex> lock(control_payload.mutex);
    control_payload.wake_semaphore[control_payload.wake_semaphore_index] = true;
    control_payload.wake_semaphore[(control_payload.wake_semaphore_index + 1) % 2] = false;

    control_payload.cv.notify_all();
    control_payload.cv.wait(lock, [&] {
        return control_payload.finished_thread == control_payload.thread_num; });
    control_payload.finished_thread = 0;
    lock.unlock();
}

void JobThread::thread_job(int id)
{
    int wake_semaphore_index = 0;
    while (!control_payload.exit) {
        std::unique_lock<std::mutex> lock(control_payload.mutex);

        control_payload.cv.wait(lock, [&] {
            return control_payload.wake_semaphore[wake_semaphore_index]
                || control_payload.exit; });
        lock.unlock();
        if (control_payload.exit) { break; }

        task_payload->task(id, control_payload.thread_num, task_payload->payload);

        lock.lock();

        control_payload.finished_thread += 1;
        wake_semaphore_index = (wake_semaphore_index + 1) % 2;
        control_payload.wake_semaphore_index = wake_semaphore_index;
        control_payload.cv.notify_all();

        lock.unlock();
    }
}