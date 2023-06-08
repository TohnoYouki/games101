#pragma once
#include <mutex>
#include <thread>
#include <vector>

class JobThread
{
public:
    typedef void (*ThreadTask) (int, int, void*);

    struct ThreadTaskPayload {
        ThreadTask task;
        void* payload;
    };
    struct ThreadControlPayload {
        int thread_num;
        bool exit = false;
        std::mutex mutex;

        int wake_semaphore_index = 0;
        bool wake_semaphore[2] = { false, false };
        int finished_thread = 0;
        std::condition_variable cv;
    };

    JobThread(int num);
    ~JobThread();

    void wake_thread_job(const ThreadTaskPayload& payload);
    void thread_job(int id);

private:
    std::vector<std::thread> threads;
    ThreadControlPayload control_payload;
    const ThreadTaskPayload* task_payload;
};