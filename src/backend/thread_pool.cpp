#include "backend/thread_pool.hpp"
#include <algorithm>

ThreadPool &ThreadPool::instance() {
  static ThreadPool pool;
  return pool;
}

ThreadPool::ThreadPool()
    : num_threads_(std::max(1u, std::thread::hardware_concurrency())), stop_(false) {
  for (size_t i = 0; i < num_threads_; ++i) {
    workers_.emplace_back([this](std::stop_token st) {
      while (!st.stop_requested()) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          cv_.wait(lock, [this, &st]() {
            return stop_ || !tasks_.empty() || st.stop_requested();
          });
          if (stop_ && tasks_.empty()) {
            return;
          }
          if (tasks_.empty()) {
            if (st.stop_requested()) {
              return;
            }
            continue;
          }
          task = std::move(tasks_.front());
          tasks_.pop();
        }
        task();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto &worker : workers_) {
    worker.request_stop();
  }
  workers_.clear();
}

void ThreadPool::enqueue(std::function<void()> task) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    tasks_.push(std::move(task));
  }
  cv_.notify_one();
}

void ThreadPool::parallel_for(size_t start, size_t end,
                              std::function<void(size_t, size_t)> loop_body) {
  size_t total_elements = end - start;
  if (total_elements == 0) {
    return;
  }

  size_t num_tasks = std::min(static_cast<size_t>(num_threads_), total_elements);
  size_t chunk_size = total_elements / num_tasks;
  size_t remainder = total_elements % num_tasks;

  std::latch sync_latch(static_cast<ptrdiff_t>(num_tasks));

  for (size_t t = 0; t < num_tasks; ++t) {
    size_t t_start = start + t * chunk_size + std::min(t, remainder);
    size_t t_end = t_start + chunk_size + (t < remainder ? 1 : 0);

    enqueue([t_start, t_end, &loop_body, &sync_latch]() {
      loop_body(t_start, t_end);
      sync_latch.count_down();
    });
  }

  sync_latch.wait();
}
