#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <latch>

class ThreadPool {
public:
  static ThreadPool &instance();

  void parallel_for(size_t start, size_t end, std::function<void(size_t, size_t)> loop_body);
  size_t num_threads() const noexcept { return num_threads_; }

private:
  ThreadPool();
  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  void enqueue(std::function<void()> task);

  size_t num_threads_;
  std::vector<std::jthread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  bool stop_;
};

#endif // THREAD_POOL_HPP
