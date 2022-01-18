#include <deque>

template <class T>
class MinQueue {
public:
  const T& min() const { return queue[0].first; }

  void push(const T& value) {
    // moves the minimum value towards the head of the queue, counting the elements removed in order to "pop" work
    unsigned k = 1;
    while (!queue.empty() && value <= queue.back().first) {
      k += queue.back().second;
      queue.pop_back();
    }
    queue.emplace_back(value, k);
  }

  void pop() {
    if (queue[0].second == 1) {
      queue.pop_front();
    } else {
      --queue[0].second;
    }
  }

  void clear() { queue.clear(); }

private:
  std::deque<std::pair<T, unsigned>> queue;
};
