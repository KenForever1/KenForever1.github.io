```c++
#include <iostream>
#include <list>
#include <utility>
#include <vector>
#include <unordered_map>
#include <memory>

using namespace std;

class LRUCache {

public:
    unordered_map<int, list<pair<int, int>>::iterator> keyToVal{};
    list<pair<int, int>> Cache{};
    int cap;
public:
    explicit LRUCache(int _cap) : cap(_cap) {

    }

    int get(int key) {
        if (keyToVal.find(key) == keyToVal.end()) {
            return -1;
        }
        auto val = keyToVal.at(key)->second;
        makeRecently(key);
        return val;
    }


    void put(int key, int val) {
        if (keyToVal.find(key) != keyToVal.end()) {
            deleteKey(key);
            addRecently(key, val);
            return;
        }

        if (keyToVal.size() >= cap) {
            removeLeastUsage();
        }

        addRecently(key, val);
    }

private:

    void makeRecently(int key) {
        auto node = *keyToVal.at(key);
        Cache.erase(keyToVal.at(key));
        Cache.push_front(node);
        keyToVal[key] = Cache.begin();
    }

    void addRecently(int key, int val) {
        auto node = std::make_pair(key, val);
        Cache.push_front(node);
        keyToVal[key] = Cache.begin();
    }

    void removeLeastUsage() {
        auto key = Cache.back().first;
        keyToVal.erase(key);
        Cache.pop_back();
    }

    void deleteKey(int key) {
        auto node = keyToVal.at(key);
        Cache.erase(node);
        keyToVal.erase(key);
    }
};


int main() {
    std::cout << "Hello, World!" << std::endl;
    LRUCache lRUCache(2);
    lRUCache.put(1, 1); // 缓存是 {1=1}
    lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
    auto res = lRUCache.get(1);    // 返回 1
    lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
    res = lRUCache.get(2);    // 返回 -1 (未找到)
    lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
    res = lRUCache.get(1);    // 返回 -1 (未找到)
    res = lRUCache.get(3);    // 返回 3
    res = lRUCache.get(4);    // 返回 4
    return 0;
}

```
