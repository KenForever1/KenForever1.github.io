 ```c++
 #include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <set>
#include <functional>

using namespace std;


class ExamRoom {
public:
    using CmpFunctionType = bool (*)(pair<int, int>, pair<int, int>);
    unordered_map<int, pair<int, int>> leftIntervalMap;
    unordered_map<int, pair<int, int>> rightIntervalMap;

    set<pair<int, int>> pq;
    int N;

private:


    void addInterval(pair<int, int> interval) {
        pq.insert(interval);
        leftIntervalMap[interval.first] = interval;
        rightIntervalMap[interval.second] = interval;
    }

    void removeInterval(pair<int, int> interval) {
        pq.erase(interval);
        leftIntervalMap.erase(interval.first);
        rightIntervalMap.erase(interval.second);
    }

    int distance(pair<int, int> interval) {
        int x = interval.first;
        int y = interval.second;
        if (x == -1) {
            return y;
        }
        if (y == N) {
            return N - x - 1;
        }
        return (y - x) / 2;
    }

public:
    ExamRoom(int n) {
        this->N = n;
        auto cmp = [this](pair<int, int> a, pair<int, int> b) {
            auto distA = distance(a);
            auto distB = distance(b);
            if (distA == distB) {
                return a.first > b.second;
            }

            return distA < distB;
            return true;
        };
        pq = set<pair<int, int>, decltype(cmp)>(cmp);

        auto mPair = std::make_pair(-1, n);
        addInterval(mPair);
    }

    int seat() {
        auto longest = *pq.rbegin();
        int x = longest.first;
        int y = longest.second;

        int seat = 0;
        if (x == -1) {
            seat = 0;
        } else if (y == N) {
            seat = N - 1;
        } else {
            seat = (y - x) / 2 + x;
        }

        auto left = std::make_pair(x, seat);
        auto right = std::make_pair(seat, y);
        removeInterval(longest);
        addInterval(left);
        addInterval(right);
        return seat;
    }

    void leave(int p) {
        auto left = leftIntervalMap[p];
        auto right = rightIntervalMap[p];

        auto merged = std::make_pair(left.first, right.second);
        removeInterval(left);
        removeInterval(right);
        addInterval(merged);
    }
};

/**
 * Your ExamRoom object will be instantiated and called as such:
 * ExamRoom* obj = new ExamRoom(n);
 * int param_1 = obj->seat();
 * obj->leave(p);
 */
int main() {
    printf("hello world \n");

}
 ```
