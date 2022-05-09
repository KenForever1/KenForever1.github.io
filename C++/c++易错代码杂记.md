```c++
std::vector<int> mi{1,2,2};
auto s = mi.size();
// vector 的size()函数返回的是unsigne Long型
for(auto i = s -1; i >= 0;i--){
    std::cout << i << std::endl;
}
```

```
2
1
0
18446744073709551615
18446744073709551614
18446744073709551613
18446744073709551612
...
```
