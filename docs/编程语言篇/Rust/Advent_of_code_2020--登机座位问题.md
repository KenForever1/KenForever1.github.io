---
title: Advent of code 2020 -- 登机座位问题
date: 2021-11-06 22:55:17
tags:
---

[Advent of code 2020 -- Binary Boarding](https://adventofcode.com/2020/day/5)

此问题，讲的是用10个字节的字符表示登机座位，飞机座位有128行、8列，如"FBFBBFFRLR"表示位置在第44行、第5列的地方。

+ B和F用于表示行数;
F:表示前一半，如0-127,那么第一个F表示位置在0-63范围;
B:表示后一半,如0-63，那么第2个B表示位置在32-63范围;
...
以此推出，行数为44

+ R和L用于表示列数;
R:表示后一半;
L:表示前一半;
...
以此推出，列数为5

### 定义位置的结构体

因为行数和列数都不超过255,那么行数和列数都可以用8bit的整数表示，在Rust中为u8。
```Rust
#[derive(Default, Debug, PartialEq)]
struct Seat {
    row: u8,
    col: u8,
}
```

### 解析位置
我们知道127的二进制表示为7个1;

```
127

0111 1111
```

那么可以前7个字符的F和B可以按照顺序表示7个bit位上的0或者1, F表示0, B表示1;

```
FBFBBFF
0101100
代表44
```

同样，可以用后3个字符的R和L可以按照顺序表示3个bit位上的0或者1, L表示0, R表示1;

```
RLR
101
代表5
```

在Rust中可以使用[bitvec](https://lib.rs/crates/bitvec)库很容易进行bit操作。
可以用命令进行导入

```
cargo add bitvec
```

定义位置解析方法：

```Rust
use bitvec::prelude::*;

impl Seat {
    const ROW_BITS: usize = 7;
    const COL_BITS: usize = 3;

    fn parse(input: &str) -> Self {
        let bytes = input.as_bytes();

        let mut res: Seat = Default::default();
        {
            let row = BitSlice::<Msb0, _>::from_element_mut(&mut res.row);

            for (i, &b) in bytes[0..Self::ROW_BITS].iter().enumerate() {
                row.set(
                    (8 - Self::ROW_BITS) + i,
                    match b {
                        b'F' => false,
                        b'B' => true,
                        _ => panic!("unexpected row letter : {}", b as char),
                    },
                );
            }
        }

        {
            let col = BitSlice::<Msb0, _>::from_element_mut(&mut res.col);
            for (i, &b) in bytes[Self::ROW_BITS..][..Self::COL_BITS].iter().enumerate() {
                col.set(
                    (8 - Self::COL_BITS) + i,
                    match b {
                        b'L' => false,
                        b'R' => true,
                        _ => panic!("unexpected col letter : {}", b as char),
                    },
                );
            }
        }

        res
    }

}

```

测试位置解析方法：
```Rust

#[test]
fn test_parse() {
    let input = "FBFBBFFRLR";
    let seat = Seat::parse(input);

    assert_eq!(seat, Seat { row: 44, col: 5 });
}
```

### 通过位置得出位置编号

我们定义了位置的数据解构Seat，位置由row和col表示。

位置编号可以表示为：
```
id = row * 8 + col
```

位置编号计算函数:

```Rust
impl Seat {
    const ROW_BITS: usize = 7;
    const COL_BITS: usize = 3;


    fn id(&self) -> u64 {
        // 通过移位操作代替行数乘以8
        ((self.row as u64) << Self::COL_BITS) + (self.col as u64)
    }
}

```

测试：
```Rust
#[test]
fn test_seat_id() {
    macro_rules! validate {
        ($input: expr, $row: expr, $col: expr, $id: expr) => {
            let seat = Seat::parse($input);

            assert_eq!(
                seat,
                Seat {
                    row: $row,
                    col: $col
                }
            );

            assert_eq!(seat.id(), $id);
        };
    }

    validate!("BFFFBBFRRR", 70, 7, 567);
    validate!("FFFBBBFRRR", 14, 7, 119);
    validate!("BBFFBBFRLL", 102, 4, 820);
}

```

### 问题1 ： 计算测试集中最大位置编号

```Rust
fn main() {

    let max_id = itertools::max(
        include_str!("input.txt")
            .lines()
            .map(Seat::parse)
            .map(|seat| seat.id()),
    );

    println!("The maximum seat ID is {:?}", max_id);

}
```

### 问题2 ： 找出丢失的位置编号

last_id,即不连续的Id，如7和9之间空缺了一个8

找出last_id，则要计算出所有的位置编号，然后进行排序，我们将所有的位置编号用Vec存储，那么要实现sort()方法，则要为Seat数据结构实现PartialOrd, Ord Trait.

更改Seat数据结构
```Rust
#[derive(Clone, Copy,Default, Debug, PartialEq, PartialOrd, Eq, Ord)]
struct Seat {
    row: u8,
    col: u8,
}
```

```Rust
fn main() {
    let mut ids: Vec<_> = include_str!("input.txt").lines().map(Seat::parse).collect();

    ids.sort();

    let mut last_id: Option<Seat> = None;

    for id in ids {
        if let Some(last_id) = last_id {
            let gap = id.id() - last_id.id();
            if gap > 1{
                println!("Our seat ID is {}", last_id.id());
                return;
            }
        }
        last_id = Some(id);
    }
}
```

参考学习：https://fasterthanli.me/series/advent-of-code-2020/part-5
