---
comments: true
---

本文以在linux上编译windows上可执行rust tui example程序为例，记录rust跨平台编译过程。
[tui-rs](https://github.com/fdehau/tui-rs)是一个Rust库，用于构建丰富的终端用户界面和仪表盘。

首先，git clone tui-rs的源码，编译并运行linux可执行程序命令如下：

```
git clone https://github.com/fdehau/tui-rs.git
```

```
cargo run --example user_input --release
```

使用如下命令，可以通过--target选项指定编译目标的平台。

```
cargo build --example user_input --release --target x86_64-pc-windows-gnu
```

但是，初次编译，会报错...

```
 cargo build --example user_input --release
--target x86_64-pc-windows-gnu
  Downloaded windows_x86_64_gnu v0.32.0
  Downloaded ntapi v0.3.7
  Downloaded miow v0.3.7
  Downloaded crossterm_winapi v0.9.0
  Downloaded winapi v0.3.9
  Downloaded windows-sys v0.32.0
  Downloaded winapi-x86_64-pc-windows-gnu v0.4.0
  Downloaded 7 crates (8.4 MB) in 16.18s (largest was `windows-sys` at 3.4 MB)
   Compiling cfg-if v1.0.0
   Compiling windows_x86_64_gnu v0.32.0
   Compiling winapi-x86_64-pc-windows-gnu v0.4.0
   Compiling winapi v0.3.9
   Compiling scopeguard v1.1.0
   Compiling smallvec v1.8.0
   Compiling bitflags v1.3.2
   Compiling ppv-lite86 v0.2.16
error[E0463]: can't find crate for `core`
  |
  = note: the `x86_64-pc-windows-gnu` target may not be installed
  = help: consider downloading the target with `rustup target add x86_64-pc-windows-gnu`

For more information about this error, try `rustc --explain E0463`.
error: could not compile `cfg-if` due to previous error
warning: build failed, waiting for other jobs to finish...
error: build failed
```

提示我们安装the `x86_64-pc-windows-gnu` target, 执行如下命令：

```
rustup target add x86_64-pc-windows-gnu
rustup toolchain install stable-x86_64-pc-windows-gnu
```

然后再次编译，遇到如下问题:

```
$ cargo build --example user_input --release --target x86_64-pc-windows-gnu
   Compiling tui v0.17.0 (/home/xxx/workspace/rust-learn/tui-rs)
error: linker `x86_64-w64-mingw32-gcc` not found
  |
  = note: No such file or directory (os error 2)

error: could not compile `tui` due to previous error
```

安装mingw-64:

```
sudo apt update && sudo apt install mingw-w64
```

再次编译，编译成功：

```
 cargo build --example user_input --release --target x86_64-pc-windows-gnu
   Compiling tui v0.17.0 (/home/xxx/workspace/rust-learn/tui-rs)
    Finished release [optimized] target(s) in 1.91s

```

可以看见project_directionary/target/下多了一个 "x86_64-pc-windows-gnu"目录，目录中可以找到编译出的windows平台可执行程序。
