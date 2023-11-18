## Trait
Trait可以实现Rust中的动态分发和静态分发。

动态分发：
在运行期确定调用类型。通过Trait实现，比如：Box<dyn SomeTrait>或者dyn &SomeTrait。

静态分发：
通过定义泛型，在编译器确定调用类型。在Rust中，静态分发采用单态化（monomorphization），会针对不同类型的调用者，在编译时生成不同版本的函数，所以范型也被称为[类型参数](https://bluejekyll.github.io/blog/posts/type-parameters/)。好处是没有虚函数调用的开销，缺点是最终的二进制文件膨胀。

参考学习：
[https://rustmagazine.github.io/rust_magazine_2021/chapter_4/ant_trait.html](https://rustmagazine.github.io/rust_magazine_2021/chapter_4/ant_trait.html)

## Cow
Cow Trait实现了写时复制，如果它指向的内容发生了改变时，会复制出一个新的对象，而不改变原来指向的对象。内容需要发生改变时，会调用to_mut() 方法，然后对复制的对象进行修改，比如对一个vec push内容。
例如：
```rust
#![allow(unused)]
fn main() {
    use std::borrow::Cow;

    let s = "foo".to_string();

    let mut cow = Cow::Borrowed(&s);
    cow.to_mut().make_ascii_uppercase();

    println!("s is : {}",s);

    println!("cow is : {}",cow);

    assert_eq!(
        cow,
        Cow::Owned(String::from("FOO")) as Cow<'_, str>
    );
}
```
运行结果:
```rust
s is : foo
cow is : FOO
```
可以看到，cow的内容从Borrowed，变成了Owned，发生了写实复制。如果没有变化，就不会发生复制。
例如：
```rust
#![allow(unused)]
fn main() {
use std::borrow::Cow;

fn abs_all(input: &mut Cow<'_, [i32]>) {
    for i in 0..input.len() {
        let v = input[i];
        if v < 0 {
            // Clones into a vector if not already owned.
            input.to_mut()[i] = -v;
        }
    }
    println!("input is : {:?}", input);
}

// No clone occurs because `input` doesn't need to be mutated.
let slice = [0, 1, 2];
let mut input = Cow::from(&slice[..]);
abs_all(&mut input);

println!("slice 1 is : {:?}", slice);

// Clone occurs because `input` needs to be mutated.
let slice = [-1, 0, 1];
let mut input = Cow::from(&slice[..]);
abs_all(&mut input);

println!("slice 2 is : {:?}", slice);

// No clone occurs because `input` is already owned.
let mut input = Cow::from(vec![-1, 0, 1]);
abs_all(&mut input);

println!("input 3 is : {:?}", input);
}
```
运行结果：
```rust
input is : [0, 1, 2]
slice 1 is : [0, 1, 2]
input is : [1, 0, 1]
slice 2 is : [-1, 0, 1]
input is : [1, 0, 1]
input 3 is : [1, 0, 1]
```
例子来源于：官方对Cow的介绍，[https://doc.rust-lang.org/std/borrow/enum.Cow.html](https://doc.rust-lang.org/std/borrow/enum.Cow.html)。
运行代码可以在Play上，[https://play.rust-lang.org](https://play.rust-lang.org/)。

Rust中几种指针的区别：

| 类型 | 区别 |
| --- | --- |
| Box | 唯一所有权的智能指针，相当于C++里面的UniquePtr |
| Rc和Arc | 共享所有权的智能指针，相当于C++里面的SharedPtr，Arc可以A可以理解成Atomic，可以在线程之间安全传递。 |
| Cow | 实现写实复制的智能指针 |

## FromStr Trait
一个类型如果实现了FromStr Trait，就可以从string字符串转换成该类型， from_str 方法规定了如何从string转换成该类型。FromStr既可以显示调用（直接调用from_str方法），也可以隐式调用，即使用str的parse方法，例如：
```rust
#![allow(unused)]
fn main() {
use std::str::FromStr;

#[derive(Debug, PartialEq)]
struct Point {
    x: i32,
    y: i32
}

#[derive(Debug, PartialEq, Eq)]
struct ParsePointError;

impl FromStr for Point {
    type Err = ParsePointError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (x, y) = s
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .and_then(|s| s.split_once(|c| {c== ',' || c == ' '} ))
            .ok_or(ParsePointError)?;

        let x_fromstr = x.parse::<i32>().map_err(|_| ParsePointError)?;
        let y_fromstr = y.parse::<i32>().map_err(|_| ParsePointError)?;

        Ok(Point { x: x_fromstr, y: y_fromstr })
    }
}

let expected = Ok(Point { x: 1, y: 2 });
// Explicit call
assert_eq!(Point::from_str("(1,2)"), expected);
// Implicit calls, through parse
assert_eq!("(1,2)".parse(), expected);
assert_eq!("(1,2)".parse::<Point>(), expected);
// Invalid input string
// assert!(Point::from_str("(1 2)").is_err());

assert_eq!("(1 2)".parse::<Point>(), expected);
}
```
例子来源于官方，从这个例子我们还可以学习到str模块提供的split_once方法以及Trait std::str::pattern::Pattern 的使用。
官方例子中只支持逗号分割，即
```rust
.and_then(|s| s.split_once(','))
```
如何同时支持空格分隔，查看split_once api，发现可以传入Pattern，而pattern不只包括str，也包括传入Fn。[https://doc.rust-lang.org/std/str/pattern/trait.Pattern.html](https://doc.rust-lang.org/std/str/pattern/trait.Pattern.html)。
因此可以实现如下：
```rust
.and_then(|s| s.split_once(|c| {c== ',' || c == ' '} ))
```
传入了一个lamda表达式，返回bool类型。

## The Rust performance book
[https://nnethercote.github.io/perf-book/linting.html#disallowing-types](https://nnethercote.github.io/perf-book/linting.html#disallowing-types)

fast高斯模糊实现
[https://github.com/fschutt/fastblur](https://github.com/fschutt/fastblur)

## Pueue
pueue创建Task，一个Task可以理解为一个Shell命令。
比如：`ls -alh`.
在程序中，如果要执行这个命令，最简单的是启动一个进程来执行。如果要执行的是一个shell命令，完整的执行命令在不同的平台和不同的shell里面都不一样。
比如在unix平台上：

```
sh -c 'ls -alh'
```
pueue为了提供扩展兼容不同的shell，比如zsh、bash、fish，不同平台unix、windows定义了如下的模版。模版中存在变量{{ pueue_command_string }}。

```
/// Unix default:
    /// `vec!["sh", "-c", "{{ pueue_command_string }}"]`.
    ///
    /// Windows default:
    /// `vec!["powershell", "-c", "[Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8; {{ pueue_command_string }}"]`
```
当用户输入执行的命令`ls -alh`，希望实现输入的命令去替换{{ pueue_command_string }}。
这个功能类似给一个字符串模版，然后通过替换对模版实例化。pueue通过handlebars crate来实现这个功能，[https://docs.rs/handlebars/latest/handlebars/](https://docs.rs/handlebars/latest/handlebars/)可以参考官网例子。
handlebars是一个模版语言，可以将模版转换成text或者html。
[https://handlebarsjs.com/guide/](https://handlebarsjs.com/guide/)

```
/// Take a platform specific shell command and insert the actual task command via templating.
pub fn compile_shell_command(settings: &Settings, command: &str) -> Command {
    let shell_command = get_shell_command(settings);

    let mut handlebars = handlebars::Handlebars::new();
    handlebars.set_strict_mode(true);
    handlebars.register_escape_fn(handlebars::no_escape);

    // Make the command available to the template engine.
    let mut parameters = HashMap::new();
    parameters.insert("pueue_command_string", command);

    // We allow users to provide their own shell command.
    // They should use the `{{ pueue_command_string }}` placeholder.
    let mut compiled_command = Vec::new();
    for part in shell_command {
        let compiled_part = handlebars
            .render_template(&part, &parameters)
            .unwrap_or_else(|_| {
                panic!("Failed to render shell command for template: {part} and parameters: {parameters:?}")
            });

        compiled_command.push(compiled_part);
    }

    let executable = compiled_command.remove(0);

    // Chain two `powershell` commands, one that sets the output encoding to utf8 and then the user provided one.
    let mut command = Command::new(executable);
    for arg in compiled_command {
        command.arg(&arg);
    }

    // Inject custom environment variables.
    if !settings.daemon.env_vars.is_empty() {
        log::info!(
            "Inject environment variables: {:?}",
            &settings.daemon.env_vars
        );
        command.envs(&settings.daemon.env_vars);
    }

    command
}
```
[https://github1s.com/Nukesor/pueue/blob/HEAD/pueue_lib/src/process_helper/mod.rs#L71](https://github1s.com/Nukesor/pueue/blob/HEAD/pueue_lib/src/process_helper/mod.rs#L71)
通过模版替换后的命令字符串，构造了std::process::Command，就可以通过创建一个进程执行命令了。

```
// Spawn the actual subprocess
        let spawned_command = command
            .current_dir(path)
            .stdin(Stdio::piped())
            .env_clear()
            .envs(envs.clone())
            .stdout(Stdio::from(stdout_log))
            .stderr(Stdio::from(stderr_log))
            .group_spawn();
```
[https://github1s.com/Nukesor/pueue/blob/HEAD/pueue/src/daemon/task_handler/spawn_task.rs](https://github1s.com/Nukesor/pueue/blob/HEAD/pueue/src/daemon/task_handler/spawn_task.rs)