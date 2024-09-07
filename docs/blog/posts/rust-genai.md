---
title: rust如何优雅的写错误处理
date: 2024-09-06
authors: [KenForever1]
categories: 
  - rust
labels: []
---
## rust如何优雅的写错误处理
<!-- more -->
定义error.rs
```rust
use derive_more::From;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, From)]
pub enum Error {
    
    #[from]
	ParamsErr(String),

}

// region:    --- Error Boilerplate

impl core::fmt::Display for Error {
	fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
		write!(fmt, "{self:?}")
	}
}

impl std::error::Error for Error {}

// endregion: --- Error Boilerplate
```

main.rs
```rust
mod error;
use error::{Error, Result};
fn main() -> Result<()>{
    Ok(())
}
```

Cargo.toml
```toml
[dependencies]
derive_more = { version = "1.0.0-beta", features = ["from", "display"] }
```

## 参考
+ [rust-genai](https://github.com/jeremychone/rust-genai)
+ [rust错误处理最佳实践](https://rust10x.com/best-practices/error-handling)