---
title: 学习OpenAI-API实现的相关库
date: 2024-10-24
authors: [KenForever1]
categories: 
  - openai
labels: []
comments: true
---

在平时工作中或者平时折腾中，你如果部署或者调用过大模型，包括语言大模型LLM、视觉大模型LVM等。那么，你肯定对OpenAI api特别熟悉了。今天一起再看一下OpenAI的api文档，python api当然是最常用的。看了一些推荐的其它语言的API。比如:

+ cpp实现：[D7EAD/liboai](https://github.com/D7EAD/liboai)

+ rust实现：[64bit/async-openai](https://github.com/64bit/async-openai)

你可以学习到：

+ 如何用C++、rust封装openai api

+ 如何实现一个基于大模型的音乐搜索app

+ 如何自己通过curl库封装一个不错的网络库

简直太美妙了！！！  

<!-- more -->
## cpp liboai

### 对话api的使用以及conversation实现
简单看一下用法, 采用c++ 17实现的，写法和注释都比较清晰。用户设置的system、user数据都通过Addxxx API，保存到Conversation对象中。模型返回的assitant信息也通过Update API加入到其中。这样Conversation对象中就保存了整个对话上下文信息。
每次对话的时候将上下文请求模型，模型就可以完成对话了。


```c++
#include "liboai.h"

using namespace liboai;

int main() {
  OpenAI oai;

  // create a conversation
  Conversation convo;

  // add a message to the conversation
  convo.AddUserData("What is the point of taxes?");

  if (oai.auth.SetKeyEnv("OPENAI_API_KEY")) {
    try {
      Response response = oai.ChatCompletion->create(
        "gpt-3.5-turbo", convo
      );

      // update our conversation with the response
      convo.Update(response);  

      // print the response
      std::cout << convo.GetLastResponse() << std::endl;
    }
    catch (std::exception& e) {
      std::cout << e.what() << std::endl;
    }
  }
}
```
在Conversation中，通过nlohmann::json（一个c++的json库，重载实现了很多函数，可以很方便的用[]去操作json元素，和python很类似）信息去保存聊天对话的上下文信息和function。
```c++
nlohmann::json _conversation;
std::optional<nlohmann::json> _functions = std::nullopt;
```

### 网络库

liboai有个很有趣的地方，他的网络库是自己通过curl库封装的。因为前几天接触一个开发板，只有curl库，也需要自己实现http请求等，今天刚好看到这个很奇妙。

它封装了GET、POST、DELETE请求，异步通过std::async实现，起一个线程单独跑。比如GET。
源码文件：liboai\include\core\netimpl.h
里面有个Session类哈，一个请求对应一个Session，调用curl的api设置header，设置参数，获取返回值等。代码很漂亮。
```c++
#include <curl/curl.h>
class Session final : private CurlHolder {
public:
    Session() = default;
    ~Session() override;

    liboai::Response Get();
    liboai::Response Post();
    liboai::Response Delete();
    liboai::Response Download(std::ofstream& file);
    void ClearContext();
    ...    
}
```
Get的实现，
```c++
liboai::Response liboai::netimpl::Session::Get() {
	#if defined(LIBOAI_DEBUG)
		_liboai_dbg(
			"[dbg] [@%s] Called PrepareGet().\n",
			__func__
		);
	#endif

	this->PrepareGet();
	
	this->Perform();
		
	return Complete();
}

void liboai::netimpl::Session::PrepareGet() {
	// holds error codes - all init to OK to prevent errors
	// when checking unset values
	CURLcode e[5]; memset(e, CURLcode::CURLE_OK, sizeof(e));

	if (this->hasBody) {
		e[0] = curl_easy_setopt(this->curl_, CURLOPT_NOBODY, 0L);
		e[1] = curl_easy_setopt(this->curl_, CURLOPT_CUSTOMREQUEST, "GET");
	}
	else {
		e[2] = curl_easy_setopt(this->curl_, CURLOPT_NOBODY, 0L);
		e[3] = curl_easy_setopt(this->curl_, CURLOPT_CUSTOMREQUEST, nullptr);
		e[4] = curl_easy_setopt(this->curl_, CURLOPT_HTTPGET, 1L);
	}
	
	ErrorCheck(e, 5, "liboai::netimpl::Session::PrepareGet()");
		
	this->Prepare();
}
```

目前讲到这里了，对源码有兴趣的朋友可以自己去研究一下。

## rust async-openai  

采用tokio的实现的一个异步库哈，api也很清晰。这里觉得有趣的，第一个可以学习一下这个库，第二个我发现作者自己写了一个应用[Song search in Rust using OpenAI](https://gigapotential.dev/blog/song-search-in-rust-using-openai/)。
和RAG知识库原理基本一样，把歌词和作者信息等通过embedding模型转换成vector信息，将歌词信息和vector信息存储到向量数据库中，然后通过距离算法，这里使用了余弦距离，计算相似度。然后搜索出排名靠前的音乐。
### async-openai 用法

```rust
use std::error::Error;

use async_openai::{
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::new();

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(512u32)
        .model("gpt-3.5-turbo")
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Who won the world series in 2020?")
                .build()?
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("The Los Angeles Dodgers won the World Series in 2020.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Where was it played?")
                .build()?
                .into(),
        ])
        .build()?;

    println!("{}", serde_json::to_string(&request).unwrap());

    let response = client.chat().create(request).await?;

    println!("\nResponse:\n");
    for choice in response.choices {
        println!(
            "{}: Role: {}  Content: {:?}",
            choice.index, choice.message.role, choice.message.content
        );
    }

    Ok(())
}
```

### 使用openai进行音乐搜索

源码[64bit/song-search-rust-openai](https://github.com/64bit/song-search-rust-openai)。
看了以后发现，你也可以打造一个自己的应用，是不是很简单。
作者当时看了很多向量数据库哈，最后选择了用pgvector - Postgres数据库的扩展。采用SQL语法，对向量存储和查询。

> During my non-exhaustive search I found various vector databases and libraries like pinecone, milvus, Weaviate, and Faiss. And there are many more. But none of them seem to have an out-of-the-box ready to go library in Rust. Except pgvector - a Postgres extension to store and query vectors! The pgvector project provides Docker image with extension already installed as well as Rust library 😎

插入：
```rust
impl Song {
    /// Save embedding for this Song in DB
    pub async fn save_embedding(&self, pg_pool: &PgPool,
      pgvector: pgvector::Vector) -> Result<()> {

        sqlx::query(r#"INSERT INTO songs
            (artist, title, album, lyric, embedding)
            VALUES ($1, $2, $3, $4, $5)"#)
            .bind(self.artist.clone())
            .bind(self.title.clone())
            .bind(self.album.clone())
            .bind(self.lyric.clone())
            .bind(pgvector)
            .execute(pg_pool)
            .await?;

        Ok(())
    }
}
```

搜索：
```rust
// Search for nearest neighbors in database
Ok(sqlx::query(
    r#"SELECT artist, title, album, lyric
        FROM songs ORDER BY embedding <-> $1 LIMIT $2::int"#,
)
.bind(pgvector)
.bind(n)
.fetch_all(pg_pool)
.await?
.into_iter()
.map(|r| Song {
    artist: r.get("artist"),
    title: r.get("title"),
    album: r.get("album"),
    lyric: r.get("lyric"),
})
.collect())
```
bye！！！有兴趣的小伙伴可以跟着我的链接，详细看一看源码。