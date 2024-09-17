---
title: helix-gpt如何实现AI code以及如何调试?
date: 2024-09-17
authors: [KenForever1]
categories: 
  - vim
  - ai
labels: []
---

## helix-gpt如何实现ai编码功能的？


[helix-gpt](https://github.com/KenForever1/helix-gpt)是个language server，通过调用大语言模型，实现对代码的智能提示和编辑，由于是使用大模型处理，所有和语言无关，所有语言都可以使用。helix本身内置languageserver协议，因为都采用了lsp协议。其实vim、vscode也是可以用的，我们看到的Github Copilot、文心快码、通义灵码等应用都是这样实现的。
<!-- more -->
作为client和server沟通好以后，包括server告诉client，我支持哪些功能啊，比如将注释翻译为代码、根据代码续写、写单测。那么client收到了以后就可以在界面上显示出按钮，我点那个功能，就发送一个对应的action request给server。
比如代码续写，想一想需要什么信息呢？肯定需要把当前代码的位置，pos（row， col）以及那个文件/home/test.ts, 还有action，一起发送给server。language server收到后，根据文件和位置读取上下文，比如前面200个字节，后面100个字节，然后通过大模型调用接口，调用大模型。

比如, 当然也可以输入中文给到中文模型：
```json
    const messages = [
      {
        "content": `You are an AI programming assistant.\nWhen asked for your name, you must respond with \"GitHub Copilot\".\nFollow the user's requirements carefully & to the letter.\n- Each code block starts with \`\`\` and // FILEPATH.\n- You always answer with ${languageId} code.\n- When the user asks you to document something, you must answer in the form of a ${languageId} code block.\nYour expertise is strictly limited to software development topics.\nFor questions not related to software development, simply give a reminder that you are an AI programming assistant.\nKeep your answers short and impersonal.`,
        "role": "system"
      },
      {
        "content": `I have the following code in the selection:\n\`\`\`${languageId}\n// FILEPATH: ${filepath.replace('file://', '')}\n${contents}`,
        "role": "user"
      },
      {
        "content": request,
        "role": "user"
      }
    ]
```
然后大模型返回结果，再发送给client，写到编辑器里面。
下面的例子是openai的接口，其它平台有对应的接口，实现不同的处理逻辑就可以了。
```json
{
        "choices": [
          {
            "index": 0,
            "message": {
              "role": "assistant",
              "content": "const name: string = \"John\";\nconsole.log(\"Hello, \" + name);"
            },
          },
          {
            "index": 1,
            "message": {
              "role": "assistant",
              "content": "console.log(\"Hello, world!\");"
            },
          },
          {
            "index": 2,
            "message": {
              "role": "assistant",
              "content": "const name: string = \"John\";\nconsole.log(\"Hello, \" + name);"
            },
          }
        ],
      }
```
这种方式只针对单一文件上下文，那么更复杂的比如需要整个项目的上下文，那肯定需要提供更多的关键信息支持gpt返回提示信息和处理信息。Cursor等AI编辑器实现了针对项目分析的功能。


## language server和client的通信机制

### Event事件

```
export enum Event {
  DidOpen = "textDocument/didOpen",
  DidChange = "textDocument/didChange",
  Completion = "textDocument/completion",
  CodeAction = "textDocument/codeAction",
  ApplyEdit = "workspace/applyEdit",
  ExecuteCommand = "workspace/executeCommand",
  Initialize = "initialize",
  Shutdown = "shutdown",
  Exit = "exit",
  PublishDiagnostics = "textDocument/publishDiagnostics",
}
```

### initialize Event交换信息
初始阶段，client和server通信交换信息。
```
Client --- initalize request          ---> Server
Client <--- response with capabilities---- Server
```

### ShutDown
client要关闭时，比如我关闭了编辑器，发送一个shutdown request给server，server直接就关闭服务，正常退出。比如：
```typescript
this.on(Event.Shutdown, () => {
    log("received shutdown request")
    process.exit(0)
})
```

这里以Initialize阶段看看Client和Server是如何通信的。

### Initialize 请求操作的处理是怎样的？

#### initialize request
下面的打印信息是server收到了request，然后发送了一个response信息，response包括了capabilities，包括了commands，比如：["resolveDiagnostics","generateDocs","improveCode","refactorFromComment","writeTest"]，生成文档、优化代码、根据注释写代码、写单测等。
```bash
APP 2024-09-16T13:21:56.244Z --> triggerCharacters: | ["{","("," "]

APP 2024-09-16T13:21:56.253Z --> received request: | {"id":"20dcc99c-b053-49a1-b972-e360366538d9","params":{"capabilities":{},"processId":298598,"rootPath":null,"rootUri":null,"workspaceFolders":null},"method":"initialize","jsonrpc":"2.0"}

APP 2024-09-16T13:21:56.253Z --> this.cap  | [object Object]

APP 2024-09-16T13:21:56.253Z --> sent request | {"jsonrpc":"2.0","id":"20dcc99c-b053-49a1-b972-e360366538d9","result":{"capabilities":{"codeActionProvider":true,"executeCommandProvider":{"commands":["resolveDiagnostics","generateDocs","improveCode","refactorFromComment","writeTest"]},"completionProvider":{"resolveProvider":false,"triggerCharacters":["{","("," "]},"textDocumentSync":{"change":1,"openClose":true}}}}
```

客户端收到的请求是怎么样的呢？包括了Content-Length和json内容，采用jsonrpc2.0协议。
```bash
 b'Content-Length: 369\r\n\r\n{"jsonrpc":"2.0","id":"0a419853-3d6a-4236-8694-9d351b40ea28","result":{"capabilities":{"codeActionProvider":true,"executeCommandProvider":{"commands":["resolveDiagnostics","generateDocs","improveCode","refactorFromComment","writeTest"]},"completionProvider":{"resolveProvider":false,"triggerCharacters":["{","("," "]},"textDocumentSync":{"change":1,"openClose":true}}}}'
```

在开发中遇到了一个问题，在server里添加打印后，发现打印内容也发送给了客户端，这应该是输出重定向的问题？
这时客户端收到的是这样的:
```bash
b'values :  [Object: null prototype] {\n  handler: "qianwen",\n ... ... ollamaTimeout: "60000"\n}\n Content-Length: 369\r\n\r\n{"jsonrpc":"2.0","id":"d2742aeb-379e-4cbb-959b-09da7ed9dfdc","result":{"capabilities":{"codeActionProvider":true,"executeCommandProvider":{"commands":["resolveDiagnostics","generateDocs","improveCode","refactorFromComment","writeTest"]},"completionProvider":{"resolveProvider":false,"triggerCharacters":["{","("," "]},"textDocumentSync":{"change":1,"openClose":true}}}}'
```
可以看到在Content-Length前面多了些内容，这正是我们language-server log打印的内容。导致python client正则表达式匹配解析不了。还以为是没有收到返回Responce。
pygls json_rpc.py:
```python
MESSAGE_PATTERN = re.compile(
    rb"^(?:[^\r\n]+\r\n)*"
    + rb"Content-Length: (?P<length>\d+)\r\n"
    + rb"(?:[^\r\n]+\r\n)*\r\n"
    + rb"(?P<body>{.*)",
    re.DOTALL,
)

# Look for the body of the message
message = b"".join(self._message_buf)
print("wt message, ", message)
found = JsonRPCProtocol.MESSAGE_PATTERN.fullmatch(message)
```

## 如何通过pytest-lsp client调试server

其实通过client调试server，就是用client模拟了我们的编辑器，比如vscode、vim、helix等等，这样就不用考虑编辑器的复杂性，只用关注client和server的通信。

pytest-lsp就是python实现的测试工具。先来看一个例子：
写个简单的python实现的server，使用pytest-lsp测试，通过了再把serve替换成我们的server不就ok了。开干！

### pytest-lsp 使用方法
先写个server.py，很简单吧。
```python
from lsprotocol.types import TEXT_DOCUMENT_COMPLETION, CompletionItem, CompletionParams
from pygls.server import LanguageServer

server = LanguageServer("hello-world", "v1")


@server.feature(TEXT_DOCUMENT_COMPLETION)
def completion(ls: LanguageServer, params: CompletionParams):
    return [
        CompletionItem(label="hello"),
        CompletionItem(label="world"),
    ]

if __name__ == "__main__":
    server.start_io()
```

然后写个Client，test_server.py,
```python
import sys

import pytest
import pytest_lsp
from lsprotocol.types import (
    ClientCapabilities,
    CompletionList,
    CompletionParams,
    InitializeParams,
    Position,
    TextDocumentIdentifier,
)
from pytest_lsp import ClientServerConfig, LanguageClient


@pytest_lsp.fixture(
    # 调用python实现的server.py 
    config=ClientServerConfig(server_command=[sys.executable, "server.py"]),

    # 调用helix-gpt实现的server
    # config=ClientServerConfig(server_command=["/usr/bin/helix-gpt", "--handler", "qianwen", "--logFile", "/home/ken/helix-gpt-1.log"]),
)
async def client(lsp_client: LanguageClient):
    # Setup
    params = InitializeParams(capabilities=ClientCapabilities())
    print(params)
    print(" Initializing session...")
    await lsp_client.initialize_session(params)
    
    print(' Init Done!')
    

    yield

    # Teardown
    await lsp_client.shutdown_session()

@pytest.mark.asyncio
async def test_completions(client: LanguageClient):
    """Ensure that the server implements completions correctly."""

    print("Completion test")
    results = await client.text_document_completion_async(
        params=CompletionParams(
            position=Position(line=1, character=0),
            text_document=TextDocumentIdentifier(uri="file:///home/ken/test.ts"),
        )
    )
    assert results is not None
    print("results", results)

    if isinstance(results, CompletionList):
        items = results.items
    else:
        items = results

    labels = [item.label for item in items]
    assert labels == ["hello", "world"]

```

运行测试：
```
python3 -m pip install pytest-lsp

# $work_dir是test_server.py存在的目录
cd $work_dir
pytest -s
```

### 使用pytest-lsp测试helix-gpt server

#### 编译helix-gpt

helix-gpt是使用typescript开发的项目，可以通过bun和deno进行编译运行。bun和demo是javascript runtime，bun是zig开发的，deno是rust开发的，和nodejs对标。[bun存在bug](https://github.com/leona/helix-gpt/issues/49)，这里采用deno编译：
```bash
deno compile --output helix-gpt --no-check --allow-env --allow-net --allow-write ./src/app.ts
```

换成我们的helix-gpt server试试：
```python
@pytest_lsp.fixture(
    # 调用python实现的server.py 
    # config=ClientServerConfig(server_command=[sys.executable, "server.py"]),

    # 调用helix-gpt实现的server
    config=ClientServerConfig(server_command=["/usr/bin/helix-gpt", "--handler", "qianwen", "--logFile", "/home/ken/helix-gpt-1.log"]),
)
```

```bash
# -s 是为了在python执行test时也可以看到print函数输出
pytest -s
```

## 增加通义千问支持

通义千问支持两种api，OPENAI兼容和DashScope，这里以OPENAI兼容API为例：
增加参数：
```typescript
qianwenKey: {
  type: 'string',
  default: Bun.env.QIANWEN_API_KEY?? "xxx"
},
qianwenContext: {
  type: 'string',
  default: Bun.env.QIANWEN_CONTEXT?.length ? Bun.env.QIANWEN_CONTEXT : context.qianwen
},
qianwenModel: {
  type: 'string',
  default: Bun.env.QIANWEN_MODEL ?? "qwen1.5-1.8b-chat"
},
qianwenMaxTokens: {
  type: 'string',
  default: Bun.env.QIANWEN_MAX_TOKENS ?? "2000"
},
qianwenEndpoint: {
  type: 'string',
  default: Bun.env.QIANWEN_ENDPOINT ?? 'https://dashscope.aliyuncs.com/'
},
```
实现qianwen provider,详细API参考[API通义千问](https://help.aliyun.com/zh/model-studio/developer-reference/use-open-source-qwen-by-calling-api?spm=a2c4g.11186623.0.i2#3a11734087e4d)。
```typescript
export default class Qianwen extends ApiBase {

  constructor() {
    super({
      url: config.qianwenEndpoint as string,
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${config.qianwenKey}`
      }
    })
  }

async completion(contents: any, filepath: string, languageId: string, suggestions = 3): Promise<types.Completion> {
    const messages = [
      {
        role: "system",
        content: config.qianwenContext?.replace("<languageId>", languageId) + "\n\n" + `End of file context:\n\n${contents.contentAfter}`
      },
      {
        role: "user",
        content: `Start of file context:\n\n${contents.contentBefore}`
      }
    ]

    const body = {
      model: config.qianwenModel,
      max_tokens: parseInt(config.qianwenMaxTokens as string),
      n: suggestions,
      temperature: suggestions > 1 ? 0.4 : 0,
      frequency_penalty: 1,
      presence_penalty: 2,
      messages
    }
    log("qianwen request completion : ", body)

    const data = await this.request({
      method: "POST",
      body,
      endpoint: "/compatible-mode/v1/chat/completions"
    })

    return types.Completion.fromResponse(data)
  }
}
```
ai编程提示的效果，和模型的相关性特别高，比如收费模型"qwen-max"相对"qwen1.5-1.8b-chat"的提示是又快又准。
