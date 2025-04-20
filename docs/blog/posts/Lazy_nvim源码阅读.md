---
title: Lazy_nvim源码阅读
date: 2024-09-17
authors: [KenForever1]
categories: 
  - vim
labels: []
comments: true
---

# Core模块
## LazyHandler手动类和继承机制实现


在~/lua/lazy/core/handler/init.lua文件中，定义了LazyHandler类。handler类中定义了四个具体的子类型，分别是keys、event、cmd和ft。分别在cmd.lua、event.lua、ft.lua和keys.lua文件中定义了具体的handler实现。

<!-- more -->

注：代码中的注释格式很好，在写lua程序时，可以按照这种模式。它基于 Lua 的文档注释格式，兼容多种工具和编辑器插件，可以提升代码的可读性和提供类型提示。
```lua
---@class LazyHandler
---@field type LazyHandlerTypes
---@field extends? LazyHandler
---@field active table<string,table<string,string>>
---@field managed table<string,string> mapping handler keys to plugin names
---@field super LazyHandler
local M = {}

---@enum LazyHandlerTypes
M.types = {
  keys = "keys",
  event = "event",
  cmd = "cmd",
  ft = "ft",
}

---@type table<string,LazyHandler>
M.handlers = {}

```

通过`M.new()`函数，可以创建一个新的LazyHandler实例。在其中设置self.super为具体handler实现类的extends变量。利用lua元表的特性，设置了__index

通过动态加载模块和手动设置元表，提供了一种灵活的对象创建和继承机制。基于字符串标识符动态选择和加载不同的处理器模块。

```lua
---@param type LazyHandlerTypes
function M.new(type)
  ---@type LazyHandler
  local handler = require("lazy.core.handler." .. type)
  local super = handler.extends or M
  local self = setmetatable({}, { __index = setmetatable(handler, { __index = super }) })
  self.super = super
  self.active = {}
  self.managed = {}
  self.type = type
  return self
end
```

```lua
local self = setmetatable({}, { __index = setmetatable(handler, { __index = super }) })
```
创建一个新的空表 {}，并通过 setmetatable 设置其元表。这个元表的 __index 指向一个新的元表，且该元表的 __index 指向 super，这样实现了多层次的继承链。这种设置方式确保：当在 self 中查找某个方法或属性时，首先查找 handler，如果找不到，再查找 super。
### 以ft.lua为例

在ft.lua中，定义了一个名为LazyFiletypeHandler的类，它.extends=Event表示它继承自Event。

```lua
---@class LazyFiletypeHandler:LazyEventHandler
local M = {}
M.extends = Event

function M:add(plugin)
  self.super.add(self, plugin)
  if plugin.ft then
    Loader.ftdetect(plugin.dir)
  end
end

function M:_parse(value)
  return {
    id = value,
    event = "FileType",
    pattern = value,
  }
end

return M
```

### 示例：扩展LazyFiletypeHandler
如果要扩展LazyFiletypeHandler以添加额外的功能或修改现有行为，可以创建一个新的Lua模块，如下所示：
```lua
local Event = require("lazy.core.handler.event")
local Loader = require("lazy.core.loader")

---@class LazyFiletypeHandler:LazyEventHandler
local M = {}
M.extends = Event

---@param plugin LazyPlugin
function M:add(plugin)
  self.super.add(self, plugin)
  if plugin.ft then
    Loader.ftdetect(plugin.dir)
  end
end

---@return LazyEvent
function M:_parse(value)
  return {
    id = value,
    event = "FileType",
    pattern = value,
  }
end

return M

```


### 启用和禁用插件
enable和disable函数分别用于启用和禁用插件。通过遍历插件的handlers，add或del方法来实现。
```lua
---@param plugin LazyPlugin
function M.disable(plugin)
  for type in pairs(plugin._.handlers or {}) do
    M.handlers[type]:del(plugin)
  end
end

---@param plugin LazyPlugin
function M.enable(plugin)
  if not plugin._.loaded then
    if not plugin._.handlers then
      M.resolve(plugin)
    end
    for type in pairs(plugin._.handlers or {}) do
      M.handlers[type]:add(plugin)
    end
  end
end
```

### Lua中的元表
> 元表（metatable）：
> 表（Table）：在 Lua 中，表是唯一的数据结构，可以用作数组、字典等。
> 元表（Metatable）：是一个表，它可以赋予另一个表特殊的行为。通过设置一个表的元表，可以改变该表的行为和操作方式。通过元表，你可以定义表的自定义行为。

+ __index：
用于定义当访问表中不存在的键时的行为。
可以是一个表或一个函数。
如果是表，则会在该表中查找键。
如果是函数，则会调用该函数，并传递表和键作为参数。

+ __newindex：
用于定义对表中不存在的键进行赋值时的行为。
函数类型，接收表、键和值作为参数。

+ __tostring：
用于定义将表转换为字符串时的行为，例如在 print 函数中。

+ __call：
使得表能够像函数一样被调用。

# Manage模块

## task中的fs实现

> vim.uv exposes the "luv" Lua bindings for the libUV library that Nvim uses for networking, filesystem, and process management, see luvref.txt.

对文件系统的操作lua调用了vim.uv，是对libuv库的封装。例如：

```lua
local function rm(dir)
  local stat = vim.uv.fs_lstat(dir)
  assert(stat and stat.type == "directory", dir .. " should be a directory!")
  Util.walk(dir, function(path, _, type)
    if type == "directory" then
      vim.uv.fs_rmdir(path)
    else
      vim.uv.fs_unlink(path)
    end
  end)
  vim.uv.fs_rmdir(dir)
end
```

## task 实现

### task 异步回调注册


Task类继承了Async.Async类，并实现了_run方法。在_run中注册了done、error和yield事件的处理函数。

```lua
local Task = setmetatable({}, { __index = Async.Async })

---@param plugin LazyPlugin
---@param name string
---@param opts? TaskOptions
---@param task LazyTaskFn
function Task.new(plugin, name, task, opts)
  local self = setmetatable({}, { __index = Task })
  ---@async
  Task.init(self, function()
    self:_run(task)
  end)
  ......
  return self
end

---@async
---@param task LazyTaskFn
function Task:_run(task)
  if Config.headless() and Config.options.headless.task then
    self:log("Running task " .. self.name, vim.log.levels.INFO)
  end

  self
    :on("done", function()
      self:_done()
    end)
    :on("error", function(err)
      self:error(err)
    end)
    :on("yield", function(msg)
      self:log(msg)
    end)
  task(self, self._opts)
end
```

在Async.Async类中，on方法用于注册事件的处理函数。例如：

```lua
---@param event AsyncEvent
---@param cb async fun(res:any, async:Async)
function Async:on(event, cb)
  --- _on 是一个表，用于存储事件的处理函数列表, 一个事件可以有多个处理函数。
  self._on[event] = self._on[event] or {}
  table.insert(self._on[event], cb)
  return self
end
```

在~/lua/lazy/async.lua, Async.init中，通过coroutine创建了一个协程，并在其中调用_fn函数。如果发生错误，则触发error事件；否则，在任务完成后触发done事件。
```lua
---@param fn async fun()
---@return Async
function Async:init(fn)
  self._fn = fn
  self._on = {}
  self._co = coroutine.create(function()
    local ok, err = pcall(self._fn)
    if not ok then
      self:_emit("error", err)
    end
    self:_emit("done")
  end)
  M._threads[self._co] = self
  return M.add(self)
end

---@private
---@param event AsyncEvent
---@param res any
function Async:_emit(event, res)
  for _, cb in ipairs(self._on[event] or {}) do
    cb(res, self)
  end
end
```



### Task:_done() 实现

在上面提到的Task:_run中，注册了done事件的处理函数_done。在插件加载完成后调用此方法。


> When running in headless mode, lazy.nvim will log any messages to the terminal. See opts.headless for more info.

在无头模式下，lazy.nvim会将任何消息记录到终端。

```lua
---@private
function Task:_done()
  --- 如果开启无头模式，打印日志到终端
  if Config.headless() and Config.options.headless.task then
    local ms = math.floor(self:time() + 0.5)
    self:log("Finished task " .. self.name .. " in " .. ms .. "ms", vim.log.levels.INFO)
  end
  self._ended = vim.uv.hrtime()
  if self._opts.on_done then
    self._opts.on_done(self)
  end
  --- 刷新界面，手动触发渲染界面，并执行自动命令。
  self:render()
  -- 触发插件注册的自动命令
  vim.schedule(function()
    vim.api.nvim_exec_autocmds("User", {
      pattern = "LazyPlugin" .. self.name:sub(1, 1):upper() .. self.name:sub(2),
      data = { plugin = self.plugin.name },
    })
  end)
end
```

[nvim_exec_autocmds api定义](https://neovim.io/doc/user/api.html#api-autocmd)。

Task中self.log实现：
```lua
  io.write(Terminal.prefix(color and color(msg.msg) or msg.msg, self:prefix()))
  io.write("\n")
```

#### render函数刷新界面

render函数用于刷新界面，手动触发渲染界面，并执行自动命令。实现：
```lua
function Task:render()
  vim.schedule(function()
    vim.api.nvim_exec_autocmds("User", { pattern = "LazyRender", modeline = false })
  end)
end
```
会触发view中注册的autocmds。~/lua/lazy/view/init.lua中注册的相关代码如下：
```lua
  for _, pattern in ipairs({ "LazyRender", "LazyFloatResized" }) do
    self:on({ "User" }, function()
      if not (self.buf and vim.api.nvim_buf_is_valid(self.buf)) then
        return true
      end
      self:update()
    end, { pattern = pattern })
  end
```
### task在哪儿被使用

```lua
---@param plugin LazyPlugin
---@param step PipelineStep
---@return LazyTask?
function Runner:queue(plugin, step)
  assert(self._running and self._running:running(), "Runner is not running")
  local def = vim.split(step.task, ".", { plain = true })
  ---@type LazyTaskDef
  local task_def = require("lazy.manage.task." .. def[1])[def[2]]
  assert(task_def, "Task not found: " .. step.task)
  local opts = step.opts or {}
  if not (task_def.skip and task_def.skip(plugin, opts)) then
    return Task.new(plugin, def[2], task_def.run, opts)
  end
end
```
可以看到在Runner:queue中，通过require引入了task的定义，并执行了Task.new。

## 基于协程的异步Runner

Runner类，用于管理异步任务。它利用lazy.async模块提供的基于协程的异步模型来处理非阻塞任务执行。

### _pipeline标准化

在new函数中有下面的代码：
```lua

---@class TaskOptions: {[string]:any}


---@param step string|(TaskOptions|{[1]:string})
self._pipeline = vim.tbl_map(function(step)
  return type(step) == "string" and { task = step } or { task = step[1], opts = step }
end, self._opts.pipeline)
```
将用户输入的任务配置标准化为统一的格式，方便后续在Runner的任务执行过程中使用。通过这种方式，任务的名称和选项可以被一致地处理，无论用户输入的是简单的任务名称字符串还是包含详细选项的表格。

vim.tbl_map 是一个高阶函数，应用于一个表格的每个元素，并将结果收集成一个新的表格。这里它用来遍历self._opts.pipeline中的每个step。
转换方法：
如果step是一个字符串，直接将其作为任务名称，返回{ task = step }。如果step是一个表格，假定第一个元素是任务名称，其他元素是选项，返回{ task = step[1], opts = step }。

### 核心start函数

_start 方法是 Runner 类中用于异步执行任务的核心方法。它负责管理任务的状态、调度任务的执行，并处理任务的并发和同步。

```lua
---@async
function Runner:_start()
  --- 获取所有插件的名称并进行排序
  ---@type string[]
  local names = vim.tbl_keys(self._plugins)
  table.sort(names)

  --- 创建状态表，用于跟踪每个插件的任务执行步骤
  ---@type table<string,RunnerTask>
  local state = {}

  --- active：当前正在运行的任务数量
  local active = 1
  --- waiting：当前等待同步的任务数量
  local waiting = 0
  ---@type number?
  --- wait_step：当前需要同步的步骤标识
  local wait_step = nil

  while active > 0 do
    continue()
    --- 如果所有任务都完成（active 为 0），且存在等待中的任务，调用同步函数。
    if active == 0 and waiting > 0 then
      local sync = self._pipeline[wait_step]
      if sync and sync.opts and type(sync.opts.sync) == "function" then
        sync.opts.sync(self)
      end
      continue(true)
    end
    --- 如果仍有活动任务，则挂起当前协程以等待任务完成。
    if active > 0 then
      self._running:suspend()
    end
  end
end

```

conitnue是_start函数的内部函数，检查每个插件的当前任务状态：是否正在运行、是否有错误、是否需要等待同步。在_start函数中，循环调用continue函数，持续调度和检查任务状态。

+ 如果任务正在运行，增加 active 计数。
+ 如果遇到 “wait” 类型的任务，并且不在恢复状态，则增加 waiting 计数。
+ 否则，将任务加入到 next 列表中，准备执行下一个任务。

然后，遍历next进行任务调度，遍历 next 列表中的插件名称，调度和执行下一个任务。检查并发限制，确保不会同时运行超过指定数量的任务。对于每个插件，尝试执行下一个步骤：
（1）如果步骤是 “wait”，则标记为等待并增加 waiting 计数。
（2）否则，通过 queue 方法将任务排队执行，并标记为正在工作。
```lua
  ---@async
  ---@param resume? boolean
  local function continue(resume)
    active = 0
    waiting = 0
    wait_step = nil
    local next = {} ---@type string[]

    -- check running tasks
    for _, name in ipairs(names) do
      state[name] = state[name] or { step = 0 }
      local s = state[name]
      local is_running = s.task and s.task:running()
      local step = self._pipeline[s.step]

      --- 如果任务正在运行，增加 active 计数
      if is_running then
        -- still running
        active = active + 1
      -- selene:allow(empty_if)
      elseif s.task and s.task:has_errors() then
        -- don't continue tasks if there are errors
      --- 如果遇到 “wait” 类型的任务，并且不在恢复状态，则增加 waiting 计数
      elseif step and step.task == "wait" and not resume then
        -- waiting for sync
        waiting = waiting + 1
        wait_step = s.step
      --- 否则，将任务加入到 next 列表中，准备执行下一个任务
      else
        next[#next + 1] = name
      end
    end

    --- 遍历 next 列表中的插件名称，调度和执行下一个任务
    -- schedule next tasks
    for _, name in ipairs(next) do
      --- 检查并发限制，确保不会同时运行超过指定数量的任务。
      if self._opts.concurrency and active >= self._opts.concurrency then
        break
      end
      local s = state[name]
      local plugin = self:plugin(name)
      while s.step <= #self._pipeline do
        if s.step == #self._pipeline then
          -- done
          s.task = nil
          plugin._.working = false
          break
        elseif s.step < #self._pipeline then
          -- next
          s.step = s.step + 1
          local step = self._pipeline[s.step]
          if step.task == "wait" then
            plugin._.working = false
            waiting = waiting + 1
            wait_step = s.step
            break
          else
            s.task = self:queue(plugin, step)
            plugin._.working = true
            if s.task then
              active = active + 1
              s.task:wake(false)
              break
            end
          end
        end
      end
    end
  end
```