# OpenKnow

基于 RAG 的知识库系统，支持语义搜索 Markdown 文档，并可作为 MCP Server 接入 Cursor、Claude Code 等 AI 编辑器。

## 安装与准备

```bash
uv sync
```

### 1. 添加知识源

```bash
uv run main.py add /path/to/your/knowledge
```

### 2. 配置文件

编辑 `config.json` 配置 embedding 与 vision（可选）：

```json
{
  "embedding": {
    "provider": "local",           // local | openai
    "local_model": "BAAI/bge-small-zh-v1.5",
    "openai_model": "text-embedding-3-small"
  },
  "vision": {
    "skip": true,                  // true 时跳过图片描述，加速索引
    "model": "gpt-4o",
    "api_key": "sk-xxx",
    "base_url": "https://..."      // 可选，公司内部 API 地址
  }
}
```

---

## MCP 使用说明

MCP Server 将知识库作为工具暴露给 AI 助手，可在 Cursor、Claude Code 中调用。

### 启动 MCP Server

**方式一：HTTP 模式（推荐，用于 Cursor）**

```bash
uv run main.py serve --transport streamable-http --host 127.0.0.1 --port 8787
```

默认即为 `streamable-http`，可简写为：

```bash
uv run main.py serve
```

**方式二：SSE 模式**

```bash
uv run main.py serve --transport sse --port 8787
```

**方式三：stdio 模式（适用于 Claude Code 子进程）**

```bash
uv run main.py serve --transport stdio
```

### 在 Cursor 中配置 MCP

#### HTTP 方式（多 Cursor 窗口共享同一 Server）

**步骤 1：启动 MCP Server（需保持运行）**

在终端中执行，并保持该窗口常开：

```bash
cd /path/to/OpenKnow
uv run main.py serve
```

输出类似 `Listening on http://127.0.0.1:8787` 即表示已启动。

**步骤 2：配置 Cursor**

打开 Cursor 设置 → **Features** → **MCP**，编辑 MCP 配置（如 `~/.cursor/mcp.json`）：

```json
{
  "mcpServers": {
    "openknow": {
      "url": "http://127.0.0.1:8787/sse"
    }
  }
}
```

**步骤 3：重启 Cursor 或重新加载 MCP**

> 若出现 `ECONNREFUSED 127.0.0.1:8787`，说明 Server 未启动。请先完成步骤 1，再在 Cursor 中重试。

---

#### stdio 方式（Cursor 按需自动启动，适合单项目）

```json
{
  "mcpServers": {
    "openknow": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/OpenKnow", "main.py", "serve", "--transport", "stdio"]
    }
  }
}
```

将 `/path/to/OpenKnow` 替换为 OpenKnow 项目的绝对路径。

### 在 Claude Code 中配置 MCP

编辑 Claude Code 的 MCP 配置（如 `claude_desktop_config.json` 或 `.claude/mcp.json`），添加：

```json
{
  "mcpServers": {
    "openknow": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/OpenKnow", "main.py", "serve", "--transport", "stdio"]
    }
  }
}
```

### MCP 提供的工具

| 工具 | 说明 |
|------|------|
| `search_knowledge` | 语义搜索知识库，返回相关内容片段 |
| `add_directory` | 将目录中的 Markdown 加入知识库并建立索引 |
| `add_content` | 添加单条内容（保存为 Markdown 并索引） |
| `list_sources` | 列出已索引的目录及统计信息 |

**示例用法（在 Cursor/Claude 对话中）：**

- 「搜索一下关于 XXX 的文档」
- 「把 /path/to/docs 加到知识库」
- 「查看当前知识库有哪些来源」

---

## CLI 命令

| 命令 | 说明 |
|------|------|
| `add <dir...>` | 添加并索引目录 |
| `search <query>` | 搜索知识库 |
| `status` | 查看索引统计 |
| `reindex` | 全量重建索引 |
| `remove <dir>` | 移除目录并删除其索引 |
| `serve` | 启动 MCP Server |
