# 🚀 superbot

**superbot** 是一个基于 **LangGraph** 和 **MCP (Model Context Protocol)** 的自主进化型 Linux AI 助手。它不仅能执行复杂的 Linux 系统任务，还能通过 RAG (经验检索) 从过去的操作中学习，并自动生成新工具以扩展自己的能力。

## 🌟 核心特性

- 🧬 **自我进化**: AI 能诊断自己的性能瓶颈，并自动编写、测试、应用 Python 工具。
- 🧠 **经验记忆 (RAG)**: 使用 ChromaDB 存储操作日志和成功模式，实现“肌肉记忆”。
- 🔧 **工具插件化**: 深度集成 MCP 协议，实现大脑（LLM）与工具（Server）的完全解耦。
- 📊 **可视化监控**: 提供流式响应的 Web 界面，实时查看任务执行进度与进化状态。
- 🛡️ **安全沙盒**: 内置危险命令检测与人工审批机制，确保系统安全。
- 🔌 **多模型支持**: 原生支持通义千问 (Qwen)、DeepSeek、GPT-4、Gemini 等主流大模型。

## 🛠️ 快速开始

### 1. 安装

```bash
# 克隆项目
git clone <repository-url>
cd superbot

# 安装依赖
pip install -r requirements.txt
pip install python-dotenv
```

### 2. 配置

创建 `.env` 文件并填入你的 API Key：

```bash
# 在根目录创建 .env
echo "QWEN_API_KEY=your-api-key-here" > .env
```

或者使用 `config.yaml` 进行更高级的配置：

```bash
cp config.example.yaml config.yaml
```

### 3. 运行

**方式 A: Web 界面 (推荐)**
```bash
python3 start_superbot.py
# 访问 http://localhost:8000
```

**方式 B: 启用进化模式启动**
```bash
python3 start_superbot.py --evolution --interval 12
```

**方式 C: 命令行交互**
```bash
superbot --provider qwen
```

## 📂 项目结构

```text
superbot/
├── src/
│   ├── core/           # 核心引擎 (LangGraph, Multi-Agent)
│   ├── evolution/      # 进化系统 (RAG, Self-Diagnosis, Auto-Fix)
│   ├── mcp/            # MCP 协议层 (Client, Servers)
│   ├── utils/          # 通用工具 (tools.py, Prompts)
│   ├── web/            # Web 界面 (FastAPI)
│   └── cli/            # 命令行接口
├── experience_db/      # 经验存储 (ChromaDB)
├── docs/               # 技术文档
├── start_superbot.py   # 主启动脚本
└── config.yaml         # 配置文件
```

## 📖 深入了解

- [架构设计详解](docs/architecture.md)
- [自我进化机制](docs/architecture.md#自我进化循环)

## 许可证

MIT License
