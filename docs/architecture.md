# Superbot 架构文档

Superbot 是一个基于 **LangGraph** 和 **MCP (Model Context Protocol)** 的自我进化型 Linux AI 助手。

## 目录结构

项目采用模块化设计，核心代码位于 `src/` 目录下：

- **`src/core/`**: 核心大脑
  - `agent.py`: 基础 Agent 实现。
  - `multi_agent.py`: 多 AI 协作与回退逻辑。
  - `orchestrator/`: 基于 LangGraph 的任务编排引擎。
- **`src/evolution/`**: 自我进化系统
  - `self_evolution.py`: 进化引擎，负责代码改进和工具生成。
  - `experience_rag.py`: 基于 ChromaDB 的经验检索增强生成（RAG）。
  - `auto_fixer.py`: 自动代码修复工具。
  - `self_diagnosis.py`: 自我诊断与性能评估。
- **`src/mcp/`**: 功能扩展层
  - `servers/`: 标准化的 MCP 服务器实现（系统监控、文件管理等）。
  - `client/`: MCP 客户端，负责与各服务器通信。
- **`src/utils/`**: 通用工具
  - `tools.py`: 核心工具集。
  - `prompts.py`: 提示词管理。
  - `common/`: 共享模型和异常定义。
- **`src/web/` & `src/cli/`**: 交互层
  - 提供基于 FastAPI 的 Web 界面和交互式命令行。

## 核心技术栈

- **LangGraph**: 构建有向有环图（DAG），实现复杂的推理-行动（ReAct）循环。
- **MCP (Model Context Protocol)**: 实现大脑与工具的彻底解耦，支持插件化扩展。
- **ChromaDB**: 存储运行经验，实现基于 RAG 的“肌肉记忆”。
- **FastAPI**: 提供流式响应的 Web 交互界面。

## 自我进化循环

1. **执行**: 执行任务并记录所有操作日志。
2. **记录**: 将成功/失败的模式存入 `experience_db`。
3. **诊断**: 定期运行 `self_diagnosis` 分析性能瓶颈。
4. **改进**: `self_evolution` 根据诊断结果生成新工具或优化现有代码。
5. **应用**: 自动测试并应用改进方案。
