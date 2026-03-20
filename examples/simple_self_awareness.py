#!/usr/bin/env python3
"""
最简单的自我感知示例 - 5 分钟体验

这个脚本展示了如何直接调用自我感知工具
"""

import asyncio
import json
from src.utils.tools import execute_tool


async def example_1_read_code():
    """示例 1: 读取自己的代码"""
    print("\n" + "="*60)
    print("示例 1: AI 读取自己的 Prompt")
    print("="*60)
    
    result = await execute_tool("read_own_code", {
        "module": "prompts",
        "search_pattern": "核心原则"
    })
    
    data = json.loads(result)
    print("\n找到的内容:")
    for filepath, info in data['files'].items():
        if "matches" in info:
            for match in info['matches'][:2]:  # 只显示前2个匹配
                print(f"\n在第 {match['line']} 行:")
                print(match['context'])


async def example_2_analyze():
    """示例 2: 分析自己的表现"""
    print("\n" + "="*60)
    print("示例 2: AI 分析自己的表现")
    print("="*60)
    
    result = await execute_tool("analyze_performance", {
        "time_range": "all",
        "focus": "success_rate"
    })
    
    data = json.loads(result)
    
    if "analysis" in data and "success_rate" in data["analysis"]:
        sr = data["analysis"]["success_rate"]
        print(f"\n📊 总任务数: {sr['total_tasks']}")
        print(f"✅ 成功: {sr['successful']}")
        print(f"❌ 失败: {sr['failed']}")
        print(f"📈 成功率: {sr['success_percentage']}%")
    
    if "improvement_suggestions" in data:
        print(f"\n💡 改进建议:")
        for suggestion in data["improvement_suggestions"]:
            print(f"   {suggestion}")


async def example_3_review():
    """示例 3: 查看历史经验"""
    print("\n" + "="*60)
    print("示例 3: AI 查看历史经验")
    print("="*60)
    
    result = await execute_tool("review_experiences", {
        "filter": "recent",
        "limit": 3,
        "analyze": True
    })
    
    data = json.loads(result)
    
    if "message" in data:
        print(f"\nℹ️  {data['message']}")
        return
    
    print(f"\n找到 {data.get('total_found', 0)} 条经验\n")
    
    for i, exp in enumerate(data.get("experiences", []), 1):
        status = "✅" if exp.get("success") else "❌"
        print(f"{status} 经验 {i}:")
        print(f"   问题: {exp.get('problem', '')[:80]}...")
        print(f"   工具: {', '.join(exp.get('tools_used', [])[:3])}")
        print()
    
    if "patterns" in data:
        patterns = data["patterns"]
        print("🔍 发现的模式:")
        if patterns.get("effective_tools"):
            print(f"   最有效的工具: {list(patterns['effective_tools'].keys())[:3]}")


async def main():
    """运行所有示例"""
    print("\n🧠 SuperLinux 自我感知 - 简单示例")
    print("="*60)
    
    await example_1_read_code()
    await example_2_analyze()
    await example_3_review()
    
    print("\n" + "="*60)
    print("✅ 示例完成！")
    print("\n下一步:")
    print("  1. 运行完整测试: python3 test_self_awareness.py")
    print("  2. 查看演示: python3 examples/self_awareness_demo.py")
    print("  3. 与 AI 对话: python3 -m src.cli")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
