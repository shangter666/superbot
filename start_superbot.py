#!/usr/bin/env python3
"""
superbot 启动脚本 - 支持命令行参数
"""

import argparse
import sys
import yaml
import uvicorn


def update_config(enable_evolution: bool, interval: int = 24, min_tasks: int = 10, auto_apply: bool = False):
    """更新配置文件"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if 'agent' not in config:
            config['agent'] = {}
        
        config['agent']['auto_evolution'] = {
            'enabled': enable_evolution,
            'check_interval_hours': interval,
            'min_tasks_before_evolution': min_tasks,
            'auto_apply_improvements': auto_apply
        }
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return True
    except Exception as e:
        print(f"❌ 配置更新失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='🚀 superbot - 自主进化的 Linux AI 助手',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 正常启动（不启用自我进化）
  python3 start_agent.py
  
  # 启用自我进化模式
  python3 start_agent.py --evolution
  
  # 启用自我进化，每12小时检查一次
  python3 start_agent.py --evolution --interval 12
  
  # 启用自我进化，自动应用改进
  python3 start_agent.py --evolution --auto-apply
  
  # 完整配置
  python3 start_agent.py --evolution --interval 6 --min-tasks 5 --auto-apply
  
  # 指定端口
  python3 start_agent.py --port 8080
        '''
    )
    
    # 自我进化相关参数
    evolution_group = parser.add_argument_group('🧬 自我进化模式')
    evolution_group.add_argument(
        '--evolution', '-e',
        action='store_true',
        help='启用自我进化模式（AI 会自动学习和改进）'
    )
    evolution_group.add_argument(
        '--infinite-evolution',
        action='store_true',
        help='启用无限进化模式（永不停止，持续改进）'
    )
    evolution_group.add_argument(
        '--interval', '-i',
        type=int,
        default=24,
        metavar='HOURS',
        help='进化检查间隔（小时，默认: 24）'
    )
    evolution_group.add_argument(
        '--min-tasks', '-m',
        type=int,
        default=10,
        metavar='N',
        help='触发进化的最小任务数（默认: 10，无限模式忽略此参数）'
    )
    evolution_group.add_argument(
        '--auto-apply', '-a',
        action='store_true',
        help='自动应用改进（不询问用户确认）'
    )
    
    # 服务器相关参数
    server_group = parser.add_argument_group('🌐 服务器设置')
    server_group.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='监听地址（默认: 0.0.0.0）'
    )
    server_group.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='监听端口（默认: 8000）'
    )
    server_group.add_argument(
        '--reload',
        action='store_true',
        help='启用热重载（开发模式）'
    )
    
    args = parser.parse_args()
    
    # 显示启动信息
    print("=" * 60)
    print("🚀 superbot")
    print("=" * 60)
    
    # 无限进化模式优先
    enable_evolution = args.evolution or args.infinite_evolution
    
    # 更新配置
    if update_config(
        enable_evolution=enable_evolution,
        interval=args.interval,
        min_tasks=0 if args.infinite_evolution else args.min_tasks,  # 无限模式不需要最小任务数
        auto_apply=args.auto_apply or args.infinite_evolution  # 无限模式自动应用
    ):
        print("✅ 配置已更新")
    
    # 显示配置信息
    print("\n📋 当前配置:")
    print(f"   监听地址: {args.host}:{args.port}")
    
    if args.infinite_evolution:
        print(f"\n🧬 无限进化模式: 已启用 ⚡")
        print(f"   检查间隔: {args.interval} 小时")
        print(f"   自动应用: 是")
        print(f"   模式: 永不停止")
        print("\n   AI 将会:")
        print("   • 每 {args.interval} 小时自动审计代码")
        print("   • 发现问题立即修复")
        print("   • 持续优化性能和质量")
        print("   • 永不停止，不断进化")
        print("\n   ⚠️  这是最激进的进化模式！")
    elif args.evolution:
        print(f"\n🧬 自我进化模式: 已启用")
        print(f"   检查间隔: {args.interval} 小时")
        print(f"   最小任务数: {args.min_tasks}")
        print(f"   自动应用: {'是' if args.auto_apply else '否'}")
        print("\n   AI 将会:")
        print("   • 自动分析自己的性能")
        print("   • 从历史经验中学习")
        print("   • 定期优化代码和策略")
        print("   • 持续提升能力")
    else:
        print(f"\n🧬 自我进化模式: 已禁用")
        print(f"   提示: 使用 --evolution 或 --infinite-evolution 启用")
    
    print("\n" + "=" * 60)
    print(f"🌐 Web 界面: http://localhost:{args.port}")
    if enable_evolution:
        print(f"📊 进化日志: http://localhost:{args.port}/evolution")
    print("=" * 60)
    print("\n按 Ctrl+C 停止服务\n")
    
    # 启动服务
    try:
        uvicorn.run(
            "src.web.web_app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()
