#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HISENSHO - Dual Verification Audit System (双重验证审计系统)

Automatically audits git diff changes through Gemini CLI with role-based prompts.

Usage:
    python hisensho.py                    # Auto-detect role from changed files
    python hisensho.py --role quant       # Force Quant role
    python hisensho.py --role security    # Force Security role
    python hisensho.py --role architect   # Force Architect role
    python hisensho.py --role fullstack   # Force Full-stack Developer role
    python hisensho.py --file path/to/file.py  # Audit specific file
    python hisensho.py --staged           # Only audit staged changes
    python hisensho.py --loop             # Keep auditing until PASS

Roles:
    quant     - 量化专家: 数据精度、反爬虫、逻辑严密、时序对齐、前视偏差
    security  - 安全专家: 漏洞、注入、密钥泄露、认证问题
    architect - 架构师: 解耦、性能、可维护性、设计模式
    fullstack - 全栈开发: 代码质量、最佳实践、错误处理
"""

import argparse
import subprocess
import sys
import os
import re
from pathlib import Path
from typing import Optional, Tuple

# Role definitions with Chinese prompts
ROLES = {
    "quant": {
        "name": "量化专家 (Quant)",
        "prompt": """你是量化专家(Quant)，请审计以下代码变更。
关注点：
- 数据精度和数值稳定性
- 反爬虫和数据获取逻辑
- 时序对齐（确保没有未来数据泄露）
- 前视偏差（look-ahead bias）
- 回测逻辑严密性
- 金融计算正确性（收益率、夏普比率、回撤等）

如果代码没有问题，请回复 PASS。
如果有问题，请指出具体修改建议。""",
        "keywords": ["backtest", "nav", "return", "sharpe", "drawdown", "portfolio",
                    "quant", "finance", "trading", "crawl", "spider", "scrape",
                    "price", "fee", "rate", "bank", "fund", "asset"]
    },
    "security": {
        "name": "安全专家 (Security)",
        "prompt": """你是安全专家(Security)，请审计以下代码变更。
关注点：
- SQL注入、XSS、CSRF等OWASP Top 10漏洞
- 命令注入和路径遍历
- 密钥和凭证泄露（API keys, passwords）
- 认证和授权问题
- 加密实现正确性
- 敏感数据处理

如果代码没有问题，请回复 PASS。
如果有问题，请指出具体修改建议。""",
        "keywords": ["auth", "login", "password", "token", "jwt", "oauth", "api_key",
                    "secret", "credential", "encrypt", "decrypt", "hash", "ssl", "tls",
                    "certificate", "permission", "role", "session", "cookie"]
    },
    "architect": {
        "name": "架构师 (Architect)",
        "prompt": """你是软件架构师(Architect)，请审计以下代码变更。
关注点：
- 模块解耦和依赖管理
- 性能和可扩展性
- 代码可维护性和可读性
- 设计模式的正确使用
- 接口设计和抽象层次
- 错误处理和边界情况

如果代码没有问题，请回复 PASS。
如果有问题，请指出具体修改建议。""",
        "keywords": ["refactor", "architect", "design", "pattern", "interface", "abstract",
                    "module", "component", "service", "layer", "dependency", "injection",
                    "factory", "singleton", "observer", "strategy"]
    },
    "fullstack": {
        "name": "全栈开发专家 (Senior Full-stack Developer)",
        "prompt": """你是资深全栈开发专家(Senior Full-stack Developer)，请审计以下代码变更。
关注点：
- 代码质量和最佳实践
- 错误处理和异常管理
- 代码复用和DRY原则
- 命名规范和代码风格
- 测试覆盖和可测试性
- 文档和注释

如果代码没有问题，请回复 PASS。
如果有问题，请指出具体修改建议。""",
        "keywords": []  # Default role, matches everything else
    }
}

# File pattern to role mapping
FILE_PATTERNS = {
    "quant": [
        r"backtest", r"nav", r"portfolio", r"signal", r"factor", r"strategy",
        r"crawl", r"spider", r"scrape", r"finance", r"trading", r"hrp", r"tft"
    ],
    "security": [
        r"auth", r"login", r"user", r"permission", r"token", r"session",
        r"crypto", r"encrypt", r"secret", r"password", r"credential"
    ],
    "architect": [
        r"config", r"setting", r"base", r"core", r"engine", r"manager",
        r"factory", r"builder", r"interface", r"abstract"
    ]
}


MAX_DIFF_SIZE = 50000  # Max characters to send to Gemini (avoid OOM)


def get_git_diff(staged_only: bool = False, file_path: Optional[str] = None) -> str:
    """Get git diff output."""
    cmd = ["git", "diff"]
    if staged_only:
        cmd.append("--staged")
    if file_path:
        cmd.append("--")
        cmd.append(file_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        diff = result.stdout

        # Truncate if too large
        if len(diff) > MAX_DIFF_SIZE:
            diff = diff[:MAX_DIFF_SIZE] + f"\n\n... [TRUNCATED - diff too large ({len(result.stdout)} chars)] ..."

        return diff
    except Exception as e:
        print(f"Error running git diff: {e}")
        return ""


def get_changed_files(staged_only: bool = False) -> list:
    """Get list of changed files."""
    cmd = ["git", "diff", "--name-only"]
    if staged_only:
        cmd.append("--staged")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return files
    except Exception as e:
        print(f"Error getting changed files: {e}")
        return []


def detect_role(files: list, diff_content: str) -> str:
    """Auto-detect the appropriate role based on changed files and content."""
    # Check file patterns
    for role, patterns in FILE_PATTERNS.items():
        for file in files:
            file_lower = file.lower()
            for pattern in patterns:
                if re.search(pattern, file_lower):
                    return role

    # Check content keywords
    diff_lower = diff_content.lower()
    for role, config in ROLES.items():
        if role == "fullstack":
            continue  # Skip default role
        for keyword in config["keywords"]:
            if keyword in diff_lower:
                return role

    # Default to fullstack
    return "fullstack"


def find_gemini_cli() -> Optional[str]:
    """Find the gemini CLI executable path."""
    # Try common locations on Windows
    possible_paths = [
        "gemini",
        "gemini.cmd",
        r"C:\nodejs-new\node-v20.18.1-win-x64\gemini.cmd",
        r"C:\nodejs\node-v20.11.0-win-x64\gemini.cmd",
    ]

    for path in possible_paths:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                shell=True,
                timeout=5
            )
            if result.returncode == 0:
                return path
        except:
            continue

    return None


def run_gemini_audit(diff_content: str, role: str) -> Tuple[bool, str]:
    """Run Gemini CLI audit and return (passed, response)."""
    if not diff_content.strip():
        return True, "No changes to audit."

    prompt = ROLES[role]["prompt"]

    try:
        # Find gemini CLI
        gemini_cmd = find_gemini_cli()
        if not gemini_cmd:
            # Fallback: try with shell=True
            gemini_cmd = "gemini"

        # Run gemini CLI with the diff piped in
        # Use shell=True on Windows to find the command
        process = subprocess.Popen(
            f'{gemini_cmd} "{prompt}"',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=True
        )
        stdout, stderr = process.communicate(input=diff_content, timeout=120)

        response = stdout.strip()

        # If stdout is empty, check stderr
        if not response and stderr:
            response = stderr.strip()

        # More robust PASS detection (English + Chinese)
        response_upper = response.upper()
        passed = (
            "APPROVED" in response_upper or
            "PASSED" in response_upper or
            "通过" in response or
            any(
                line.strip().upper().startswith("PASS") or
                "[PASS" in line.upper() or
                "**PASS" in line.upper() or
                "PASS]" in line.upper()
                for line in response.splitlines()
            )
        )

        return passed, response

    except subprocess.TimeoutExpired:
        process.kill()
        return False, "ERROR: Gemini CLI timeout (120s)"
    except FileNotFoundError:
        return False, "ERROR: Gemini CLI not found. Please install it first."
    except Exception as e:
        return False, f"ERROR: {str(e)}"


def print_header(role: str):
    """Print audit header."""
    role_name = ROLES[role]["name"]
    print("\n" + "=" * 60)
    print(f"  HISENSHO - Dual Verification Audit System")
    print(f"  Role: {role_name}")
    print("=" * 60)


def print_result(passed: bool, response: str):
    """Print audit result."""
    # Handle Unicode encoding for Windows console
    try:
        safe_response = response.encode('gbk', errors='replace').decode('gbk')
    except:
        safe_response = response.encode('ascii', errors='replace').decode('ascii')

    print("\n" + "-" * 60)
    if passed:
        print("  RESULT: PASS")
        print("-" * 60)
        print(safe_response)
    else:
        print("  RESULT: NEEDS REVISION")
        print("-" * 60)
        print(safe_response)
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="HISENSHO - Dual Verification Audit System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--role", "-r",
        choices=["quant", "security", "architect", "fullstack", "auto"],
        default="auto",
        help="Audit role (default: auto-detect)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Specific file to audit"
    )
    parser.add_argument(
        "--staged", "-s",
        action="store_true",
        help="Only audit staged changes"
    )
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        help="Keep prompting until PASS (interactive mode)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only show result"
    )

    args = parser.parse_args()

    # Get changed files
    files = get_changed_files(args.staged)
    if not files and not args.file:
        print("No changes detected. Nothing to audit.")
        return 0

    # Get diff content
    diff_content = get_git_diff(args.staged, args.file)
    if not diff_content.strip():
        print("No diff content. Nothing to audit.")
        return 0

    # Determine role
    if args.role == "auto":
        role = detect_role(files, diff_content)
    else:
        role = args.role

    # Print header
    if not args.quiet:
        print_header(role)
        print(f"\nFiles to audit:")
        for f in (files if not args.file else [args.file]):
            print(f"  - {f}")
        print(f"\nDiff size: {len(diff_content)} characters")

    # Run audit loop
    iteration = 1
    while True:
        if not args.quiet:
            print(f"\n>>> Running Gemini audit (iteration {iteration})...")

        passed, response = run_gemini_audit(diff_content, role)

        print_result(passed, response)

        if passed:
            print("Audit PASSED. You may proceed with commit.")
            return 0

        if not args.loop:
            print("Audit requires revisions. Fix the issues and run again.")
            return 1

        # Interactive loop mode
        print("\nFix the issues above, then press Enter to re-audit (or 'q' to quit):")
        user_input = input().strip().lower()
        if user_input == 'q':
            print("Audit cancelled.")
            return 1

        # Refresh diff content
        diff_content = get_git_diff(args.staged, args.file)
        if not diff_content.strip():
            print("No more changes. Audit complete.")
            return 0

        iteration += 1


if __name__ == "__main__":
    sys.exit(main())
