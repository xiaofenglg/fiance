#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10 -> V11 迁移脚本

功能:
- 检查 V10 代码结构
- 验证 V11 模块完整性
- 生成迁移报告
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent


def check_v10_structure():
    """检查 V10 结构"""
    v10_dir = project_root / "V10"

    required_files = [
        "factors/factor_engine.py",
        "models/lgbm_signal.py",
        "models/lstm_signal.py",
        "models/ensemble_aggregator.py",
        "risk/cvar_controller.py",
        "risk/atr_stop_loss.py",
        "position/kelly_sizer.py",
    ]

    print("=== V10 Structure Check ===")

    missing = []
    for f in required_files:
        path = v10_dir / f
        if path.exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            missing.append(f)

    return len(missing) == 0


def check_v11_structure():
    """检查 V11 结构"""
    src_dir = project_root / "src" / "aifinance"

    required_modules = [
        ("__init__.py", "Main module"),
        ("data/__init__.py", "Data module"),
        ("data/nav_loader.py", "NAV loader"),
        ("data/unsmoothing.py", "GLM unsmoothing"),
        ("data/macro_loader.py", "Macro data"),
        ("data/qlib_provider.py", "Qlib integration"),
        ("factors/__init__.py", "Factors module"),
        ("factors/factor_engine.py", "Factor engine"),
        ("models/__init__.py", "Models module"),
        ("models/tft_signal.py", "TFT model"),
        ("models/lgbm_signal.py", "LightGBM model"),
        ("models/ensemble.py", "Model ensemble"),
        ("portfolio/__init__.py", "Portfolio module"),
        ("portfolio/hrp_optimizer.py", "HRP optimizer"),
        ("portfolio/risk_control.py", "Risk control"),
        ("portfolio/position_sizer.py", "Position sizing"),
        ("backtest/__init__.py", "Backtest module"),
        ("backtest/engine.py", "Backtest engine"),
        ("utils/__init__.py", "Utils module"),
        ("utils/logging_config.py", "Logging config"),
    ]

    print("\n=== V11 Structure Check ===")

    missing = []
    for f, desc in required_modules:
        path = src_dir / f
        if path.exists():
            print(f"  [OK] {f} ({desc})")
        else:
            print(f"  [MISSING] {f} ({desc})")
            missing.append(f)

    return len(missing) == 0


def check_pyproject():
    """检查 pyproject.toml"""
    pyproject = project_root / "pyproject.toml"

    print("\n=== pyproject.toml Check ===")

    if pyproject.exists():
        print("  [OK] pyproject.toml exists")

        content = pyproject.read_text(encoding="utf-8")

        checks = [
            ('version = "11.0.0"', "Version 11.0.0"),
            ("neuralforecast", "neuralforecast dependency"),
            ("riskfolio-lib", "riskfolio-lib dependency"),
            ("torch", "torch dependency"),
            ("lightgbm", "lightgbm dependency"),
        ]

        for pattern, desc in checks:
            if pattern in content:
                print(f"  [OK] {desc}")
            else:
                print(f"  [MISSING] {desc}")

        return True
    else:
        print("  [MISSING] pyproject.toml does not exist")
        return False


def check_imports():
    """检查模块导入"""
    print("\n=== Module Import Check ===")

    sys.path.insert(0, str(project_root / "src"))

    modules_to_check = [
        "aifinance",
        "aifinance.data",
        "aifinance.factors",
        "aifinance.models",
        "aifinance.portfolio",
        "aifinance.backtest",
        "aifinance.utils",
    ]

    success = True
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError as e:
            print(f"  [FAIL] {module}: {e}")
            success = False

    return success


def generate_migration_report():
    """生成迁移报告"""
    print("\n" + "=" * 60)
    print("               V10 -> V11 Migration Report")
    print("=" * 60)

    v10_ok = check_v10_structure()
    v11_ok = check_v11_structure()
    pyproject_ok = check_pyproject()
    import_ok = check_imports()

    print("\n" + "=" * 60)
    print("                    Summary")
    print("=" * 60)
    print(f"  V10 Structure:   {'[OK] Complete' if v10_ok else '[FAIL] Missing files'}")
    print(f"  V11 Structure:   {'[OK] Complete' if v11_ok else '[FAIL] Missing files'}")
    print(f"  pyproject.toml:  {'[OK] Correct' if pyproject_ok else '[FAIL] Needs check'}")
    print(f"  Module imports:  {'[OK] Success' if import_ok else '[FAIL] Issues found'}")

    all_ok = v10_ok and v11_ok and pyproject_ok and import_ok

    print("\n" + "-" * 60)
    if all_ok:
        print("Migration Status: [OK] Complete")
        print("\nNext Steps:")
        print("  1. Run tests: pytest tests/")
        print("  2. Install package: pip install -e .")
        print("  3. Run backtest: python scripts/run_backtest.py")
    else:
        print("Migration Status: [FAIL] Needs fixing")
        print("\nPlease check the errors above and fix them.")

    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = generate_migration_report()
    sys.exit(0 if success else 1)
