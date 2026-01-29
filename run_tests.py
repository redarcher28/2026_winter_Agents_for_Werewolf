"""测试运行脚本"""
import subprocess
import sys
import os


def run_tests():
    """运行所有测试"""
    test_files = [
        "test_config.py",
        "test_memory.py",
        "test_communication.py",
        "test_llm.py",
        "test_integration.py"
    ]

    results = []

    for test_file in test_files:
        print(f"\n{'=' * 60}")
        print(f"运行测试: {test_file}")
        print('=' * 60)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✅ 测试通过")
                results.append((test_file, True, result.stdout))
            else:
                print("❌ 测试失败")
                print(result.stdout)
                print(result.stderr)
                results.append((test_file, False, result.stderr))

        except Exception as e:
            print(f"⚠️  测试执行错误: {e}")
            results.append((test_file, False, str(e)))

    # 生成测试报告
    print(f"\n{'=' * 60}")
    print("测试报告汇总")
    print('=' * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"总测试文件数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {passed / total * 100:.1f}%")

    if passed < total:
        print("\n失败的测试:")
        for test_file, success, error in results:
            if not success:
                print(f"  - {test_file}: {error[:100]}...")
        sys.exit(1)
    else:
        print("\n🎉 所有测试通过！")
        sys.exit(0)


if __name__ == "__main__":
    # 设置环境变量用于测试
    os.environ["OPENAI_API_KEY"] = "test_key_for_testing"

    # 运行测试
    run_tests()