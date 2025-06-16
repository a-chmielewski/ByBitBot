#!/usr/bin/env python3
"""
Strategy Test Runner

This script runs all strategy tests and provides a comprehensive test report.
It can be used for continuous integration and automated testing.
"""

import unittest
import sys
import os
from pathlib import Path
import time
import logging

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def discover_and_run_strategy_tests():
    """Discover and run all strategy tests."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    print("🧪 Cryptocurrency Trading Bot - Strategy Test Suite")
    print("=" * 60)
    print(f"Test directory: {tests_dir}")
    print()
    
    # Discover all test files that start with 'test_' and contain 'strategy'
    strategy_test_files = []
    for test_file in tests_dir.glob('test_*strategy*.py'):
        if test_file.name != 'test_strategy_generator.py':  # Skip the generator
            strategy_test_files.append(test_file)
    
    print(f"Found {len(strategy_test_files)} strategy test files:")
    for test_file in strategy_test_files:
        print(f"  📄 {test_file.name}")
    print()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load tests from strategy test files
    for test_file in strategy_test_files:
        try:
            # Convert file path to module name
            module_name = test_file.stem
            
            # Load the test module
            spec = __import__(module_name)
            
            # Discover tests in the module
            module_tests = loader.loadTestsFromModule(spec)
            suite.addTests(module_tests)
            
        except ImportError as e:
            print(f"⚠️  Warning: Could not import {test_file.name}: {e}")
        except Exception as e:
            print(f"❌ Error loading tests from {test_file.name}: {e}")
    
    # Create a test runner with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True,  # Buffer stdout/stderr during tests
        descriptions=True,
        failfast=False  # Continue running tests even if some fail
    )
    
    print("🚀 Running Strategy Tests...")
    print("-" * 60)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests Run: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"🚨 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"⏱️  Execution Time: {end_time - start_time:.2f} seconds")
    
    if failures > 0:
        print(f"\n🔍 FAILURE DETAILS:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print(f"   {traceback.strip()}")
    
    if errors > 0:
        print(f"\n🚨 ERROR DETAILS:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print(f"   {traceback.strip()}")
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = (passed / total_tests) * 100
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 Excellent! Most tests are passing.")
        elif success_rate >= 75:
            print("👍 Good! Most tests are passing.")
        elif success_rate >= 50:
            print("⚠️  Some issues detected. Consider reviewing failed tests.")
        else:
            print("🚨 Many tests are failing. Review and fix issues.")
    else:
        print("⚠️  No tests were run.")
    
    print("\n" + "=" * 60)
    
    # Return whether all tests passed
    return total_tests > 0 and failures == 0 and errors == 0

def run_specific_strategy_test(strategy_name: str):
    """Run tests for a specific strategy."""
    
    test_file_name = f"test_{strategy_name}.py"
    test_file_path = Path(__file__).parent / test_file_name
    
    if not test_file_path.exists():
        print(f"❌ Test file not found: {test_file_name}")
        return False
    
    print(f"🧪 Running tests for {strategy_name}")
    print("-" * 40)
    
    # Load and run the specific test
    loader = unittest.TestLoader()
    
    try:
        # Import the test module
        module_name = test_file_path.stem
        spec = __import__(module_name)
        
        # Load tests
        suite = loader.loadTestsFromModule(spec)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Return success status
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running tests for {strategy_name}: {e}")
        return False

def list_available_strategy_tests():
    """List all available strategy tests."""
    
    tests_dir = Path(__file__).parent
    strategy_test_files = []
    
    for test_file in tests_dir.glob('test_*strategy*.py'):
        if test_file.name != 'test_strategy_generator.py':
            # Extract strategy name from test file name
            strategy_name = test_file.stem.replace('test_', '')
            strategy_test_files.append(strategy_name)
    
    print("📋 Available Strategy Tests:")
    for i, strategy_name in enumerate(sorted(strategy_test_files), 1):
        print(f"  {i}. {strategy_name}")
    
    return strategy_test_files

def main():
    """Main function with command line interface."""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list':
            list_available_strategy_tests()
            
        elif command == 'run' and len(sys.argv) > 2:
            strategy_name = sys.argv[2]
            success = run_specific_strategy_test(strategy_name)
            sys.exit(0 if success else 1)
            
        elif command == 'help':
            print("Strategy Test Runner")
            print("Usage:")
            print("  python run_all_strategy_tests.py           - Run all strategy tests")
            print("  python run_all_strategy_tests.py list      - List available tests")
            print("  python run_all_strategy_tests.py run <name> - Run specific strategy test")
            print("  python run_all_strategy_tests.py help      - Show this help")
            
        else:
            print("❌ Invalid command. Use 'help' for usage information.")
            sys.exit(1)
    else:
        # Run all tests
        success = discover_and_run_strategy_tests()
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 