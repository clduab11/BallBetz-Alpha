#!/usr/bin/env python3
"""
Manual Test Runner for Hello SPARC

This script performs basic tests of the Hello SPARC application
without requiring external testing packages. It demonstrates
basic SPARC methodology testing principles.
"""

import os
import sys
import traceback
from typing import Callable, Dict, List, Any

# Add the parent directory to sys.path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the modules to test
from hello_sparc.config import Config
from hello_sparc.utils.formatter import format_greeting, format_colored_text, wrap_in_box
from hello_sparc.app import Greeter


def run_test(name: str, test_func: Callable[[], Any]) -> bool:
    """Run a test and report the result."""
    print(f"Running test: {name}...")
    try:
        result = test_func()
        print(f"✅ PASS: {name}")
        return True
    except AssertionError as e:
        print(f"❌ FAIL: {name} - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {name} - Unexpected error: {e}")
        traceback.print_exc()
        return False


def test_default_config() -> None:
    """Test default configuration values."""
    config = Config()
    assert config.app_name == 'Hello SPARC', f"Expected 'Hello SPARC', got '{config.app_name}'"
    assert config.debug_mode is False, f"Expected debug_mode=False, got {config.debug_mode}"
    assert '{name}' in config.greeting_template, f"Missing '{{name}}' in template: '{config.greeting_template}'"
    assert config.default_name == 'World', f"Expected default_name='World', got '{config.default_name}'"


def test_env_override() -> None:
    """Test environment variable overrides."""
    # Save original environment
    original_env = {}
    for key in ['APP_NAME', 'DEBUG_MODE', 'GREETING_TEMPLATE', 'DEFAULT_NAME']:
        original_env[key] = os.environ.get(key)
    
    try:
        # Set test environment variables
        os.environ['APP_NAME'] = 'Test App'
        os.environ['DEBUG_MODE'] = 'true'
        os.environ['GREETING_TEMPLATE'] = 'Hey there, {name}!'
        os.environ['DEFAULT_NAME'] = 'Tester'
        
        # Create config with new environment
        config = Config()
        assert config.app_name == 'Test App', f"Expected 'Test App', got '{config.app_name}'"
        assert config.debug_mode is True, f"Expected debug_mode=True, got {config.debug_mode}"
        assert config.greeting_template == 'Hey there, {name}!', f"Wrong template: '{config.greeting_template}'"
        assert config.default_name == 'Tester', f"Expected default_name='Tester', got '{config.default_name}'"
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value


def test_format_greeting() -> None:
    """Test greeting formatter."""
    result = format_greeting("Hello, {name}!", "SPARC")
    assert result == "Hello, SPARC!", f"Expected 'Hello, SPARC!', got '{result}'"


def test_format_greeting_missing_placeholder() -> None:
    """Test greeting formatter with missing placeholder."""
    result = format_greeting("Hello there!", "SPARC")
    assert result == "Hello, SPARC!", f"Expected fallback 'Hello, SPARC!', got '{result}'"


def test_wrap_in_box() -> None:
    """Test box wrapper."""
    result = wrap_in_box("Test")
    assert "┌" in result, "Box missing top-left corner"
    assert "┐" in result, "Box missing top-right corner"
    assert "└" in result, "Box missing bottom-left corner"
    assert "┘" in result, "Box missing bottom-right corner"
    assert "Test" in result, "Box missing content"


def test_greeter() -> None:
    """Test the Greeter class."""
    # Create a test config
    os.environ['GREETING_TEMPLATE'] = "Hello, {name}!"
    os.environ['DEFAULT_NAME'] = "World"
    
    config = Config()
    greeter = Greeter(config)
    
    # Test with custom name
    result = greeter.generate_greeting("SPARC")
    assert result == "Hello, SPARC!", f"Expected 'Hello, SPARC!', got '{result}'"
    
    # Test with default name
    result = greeter.generate_greeting()
    assert result == "Hello, World!", f"Expected 'Hello, World!', got '{result}'"


def run_all_tests() -> None:
    """Run all tests and report results."""
    print("\n🔍 Running Hello SPARC Manual Tests\n")
    
    tests: Dict[str, Callable[[], None]] = {
        "Default Configuration": test_default_config,
        "Environment Variable Override": test_env_override,
        "Format Greeting": test_format_greeting,
        "Format Greeting with Missing Placeholder": test_format_greeting_missing_placeholder,
        "Wrap in Box": test_wrap_in_box,
        "Greeter Class": test_greeter
    }
    
    results = []
    for name, test_func in tests.items():
        results.append(run_test(name, test_func))
    
    # Print summary
    print("\n📊 Test Summary")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {results.count(True)}")
    print(f"Failed: {results.count(False)}")
    
    if all(results):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed.")


if __name__ == "__main__":
    run_all_tests()