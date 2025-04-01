"""
Test suite for Hello SPARC application.

This demonstrates a Test-Driven Development (TDD) approach as part of
the SPARC methodology, with tests for all core components.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to make imports work
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import modules to test
from hello_sparc.config import Config
from hello_sparc.utils.formatter import format_greeting, format_colored_text, wrap_in_box
from hello_sparc.app import Greeter


class TestConfig:
    """Tests for the configuration module."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.app_name == 'Hello SPARC'
        assert config.debug_mode is False
        assert '{name}' in config.greeting_template
        assert config.default_name == 'World'
    
    def test_env_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            'APP_NAME': 'Test App',
            'DEBUG_MODE': 'true',
            'GREETING_TEMPLATE': 'Hey there, {name}!',
            'DEFAULT_NAME': 'Tester'
        }):
            config = Config()
            assert config.app_name == 'Test App'
            assert config.debug_mode is True
            assert config.greeting_template == 'Hey there, {name}!'
            assert config.default_name == 'Tester'
    
    def test_validation(self):
        """Test configuration validation."""
        with patch.dict(os.environ, {'GREETING_TEMPLATE': 'Invalid template'}):
            with pytest.raises(ValueError):
                Config()


class TestFormatter:
    """Tests for the formatter utilities."""
    
    def test_format_greeting(self):
        """Test greeting formatter."""
        result = format_greeting("Hello, {name}!", "SPARC")
        assert result == "Hello, SPARC!"
    
    def test_format_greeting_missing_placeholder(self):
        """Test greeting formatter with missing placeholder."""
        result = format_greeting("Hello there!", "SPARC")
        assert result == "Hello, SPARC!"
    
    def test_format_colored_text(self):
        """Test colored text formatter."""
        result = format_colored_text("Hello", "blue")
        assert result.startswith('\033[34m')
        assert result.endswith('\033[0m')
        assert "Hello" in result
    
    def test_format_colored_text_invalid_color(self):
        """Test colored text formatter with invalid color."""
        result = format_colored_text("Hello", "not_a_color")
        assert '\033[0m' in result  # Should use reset color
    
    def test_wrap_in_box(self):
        """Test box wrapper."""
        result = wrap_in_box("Test")
        assert "┌" in result  # Top-left corner
        assert "┐" in result  # Top-right corner
        assert "└" in result  # Bottom-left corner
        assert "┘" in result  # Bottom-right corner
        assert "Test" in result  # Content


class TestGreeter:
    """Tests for the Greeter class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.app_name = "Test App"
        config.greeting_template = "Hello, {name}!"
        config.default_name = "Tester"
        config.text_color = "blue"
        config.debug_mode = False
        return config
    
    def test_generate_greeting(self, mock_config):
        """Test greeting generation."""
        greeter = Greeter(mock_config)
        result = greeter.generate_greeting("SPARC")
        assert result == "Hello, SPARC!"
    
    def test_generate_greeting_default_name(self, mock_config):
        """Test greeting generation with default name."""
        greeter = Greeter(mock_config)
        result = greeter.generate_greeting()
        assert result == "Hello, Tester!"
    
    @patch('examples.hello_sparc.app.format_colored_text')
    @patch('examples.hello_sparc.app.wrap_in_box')
    def test_display_greeting(self, mock_wrap, mock_color, mock_config, capsys):
        """Test greeting display."""
        # Set up mocks
        mock_wrap.return_value = "[Wrapped Greeting]"
        mock_color.return_value = "[Colored Greeting]"
        
        # Create greeter and display greeting
        greeter = Greeter(mock_config)
        greeter.display_greeting("SPARC")
        
        # Check output
        captured = capsys.readouterr()
        assert "[Colored Greeting]" in captured.out


if __name__ == "__main__":
    pytest.main(["-v"])