# Hello SPARC Documentation

This documentation follows the SPARC methodology principles of modularity and clarity.

## Application Overview

Hello SPARC is a simple greeting application that demonstrates SPARC methodology principles including:

- Modular code design
- Configuration via environment variables
- Testable components
- Clean architecture

## Directory Structure

```
hello_sparc/
├── app.py             # Main application entry point
├── config.py          # Configuration handling
├── .env.example       # Example environment variables
├── docs/              # Documentation
│   └── README.md      # This file
├── tests/             # Test modules
│   ├── __init__.py
│   └── test_greeting.py
└── utils/             # Utilities
    ├── __init__.py
    ├── formatter.py   # Text formatting utilities
    └── logger.py      # Logging utilities
```

## Configuration

The application uses environment variables for configuration, which can be set in a `.env` file or directly in the environment:

- `APP_NAME`: Name of the application (default: "Hello SPARC")
- `DEBUG_MODE`: Enable debug output (default: "False")
- `GREETING_TEMPLATE`: Template for greeting messages (default: "Hello, {name}!")
- `DEFAULT_NAME`: Default name to use in greetings (default: "World")
- `TEXT_COLOR`: Color for text output (default: "blue")

## Usage

### Basic Usage

```bash
python examples/hello_sparc/app.py
```

This will display a greeting using the default name from configuration.

### Custom Greeting

```bash
python examples/hello_sparc/app.py "SPARC User"
```

This will display a greeting using the provided name.

### Environment Customization

You can customize the application by setting environment variables:

```bash
DEBUG_MODE=true GREETING_TEMPLATE="Hey there, {name}!" python examples/hello_sparc/app.py
```

## Development

### Running Tests

Tests follow the TDD approach and can be run with pytest:

```bash
pytest examples/hello_sparc/tests/
```

### Adding Features

When adding features, follow these SPARC methodology principles:

1. Create tests first
2. Keep files under 500 lines
3. Use configuration for any variable settings
4. Never hardcode environment variables or secrets
5. Document clearly