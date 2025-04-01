#!/usr/bin/env python3
"""
Hello SPARC Example Runner

This script provides an easy way to run the Hello SPARC example
application directly from the project root directory.

Usage:
    python examples/run_hello_sparc.py [name]
"""

import os
import sys
import argparse


def setup_paths():
    """Set up proper import paths."""
    # Add the examples directory to the Python path
    examples_dir = os.path.abspath(os.path.dirname(__file__))
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Hello SPARC Example')
    parser.add_argument('name', nargs='?', default=None,
                        help='Name to use in greeting (default: from config)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--color', default=None,
                        help='Set the text color')
    
    return parser.parse_args()


def set_env_vars(args):
    """Set environment variables based on command line arguments."""
    if args.debug:
        os.environ['DEBUG_MODE'] = 'true'
    
    if args.color:
        os.environ['TEXT_COLOR'] = args.color


def main():
    """Main entry point."""
    # Set up import paths
    setup_paths()
    
    # Parse arguments
    args = parse_args()
    
    # Set environment variables based on arguments
    set_env_vars(args)
    
    # Import the app module (after setting up paths)
    from hello_sparc.app import main as app_main
    
    # Run the app with any provided name
    sys_args = []
    if args.name:
        sys_args.append(args.name)
    
    # Execute the app
    exit_code = app_main(sys_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()