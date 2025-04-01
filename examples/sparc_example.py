"""
SPARC Methodology Example

This script demonstrates how to use the SPARC methodology workflow
to break down a complex task into smaller subtasks handled by
specialized modes.

Usage:
    python examples/sparc_example.py
"""

import os
import sys
import json
from datetime import datetime


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step, description):
    """Print a step in the SPARC workflow."""
    print(f"\n🔹 STEP {step}: {description}")


def print_task_delegation(mode, task):
    """Print a task delegation example."""
    print(f"\n📤 Delegating to {mode} mode:")
    print(f"""
<new_task>
<mode>{mode}</mode>
<message>{task}</message>
</new_task>
""")


def print_completion(result):
    """Print an attempt_completion example."""
    print(f"\n📥 Completing task:")
    print(f"""
<attempt_completion>
<result>
{result}
</result>
</attempt_completion>
""")


def verify_sparc_setup():
    """Verify that SPARC is properly set up."""
    if not os.path.exists('.roomodes'):
        print("❌ ERROR: .roomodes file not found. SPARC setup is incomplete.")
        sys.exit(1)
    
    try:
        with open('.roomodes', 'r') as f:
            json.load(f)
        print("✅ SPARC configuration is valid.")
    except:
        print("❌ ERROR: .roomodes file contains invalid JSON.")
        sys.exit(1)


def simulate_sparc_workflow():
    """Simulate a full SPARC workflow for a feature request."""
    print_header("SPARC METHODOLOGY WORKFLOW EXAMPLE")
    print("\nThis example demonstrates how to break down a feature request using SPARC.\n")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start with feature request
    feature_request = "Implement a prediction leaderboard where users can see top-performing predictions"
    
    print_step(1, "Start with SPARC Orchestrator")
    print(f"\nFeature request: {feature_request}")
    print("\nThe SPARC Orchestrator breaks this down into subtasks for specialized modes:")
    
    # Specification Writer
    print_step(2, "Specification & Pseudocode")
    print_task_delegation("spec-pseudocode", 
        "Create pseudocode for a prediction leaderboard feature that shows top-performing user predictions")
    
    spec_result = """
Pseudocode completed for prediction leaderboard:
- Data model defines Prediction and Leaderboard schemas
- API endpoints for fetching, filtering, and sorting leaderboard data
- UI components with responsive design for desktop and mobile
- TDD test anchors for all core functionality
"""
    print_completion(spec_result)
    
    # Architect
    print_step(3, "Architecture Design")
    print_task_delegation("architect", 
        "Design system architecture for prediction leaderboard feature with data flows and API endpoints")
    
    architect_result = """
Architecture design completed:
- Created data flow diagrams showing prediction data pipeline
- Defined REST API endpoints for leaderboard functionality
- Designed database schema extensions for leaderboard metrics
- Specified caching strategy for performance optimization
"""
    print_completion(architect_result)
    
    # Code Implementation
    print_step(4, "Code Implementation")
    print_task_delegation("code", 
        "Implement leaderboard feature based on pseudocode and architecture design")
    
    code_result = """
Code implementation completed:
- Created leaderboard model with database migrations
- Implemented API endpoints with proper validation
- Built frontend components with responsive design
- Added configuration through environment variables
- All files under 500 lines with modular structure
"""
    print_completion(code_result)
    
    # Testing
    print_step(5, "Test-Driven Development")
    print_task_delegation("tdd", 
        "Create tests for leaderboard feature using TDD methodology")
    
    tdd_result = """
Tests completed:
- Unit tests for leaderboard data processing
- Integration tests for API endpoints
- Frontend component tests
- All tests passing with 92% coverage
"""
    print_completion(tdd_result)
    
    # Security Review
    print_step(6, "Security Review")
    print_task_delegation("security-review", 
        "Perform security audit on leaderboard implementation")
    
    security_result = """
Security review completed:
- No hardcoded secrets found
- Proper input validation implemented
- Rate limiting added to leaderboard API
- Authorization checks verified
"""
    print_completion(security_result)
    
    # Documentation
    print_step(7, "Documentation")
    print_task_delegation("docs-writer", 
        "Create documentation for leaderboard feature")
    
    docs_result = """
Documentation completed:
- API endpoints documented with examples
- Database schema changes documented
- Usage instructions for frontend and backend
- Configuration options detailed
"""
    print_completion(docs_result)
    
    # Integration
    print_step(8, "Integration")
    print_task_delegation("integration", 
        "Integrate leaderboard feature into main application")
    
    integration_result = """
Integration completed:
- Feature merged into main application
- Dependencies properly connected
- Navigation updated to include leaderboard
- Verified working in staging environment
"""
    print_completion(integration_result)
    
    print_header("SPARC WORKFLOW COMPLETE")
    print("\nThe feature has been successfully implemented following the SPARC methodology.")
    print("\nBenefits achieved:")
    print("✅ Modular code with files under 500 lines")
    print("✅ No hardcoded secrets or environment variables")
    print("✅ Comprehensive test coverage")
    print("✅ Clear documentation")
    print("✅ Security validation")
    print("✅ Seamless integration")


if __name__ == "__main__":
    verify_sparc_setup()
    simulate_sparc_workflow()