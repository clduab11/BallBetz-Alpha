# SPARC Methodology Guide

## Overview

SPARC stands for Specification, Pseudocode, Architecture, Refinement, and Completion. This methodology enables breaking down complex projects into manageable subtasks using specialized modes.

## Core Principles

1. **Modular Design**: All files should be under 500 lines
2. **Secure Configuration**: No hard-coded environment variables 
3. **Testable Components**: Each module should have clear test anchors
4. **Delegation**: Using `new_task` to assign specialized subtasks
5. **Completion**: Every subtask ends with `attempt_completion`

## Available Modes

- **⚡️ SPARC Orchestrator** (`sparc`): Breaks down objectives into delegated subtasks
- **📋 Specification Writer** (`spec-pseudocode`): Creates modular pseudocode with TDD anchors
- **🏗️ Architect** (`architect`): Designs scalable, secure, and modular architectures
- **🧠 Auto-Coder** (`code`): Writes clean, efficient, modular code
- **🧪 Tester** (`tdd`): Implements Test-Driven Development
- **🪲 Debugger** (`debug`): Troubleshoots runtime bugs and integration failures
- **🛡️ Security Reviewer** (`security-review`): Performs security audits
- **📚 Documentation Writer** (`docs-writer`): Creates clear, modular Markdown documentation
- **🔗 System Integrator** (`integration`): Merges outputs from all modes
- **📈 Deployment Monitor** (`post-deployment-monitoring-mode`): Observes systems post-launch
- **🧹 Optimizer** (`refinement-optimization-mode`): Refactors and improves system performance
- **❓ Ask** (`ask`): Helps with task formulation and delegation
- **🚀 DevOps** (`devops`): Manages deployments and infrastructure
- **📘 SPARC Tutorial** (`tutorial`): Guides users through the SPARC development process

## SPARC Workflow

1. **Start with SPARC Orchestrator**: Begin by delegating tasks to specialized modes
2. **Specification & Pseudocode**: Define requirements and create modular pseudocode
3. **Architecture**: Design the system with clear boundaries and integration points
4. **Implementation**: Code the solution following best practices
5. **Testing**: Follow TDD principles for robust code
6. **Refinement**: Debug, secure, and optimize
7. **Completion**: Integrate, document, and monitor

## Using SPARC with Roo Code Boomerang

To use SPARC methodology effectively:

1. Select "SPARC Orchestrator" as your primary mode
2. Use `new_task` to delegate to specialized modes
3. Each mode completes its task via `attempt_completion`
4. Maintain isolated contexts for each subtask
5. Ensure all code follows modular, secure principles
6. Monitor and optimize continuously

## Example Task Delegation

```
<new_task>
<mode>spec-pseudocode</mode>
<message>Create a pseudocode specification for user authentication flow with email verification</message>
</new_task>
```

## Best Practices

- Keep files under 500 lines
- Never hardcode environment variables or secrets
- Use modular, testable design patterns
- Document clearly with consistent standards
- Implement proper error handling and logging
- Follow security best practices throughout