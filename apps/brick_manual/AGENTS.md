# Agent Guidance for BrickManual App

This document captures guidance for the AI agent working on the BrickManual Flutter application.

## Core Principles

- **Best Practices:** Always adhere to modern Flutter best practices.
- **Cleanliness & Simplicity:** Strive for code that is clean, simple, and easy to understand.
- **Robustness:** Code should be robust and well-tested.
- **Testability:** Structure code to be easily testable. Prefer patterns like dependency injection and repositories that allow for easy mocking.
- **Standard Patterns:** Use standard, well-established Flutter patterns (e.g., `FutureBuilder` for asynchronous UI) over complex or custom solutions.
- **No Backwards Compatibility Concern:** When refactoring, do not worry about backwards compatibility. Prioritize improving the code quality.
