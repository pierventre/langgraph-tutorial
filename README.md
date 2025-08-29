# LangGraph Tutorial

This project includes examples of chaining, augmentation, parallelization, and routing workflows using LangGraph and LangChain. The examples are adapted from the tutorial of LangChain academy.

## Project Structure

- `main_augmentation.py` — Example of augmenting workflows with additional context or steps.
- `main_chaining.py` — Demonstrates chaining multiple steps in a workflow.
- `main_parallelization.py` — Shows how to run steps in parallel and aggregate results.
- `main_routing.py` — Example of routing logic to different workflow branches based on conditions.
- `main_orchestrator.py` — Orchestrates complex workflows with planning and worker nodes.
- `chaining_workflow.png`, `parallel_workflow.png` — Visual diagrams of workflow structures.
- `pyproject.toml` — Project dependencies and configuration.

## Getting Started

1. **Install dependencies**

   ```bash
   uv pip install -r requirements.txt
   ```

   or use your preferred Python environment manager.

2. **Run Examples**

   You can run any of the workflow scripts directly:

   ```bash
   uv run python -m main_chaining
   uv run python -m main_augmentation
   uv run python -m main_parallelization
   uv run python -m main_routing
   uv run python -m main_orchestrator
   ```

## Concepts Demonstrated

- **State Management:** How to define and share state between workflow nodes.
- **Parallel Execution:** Running multiple tasks in parallel and merging results.
- **Conditional Routing:** Directing workflow execution based on dynamic conditions.
- **Tool Integration:** Using external tools and APIs within workflow steps.
- **Planning and Orchestration:** Building multi-step plans and coordinating worker nodes.

## Requirements

- Python 3.10+
- LangGraph
- LangChain
- (Optional) Google Generative AI API key for LLM examples

## Visualizations

Workflow diagrams are provided in the PNG files to help you understand the structure and flow of each example.

## License

This project is licensed under the MIT License.

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

---

Feel free to explore and modify the examples to fit your own AI workflow needs!
