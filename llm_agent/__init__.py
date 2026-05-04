"""
LLM-Augmented Reasoning for IoT Network Traffic Classification.

Two modes: **ml_agent** (tree model + SHAP + optional LLM explainer) and **zeroshot** (LLM-only).
"""

from llm_agent.config import AgentConfig, ClassifyMode, load_config

__all__ = ["AgentConfig", "ClassifyMode", "load_config", "__version__"]

__version__ = "0.1.0"
