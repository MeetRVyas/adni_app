# data_pipeline/__init__.py
"""
BDA Data Pipeline — Multi-Disease Medical Imaging.

Public entry point:
    from data_pipeline import run_pipeline
"""
from data_pipeline.pipeline import run_pipeline

__all__ = ["run_pipeline"]
