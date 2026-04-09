"""Machine learning modules for market regime detection and RL portfolio allocation."""

__version__ = "0.1.0"

__all__: list[str] = []

try:
    from .explainability_plotly import (
        compute_feature_saliency_from_states,
        create_attention_diagnostics_figure,
        create_finance_attention_heads_figure,
        create_attention_sankey_figure,
        create_feature_explainability_figure,
        default_state_feature_labels,
        make_time_token_labels,
        validate_attention_inputs,
    )
except ModuleNotFoundError:
    # Keep the package importable when optional explainability dependencies
    # such as torch are absent. Submodules can still be imported directly.
    pass
else:
    __all__.extend(
        [
            "compute_feature_saliency_from_states",
            "create_attention_diagnostics_figure",
            "create_finance_attention_heads_figure",
            "create_attention_sankey_figure",
            "create_feature_explainability_figure",
            "default_state_feature_labels",
            "make_time_token_labels",
            "validate_attention_inputs",
        ]
    )
