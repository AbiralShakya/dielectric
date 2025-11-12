import pytest

from src.backend.agents.intent_agent import IntentAgent


def test_percentage_weights_parsing():
    agent = IntentAgent()
    # Use fallback directly to avoid API calls
    alpha, beta, gamma = agent._intent_to_weights_fallback(
        "Optimize for: Signal integrity (40%), Thermal management (30%), Manufacturability (20%), Cost (10%)",
        geometry_data=None
    )
    # Expect roughly 0.4 / 0.3 / 0.1 after splitting manufacturability between gamma and delta
    assert abs(alpha - 0.4) < 0.05
    assert abs(beta - 0.3) < 0.05
    assert abs(gamma - 0.1) < 0.05

