"""
Test that modules can be imported.
"""

def test_import():
    import mace_rl
    import mace_rl.data.fi2010
    import mace_rl.features.microstructure
    import mace_rl.environment.execution_env
    import mace_rl.models.flows
    import mace_rl.models.manifold
    import mace_rl.models.policy
    import mace_rl.models.value
    import mace_rl.training.base
    import mace_rl.training.flow_trainer
    import mace_rl.utils.config
    import mace_rl.utils.logging
    assert True