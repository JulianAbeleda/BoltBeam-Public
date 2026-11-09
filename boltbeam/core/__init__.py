"""Truth, identity, and immutable experiment contracts.

Import the submodule you need (core.canonical, core.experiment, core.facts, core.policy, core.requirements, core.system).
The package itself imports nothing: artifacts.base reads core.canonical, and core.experiment and core.facts read
artifacts.base back, so an eager re-export here made whichever side was imported first fail with a partial module.
"""
