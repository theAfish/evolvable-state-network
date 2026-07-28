"""Protect the canonical backend package surface."""

import unittest

from evolvable_state_network.application import ApplicationRuntime
from evolvable_state_network.application.runtime import (
    ApplicationRuntime as RuntimeImplementation,
)
from evolvable_state_network.evolution import (
    AsyncEvolutionRunner,
    CandidateEvaluator,
    EvolutionRunner,
    GenomeCodec,
    RuleArchitecture,
)
from evolvable_state_network.evolution.asynchronous import (
    AsyncEvolutionRunner as AsynchronousImplementation,
)
from evolvable_state_network.evolution.candidate import (
    RuleArchitecture as ArchitectureImplementation,
)
from evolvable_state_network.evolution.evaluation import (
    CandidateEvaluator as EvaluatorImplementation,
)
from evolvable_state_network.evolution.generational import (
    EvolutionRunner as GenerationalImplementation,
)
from evolvable_state_network.evolution.genome import GenomeCodec as CodecImplementation
from evolvable_state_network.simulation import Simulation
from evolvable_state_network.simulation.engine import Simulation as SimulationImplementation


class PackageLayoutTests(unittest.TestCase):
    def test_package_exports_resolve_to_canonical_implementations(self) -> None:
        self.assertIs(ApplicationRuntime, RuntimeImplementation)
        self.assertIs(AsyncEvolutionRunner, AsynchronousImplementation)
        self.assertIs(RuleArchitecture, ArchitectureImplementation)
        self.assertIs(CandidateEvaluator, EvaluatorImplementation)
        self.assertIs(EvolutionRunner, GenerationalImplementation)
        self.assertIs(GenomeCodec, CodecImplementation)
        self.assertIs(Simulation, SimulationImplementation)


if __name__ == "__main__":
    unittest.main()
