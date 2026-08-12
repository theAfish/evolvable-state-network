"""Task adapters: the only layer allowed to depend on both evolution and environments."""

from .food_web import FoodWebEvaluation, FoodWebTaskConfig, FoodWebTaskEvaluator, MLPFoodWebControllerBlueprint
from .embodied_food_web import (
    BatchFoodWebCoevolutionRunner,
    BatchFoodWebConfig,
    EmbodiedFoodWebEvaluation,
    EmbodiedFoodWebEvaluator,
    EmbodiedFoodWebTaskConfig,
    EmbodiedRuleEvolutionConfig,
    EmbodiedRuleEvolutionRunner,
    EvolutionTerminated,
    FoodWebCoevolutionEvaluator,
    FoodWebCoevolutionEvaluation,
    FoodWebCoevolutionRunner,
    ContinuousFoodWebConfig,
    ContinuousFoodWebCoevolutionRunner,
    FoodWebDemonstration,
)

__all__ = ["BatchFoodWebCoevolutionRunner", "BatchFoodWebConfig", "ContinuousFoodWebConfig", "ContinuousFoodWebCoevolutionRunner", "EmbodiedFoodWebEvaluation", "EmbodiedFoodWebEvaluator", "EmbodiedFoodWebTaskConfig", "EmbodiedRuleEvolutionConfig", "EmbodiedRuleEvolutionRunner", "EvolutionTerminated", "FoodWebCoevolutionEvaluation", "FoodWebCoevolutionEvaluator", "FoodWebCoevolutionRunner", "FoodWebDemonstration", "FoodWebEvaluation", "FoodWebTaskConfig", "FoodWebTaskEvaluator", "MLPFoodWebControllerBlueprint"]
