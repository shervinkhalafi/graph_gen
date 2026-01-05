"""Custom exceptions for experiment utilities.

This module defines a hierarchy of exceptions for the experiment_utils package,
allowing callers to catch specific error types rather than generic ValueError
or RuntimeError exceptions.
"""


class ExperimentUtilsError(Exception):
    """Base exception for experiment_utils module.

    All custom exceptions in experiment_utils inherit from this class,
    allowing callers to catch any experiment_utils error with a single
    except clause if desired.
    """

    pass


class ConfigurationError(ExperimentUtilsError):
    """Raised when configuration is invalid.

    Examples: unknown loss_type, unknown optimizer_type, invalid scheduler config.
    """

    pass


class CheckpointMismatchError(ExperimentUtilsError):
    """Raised when checkpoint doesn't match expected model signature.

    This occurs when loading a checkpoint that contains hyperparameters
    not accepted by the model's __init__ method.
    """

    pass


class DataModuleStateError(ExperimentUtilsError):
    """Raised when data module operations are called in wrong order.

    For example, calling setup() before prepare_data() when the data module
    requires a specific initialization sequence.
    """

    pass
