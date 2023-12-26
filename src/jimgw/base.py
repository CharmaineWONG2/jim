from abc import ABC, abstractmethod
import equinox as eqx
from jaxtyping import Array, Float
from jimgw.jim import Jim

from jimgw.prior import Prior


class Data(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fetch(self):
        raise NotImplementedError


class Model(eqx.Module):
    params: dict

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(self, x: Array) -> float:
        raise NotImplementedError


class LikelihoodBase(ABC):
    """
    Base class for likelihoods.
    Note that this likelihood class should work
    for a some what general class of problems.
    In light of that, this class would be some what abstract,
    but the idea behind it is this handles two main components of a likelihood:
    the data and the model.
    It should be able to take the data and model and evaluate the likelihood for
    a given set of parameters.

    """

    _model: object
    _data: object

    @property
    def model(self):
        """
        The model for the likelihood.
        """
        return self._model

    @property
    def data(self):
        """
        The data for the likelihood.
        """
        return self._data

    @abstractmethod
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError


class RunManager(ABC):
    """
    Base class for run managers.

    A run manager is a class that help with book keeping for a run.
    It should be able to log metadata, summarize the run, save the run, and load the run.
    Individual use cases can extend this class to implement the actual functionality.
    This class is meant to be a template for the functionality.

    """

    likelihood: LikelihoodBase
    prior: Prior
    jim: Jim

    def __init__(self, likelihood: LikelihoodBase, prior: Prior, jim: Jim):
        """
        Initialize the run manager.

        Parameters
        ----------
        likelihood : LikelihoodBase
            The likelihood for the run.
        prior : Prior
            The prior for the run.
        jim : Jim
            The jim instance for the run.
        """
        self.likelihood = likelihood
        self.prior = prior
        self.jim = jim

    @abstractmethod
    def log_metadata(self):
        """
        Log metadata for the run.
        """
        raise NotImplementedError

    @abstractmethod
    def summarize(self):
        """
        Summarize the run.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """
        Save the run.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """
        Load the run.
        """
        raise NotImplementedError
