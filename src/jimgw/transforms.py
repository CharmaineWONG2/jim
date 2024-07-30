from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
from chex import assert_rank
from jaxtyping import Float


class Transform(ABC):
    """
    Base class for transform.
    The purpose of this class is purely for keeping name
    """

    name_mapping: tuple[list[str], list[str]]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        self.name_mapping = name_mapping

    def propagate_name(self, x: list[str]) -> list[str]:
        input_set = set(x)
        from_set = set(self.name_mapping[0])
        to_set = set(self.name_mapping[1])
        return list(input_set - from_set | to_set)

class BijectiveTransform(Transform):

    transform_func: Callable[[dict[str, Float]], dict[str, Float]]
    inverse_transform_func: Callable[[dict[str, Float]], dict[str, Float]]

    def __call__(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        return self.transform(x)

    @abstractmethod
    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """
        Transform the input x to transformed coordinate y and return the log Jacobian determinant.
        This only works if the transform is a N -> N transform.

        Parameters
        ----------
        x : dict[str, Float]
                The input dictionary.

        Returns
        -------
        y : dict[str, Float]
                The transformed dictionary.
        log_det : Float
                The log Jacobian determinant.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Push forward the input x to transformed coordinate y.

        Parameters
        ----------
        x : dict[str, Float]
                The input dictionary.

        Returns
        -------
        y : dict[str, Float]
                The transformed dictionary.
        """
        raise NotImplementedError
    
    @abstractmethod
    def inverse(self, y: dict[str, Float]) -> dict[str, Float]:
        """
        Inverse transform the input y to original coordinate x.

        Parameters
        ----------
        y : dict[str, Float]
                The transformed dictionary.

        Returns
        -------
        x : dict[str, Float]
                The original dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, y: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """
        Pull back the input y to original coordinate x and return the log Jacobian determinant.

        Parameters
        ----------
        y : dict[str, Float]
                The transformed dictionary.

        Returns
        -------
        x : dict[str, Float]
                The original dictionary.
        log_det : Float
                The log Jacobian determinant.
        """
        raise NotImplementedError    

class NonBijectiveTransform(Transform):
    
    def __call__(self, x: dict[str, Float]) -> dict[str, Float]:
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Push forward the input x to transformed coordinate y.

        Parameters
        ----------
        x : dict[str, Float]
                The input dictionary.

        Returns
        -------
        y : dict[str, Float]
                The transformed dictionary.
        """
        raise NotImplementedError

class UnivariateTransform(Transform):

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)

    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        input_params = x.pop(self.name_mapping[0][0])
        assert_rank(input_params, 0)
        output_params = self.transform_func(input_params)
        jacobian = jax.jacfwd(self.transform_func)(input_params)
        x[self.name_mapping[1][0]] = output_params
        return x, jnp.log(jacobian)

    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        input_params = x.pop(self.name_mapping[0][0])
        assert_rank(input_params, 0)
        output_params = self.transform_func(input_params)
        x[self.name_mapping[1][0]] = output_params
        return x


class ScaleTransform(BijectiveTransform):
    scale: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        scale: Float,
    ):
        super().__init__(name_mapping)
        self.scale = scale
        self.transform_func = lambda x: x * self.scale
        self.inverse_transform_func = lambda x: x / self.scale
    
    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        input_params = x.pop(self.name_mapping[0][0])
        assert_rank(input_params, 0)
        output_params = self.transform_func(input_params)
        jacobian = jnp.log(jnp.abs(self.scale))
        x[self.name_mapping[1][0]] = output_params
        return x, jacobian


class OffsetTransform(UnivariateTransform):
    offset: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        offset: Float,
    ):
        super().__init__(name_mapping)
        self.offset = offset
        self.transform_func = lambda x: x + self.offset


class LogitTransform(UnivariateTransform):
    """
    Logit transform following

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        self.transform_func = lambda x: 1 / (1 + jnp.exp(-x))


class ArcSineTransform(UnivariateTransform):
    """
    ArcSine transformation

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        self.transform_func = lambda x: jnp.arcsin(x)


class PowerLawTransform(UnivariateTransform):
    """
    PowerLaw transformation
    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.
    """

    xmin: Float
    xmax: Float
    alpha: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
        alpha: Float,
    ):
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        self.alpha = alpha
        self.transform_func = lambda x: (
            self.xmin ** (1.0 + self.alpha)
            + x * (self.xmax ** (1.0 + self.alpha) - self.xmin ** (1.0 + self.alpha))
        ) ** (1.0 / (1.0 + self.alpha))


class ParetoTransform(UnivariateTransform):
    """
    Pareto transformation: Power law when alpha = -1
    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
    ):
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        self.transform_func = lambda x: self.xmin * jnp.exp(
            x * jnp.log(self.xmax / self.xmin)
        )
