import numpy as np

from typing import Union


def complex_sinusoid(w: float, t: Union[np.ndarray, float]) -> np.ndarray:
    """ This function evaluates a complex sinusoid.


    Parameters
    ----------

    w : float
        The angular frequency.

    t : np.ndarray
        The instances to evaluated at.

    Returns
    -------

    np.ndarray
        A complex sinusoid.

    """

    return np.exp(1j * w * t)


def steering_vector(theta: float, n_sensors: int, d: float = 0.5) -> np.ndarray:
    """ This function returns a steering vector pointing at direction `theta`.

    Parameters
    ----------

    theta : float
        The direction of arrival in degrees

    n_sensors : int
        The number of recievers

    d : float
        Antenna spacing in number of wavelengths

    Returns
    -------

    a : np.ndarray
        The steering vector

    """

    s = d * np.arange(n_sensors)
    w = 2 * np.pi * np.sin(np.pi * theta / 180)
    a = np.exp(-1j * w * s)
    return a
