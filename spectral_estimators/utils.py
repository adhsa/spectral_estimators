import numpy as np

from typing import Union, List


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


def steering_matrix(thetas: Union[np.ndarray, List], n_sensors: int, d: float = 0.5) -> np.ndarray:
    """ This function returns the steering matrix of an array.

    Parameters
    ----------
    thetas : Union[np.ndarray, List]
        A list with the direction of arrivals.

    n_sensors : int
        Then number of sensors

    d : float
        The array spacing in wavelengths.

    Returns
    -------
    np.ndarray
        The steering matrix of shape (n_sensors, n_doas).

    """

    return np.array([steering_vector(theta, n_sensors, d) for theta in thetas]).T


def single_radar_target(theta: float, n_sensors: int, s: np.ndarray = None, d: float=0.5, rnd_phase: bool=True) -> np.ndarray:
    """ This function measures a single incoming radar target

    Parameters
    ----------
    s : np.ndarray
        The source signal

    theta : float
        The direction of arrival

    n_sensors : int
        The number of sensors

    d : float
        The element spacing in number of wavelengths

    rnd_phase : bool
        If true, adds a random phase shift to the signal at each snapshot

    Returns
    -------
    np.ndarray
        The measured source

    """

    if s is None:
        s = np.random.randn(32).astype(np.complex128)

    if rnd_phase:
        s *= np.exp(1j * 2 * np.pi * np.random.rand(1))

    a = steering_vector(theta, n_sensors, d=d)
    return np.outer(a, s)


def wideband_steering_vector(theta1, theta2, n_sensors, d=0.5):
    s = d * np.arange(n_sensors)

    wa = 2 * np.pi * np.sin(np.pi * theta1 / 180)
    wb = 2 * np.pi * np.sin(np.pi * theta2 / 180)

    a = (np.exp(1j * wb * s) - np.exp(-1j * wa * s)) / (1j * 2 * np.pi * s)
    return a

