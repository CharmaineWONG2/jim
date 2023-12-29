from jimgw.base import RunManager
from dataclasses import dataclass


@dataclass
class SingleEventRun:
    seed: int
    path: str

    detectors: list[str]
    data: list[str]
    psds: list[str]
    priors: list[str]
    waveform: str
    waveform_parameters: dict[str, str | float | int | bool]
    jim_parameters: dict[str, str | float | int | bool]
    likelihood_parameters: dict[str, str | float | int | bool]
    trigger_time: int
    duration: int
    post_trigger_duration: int
    fmin: float
    fmax: float
    injection_parameters: dict[str, float]


class SingleEventPERunManager(RunManager):
    run: SingleEventRun

    @property
    def waveform(self):
        return self.likelihood.waveform

    @property
    def detectors(self):
        return self.run.detectors

    @property
    def data(self):
        return [detector.data for detector in self.likelihood.detectors]

    @property
    def psds(self):
        return self.run.detectors

    def __init__(self, path: str, **kwargs):
        if "run" in kwargs:
            print("Run instance provided. Loading from instance.")
            self.run = kwargs["run"]
        else:
            try:
                print("Run instance not provided. Loading from path.")
                self.run = self.load(path)
            except Exception as e:
                print("Fail to load from path. Please check the path.")
                raise e

    def log_metadata(self):
        pass

    def summarize(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str) -> SingleEventRun:
        raise NotImplementedError

    def fetch_data(self):
        """
        Given a run config that specify using real data, fetch the data from the server.


        """
        try:
            pass
        except Exception as e:
            raise e

    def generate_data(self):
        """
        Given a run config that specify using simulated data, generate the data.
        """
        try:
            pass
        except Exception as e:
            raise e

    def initialize_detector(self):
        """
        Initialize the detectors.
        """
        try:
            pass
        except Exception as e:
            raise e

    def initialize_waveform(self):
        """
        Initialize the waveform.
        """
        try:
            pass
        except Exception as e:
            raise e
