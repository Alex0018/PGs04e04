from abc import ABC, abstractmethod

class IParamsGrid(ABC):
    
    @abstractmethod
    def get_params_grid(self):
        pass