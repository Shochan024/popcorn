from abc import ABCMeta , abstractmethod

class Prediction(object,metaclass=ABCMeta):
    @abstractmethod
    def predict( self ):
        """
        推定メソッド
        --------------------------------
        """
        raise NotImplementedError()

class decision_tree(Prediction):
    def __init__( self ):
        pass

    def predict( self ):
        pass
