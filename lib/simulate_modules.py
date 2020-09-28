#!-*-coding:utf-8-*-
from abc import ABCMeta , abstractmethod
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

__all__ = ["logisticRegression"]

class Simurater(object,metaclass=ABCMeta):
    @abstractmethod
    def performance( self ):
        """
        modelの性能を測るメソッド
        """
        raise NotImplementedError()


class logisticRegression:
    def __init__( self , options ):
        self.options = options

    def performance( self ):

        """
        N=10000
        X, Y = make_classification(n_samples=N, n_features=20, n_informative=2, n_redundant=2)
        fig = self.__calibration( X=X,Y=Y )

        filename = "a.png"
        if os.path.exists( os.path.dirname( filename ) ) is not True:
            message( "mkdir {}".format( os.path.dirname( filename ) ) )
            os.makedirs( os.path.dirname( filename ) )

        message( "saved calibration_curve image as {}".format( filename ) )
        plt.savefig( filename )
        """
        print(self.options)

    def __calibration( X , Y ):
        X_train , X_test , Y_train , Y_test = train_test_split( X , Y )
        N = len( Y_test )
        bins_num = int( ( ( N / N ** 0.5 ) * 0.5 ) * 2 )

        prob = model.predict_proba( X_test )[:,1]
        prob_true , prob_pred = calibration_curve( y_true=Y_test ,\
         y_prob=prob , n_bins=bins_num )

        fig = plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax1.plot( prob_pred , prob_true , marker="s" , label="calibration_curve" )
        ax1.plot( [0,1],[0,1],linestyle="--",label="ideal" )
        ax1.legend()

        ax2 = plt.subplot(2,1,2)
        ax2.hist( prob , bins=40 , histtype="step" )
        ax2.set_xlim(0,1)

        return fig
