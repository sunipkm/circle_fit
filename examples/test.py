from circle_fit import CircleFit
import numpy as np

def cli():
    """
    Simple demo of fitting a circle
    """
    x = np.r_[  9, 35, -13,  10,  23,   0]
    y = np.r_[ 34, 10,   6, -14,  27, -10]

    cfit = CircleFit(x, y, 'odr+jacobian')
    print(cfit)
    cfit.plot()

if __name__ == '__main__':
    cli()