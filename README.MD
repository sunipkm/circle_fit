# Circle Fitting
Fits supplied data to a circle.

### Example
Use from other scripts like:

```py
    from circle_fit import CircleFit
    import numpy as np

    x = np.r_[  9, 35, -13,  10,  23,   0]
    y = np.r_[ 34, 10,   6, -14,  27, -10]

    cfit = CircleFit(x, y, 'leastsq')
    print(cfit)
    cfit.plot()
```