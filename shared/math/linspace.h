#ifndef APH_MATH_LIN_SPACE_H
#define APH_MATH_LIN_SPACE_H
/*
 * linspace(x1,x2,n) generate n points spacing (x2-x1)/(n-1)
 */
namespace aphid {
    
template<typename T>
inline void linspace(T * y, T x1, T x2, int n)
{
    T d = (x2 - x1)/(T)(n-1);
    for(int i=0;i<n;++i) {
        y[i] = x1 + d * i;
    }
}

template<typename T>
inline void linspace_center(T * y, T x1, T x2, int n)
{
    T d = (x2 - x1)/(T)n;
    for(int i=0;i<n;++i) {
        y[i] = x1 + d * (i + 0.5);
    }
}

template<typename T>
inline void linspace_center_reverse(T * y, T x1, T x2, int n)
{
    T d = (x2 - x1)/(T)n;
    for(int i=0;i<n;++i) {
        y[i] = x2 - d * (i + 0.5);
    }
}

}
#endif
