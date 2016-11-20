#ifndef APH_GPR_LINE_SPACE_H
#define APH_GPR_LINE_SPACE_H
/*
 * linspace(x1,x2,n) generate n points spacing (x2-x1)/(n-1)
 */
namespace aphid {
namespace gpr {
    
template<typename T>
inline void linspace(T * y, T x1, T x2, int n)
{
    T d = (x2 - x1)/(n-1);
    for(int i=0;i<n;++i) {
        y[i] = x1 + d * i;
    }
}

}
}
#endif
