#ifndef DCT_H
#define DCT_H

#include <cmath>
namespace lfr {
    
template <typename T>
class Dct {
public:
    Dct();
    virtual ~Dct();
    
    static void BasisFunc(T * dst, int n, int p, int q);
    static void ApplyBasisFunc(T * dst, int n, int p, int q, T weight);
    static void EnhanceBasisFunc(T * dst, int n, int p, int q, T weight);
    static void AddBasisFunc(T * dst, int n, int p, int q, T weight);
protected:
    
private:
    static T BasisFunc(int n, int i, int j, int p, int q);
    
};

template <typename T>
Dct<T>::Dct() {}

template <typename T>
Dct<T>::~Dct() {}

/// http://cn.mathworks.com/help/images/discrete-cosine-transform.html

template <typename T>
void Dct<T>::BasisFunc(T * dst, int n, int p, int q)
{    
    int i, j;
    for(i=0;i<n;++i) {
        for(j=0;j<n;++j) {
            dst[j*n+i] = BasisFunc(n ,i,j, p, q);
        }
    }
}

template <typename T>
void Dct<T>::ApplyBasisFunc(T * dst, int n, int p, int q, T weight)
{  
    int i, j;
    for(i=0;i<n;++i) {
        for(j=0;j<n;++j) {
            dst[j*n+i] = dst[j*n+i] * (1.f - weight) + dst[j*n+i] 
                * BasisFunc(n ,i,j, p, q) * weight;
        }
    }
}

template <typename T>
void Dct<T>::AddBasisFunc(T * dst, int n, int p, int q, T weight)
{  
    int i, j;
    for(i=0;i<n;++i) {
        for(j=0;j<n;++j) {
            dst[j*n+i] += BasisFunc(n ,i,j, p, q) * weight;
        }
    }
}

template <typename T>
void Dct<T>::EnhanceBasisFunc(T * dst, int n, int p, int q, T weight)
{  
    int i, j;
    for(i=0;i<n;++i) {
        for(j=0;j<n;++j) {
            dst[j*n+i] *= 1.0 + BasisFunc(n ,i,j, p, q) * weight;
        }
    }
}

template <typename T>
T Dct<T>::BasisFunc(int n, int i, int j, int p, int q)
{
    T ap;
    if(p==0) ap = 1.0 / sqrt(n);
    else ap = sqrt(2.0/n);
    
    T aq;
    if(q==0) aq = 1.0 / sqrt(n);
    else aq = sqrt(2.0/n);
    
    return ap * aq * cos( T(2 * j + 1)*3.14159269*q/T(2*n) )
               * cos( T(2 * i + 1)*3.14159269*p/T(2*n) );
}
    
}
#endif        //  #ifndef DCT_H

