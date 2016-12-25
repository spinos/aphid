#ifndef APH_MATH_DISTANCE_2_H
#define APH_MATH_DISTANCE_2_H
/*
 *  http://cseweb.ucsd.edu/~mdailey/netlab-help/dist2.htm
 *  C = dist2(A, B)
 *  Euclidean distances between vectors in A and B
 *  p-dimensional data points stored rowwise
 *  The i, jth entry is the squared distance from the ith row of A to the jth row of B.
 *  A m-by-p
 *  B n-by-p
 *  C m-by-n
 */
 
#include <math/linearMath.h>

namespace aphid {
   
template<typename T>
inline void dist2(DenseMatrix<T> & C,
                    const DenseMatrix<T> & A,
                    const DenseMatrix<T> & B)
{
    const int nr = A.numRows();
    const int nc = B.numRows();
    const int np = A.numColumns();
    DenseVector<T> irow(np);
    DenseVector<T> jrow(np);
    C.resize(nr, nc);
    int i, j;
    for(j=0;j<nc;++j) {
        T * cc = C.column(j);
        B.getRow(jrow, j);
        
        for(i=0;i<nr;++i) {
            A.getRow(irow, i);
            
            cc[i] = (irow - jrow).normSq();
        }
    }
}

}
#endif

