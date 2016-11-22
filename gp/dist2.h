#ifndef APH_GPR_DISTANCE_2_H
#define APH_GPR_DISTANCE_2_H
/*
 *  http://cseweb.ucsd.edu/~mdailey/netlab-help/dist2.htm
 *  C = dist2(A, B)
 *  Euclidean distances between vectors in A and B
 *  The i, jth entry is the squared distance from the ith row of A to the jth row of B.
 *  A m-by-p
 *  B n-by-p
 *  C m-by-n
 */
 
#include <linearMath.h>

namespace aphid {
namespace gpr {
    
template<typename T>
inline void dist2(lfr::DenseMatrix<T> & C,
                    const lfr::DenseMatrix<T> & A,
                    const lfr::DenseMatrix<T> & B)
{
    const int nr = A.numRows();
    const int nc = B.numRows();
    const int np = A.numColumns();
    lfr::DenseVector<T> irow(np);
    lfr::DenseVector<T> jrow(np);
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
}
#endif

