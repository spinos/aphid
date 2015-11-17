#include <iostream>
#include "clapackTempl.h"

int main ( )
{ 
    int m = 5, n = 4;
    double * A = new double[m*n];
    double * X = new double[m*n];
    int i, j;
    for(i=0; i< m; i++) {
        for(j=0; j< n; j++) {
            if(i==j) A[j*m + i] = 1.0;
            else A[j*m + i] = 0.0;
            X[j*m + i] = j*m + i;
        }
        
    }
    
    std::cout<<"\n A";
    for(i=0; i< m; i++) {
        std::cout<<"\n| ";
        for(j=0; j< n; j++) {
            std::cout<<" "<<A[j*m + i];
        }
        std::cout<<" |";
    }
    
    std::cout<<"\n X";
    for(i=0; i< m; i++) {
        std::cout<<"\n| ";
        for(j=0; j< n; j++) {
            std::cout<<" "<<X[j*m + i];
        }
        std::cout<<" |";
    }
    
    std::cout<<"\n A <- A + 0.5 X";
    clapack_axpy<double>(m*n, .5, X, 1, A, 1);
    
    for(i=0; i< m; i++) {
        std::cout<<"\n| ";
        for(j=0; j< n; j++) {
            std::cout<<" "<<A[j*m + i];
        }
        std::cout<<" |";
    }
    
    return 1;
}
