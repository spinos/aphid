/*
 *  sparse linear math test
 */
 
#include <iostream>
#include <math/sparseLinearMath.h>
using namespace aphid;

void testSpVec()
{
	std::cout<<"\n test sparse vec";
	
	static const int elmInd[] = {40, 11, 19, 43, 3, 18, 199, 39, 300, 31, 42, 97};
	static const int elmV[] = {0,1,2,3,4,5,6,7,8,9, 10, 11};
	static const int nElm = 12;
	
	SparseVector<int> vecx;
	
	std::cout<<"\n add elms\n";
	
	for(int i=0;i<nElm;++i) {
		std::cout<<" "<<elmInd[i]<<":"<<elmV[i];
		std::cout.flush();
		vecx.set(elmInd[i], elmV[i]);
	}
	
	std::cout<<"\n index:value\n";
	
	SparseIterator<int> iter = vecx.begin();
	SparseIterator<int> itEnd = vecx.end();
	for(;iter != itEnd;++iter) {
		std::cout<<" "<<iter.index()<<":"<<iter.value();
	}
	
	std::cout<<"\n remove 43\n";
	vecx.remove(43);
	
	iter = vecx.begin();
	itEnd = vecx.end();
	for(;iter != itEnd;++iter) {
		std::cout<<" "<<iter.index()<<":"<<iter.value();
	}

}

void printSpMat(const SparseMatrix<int>& matx)
{
	const int& nrow = matx.numRows();
	const int& ncol = matx.numCols();
	std::cout<<"\n "<<nrow<<"-by-"<<ncol<<" mat";
	if(matx.isColumnMajor() ) {
		std::cout<<" column-major\n";
	} else {
		std::cout<<" row-major\n";
	}
	for(int i=0;i<nrow;++i) {
		for(int j=0;j<ncol;++j) {
			int e = matx.get(i,j);
			std::cout<<" "<<e;
		}
		std::cout<<"\n";
	}
	
}

void testSpMat()
{
	std::cout<<"\n test sparse mat";
	
	SparseMatrix<int> matx;
	static const int nrow = 30;
	static const int ncol = 40;
	matx.create(nrow, ncol);
	
	std::cout<<"\n add elms\n";
	
	for(int i=0;i<nrow;++i) {
		matx.set(i,i,1);
		for(int j=0; j< 3;++j) {
			matx.set(i,rand() % ncol ,1);
		}
	}
	
	printSpMat(matx);
	
	SparseMatrix<int> matt = matx.transposed();
	
	printSpMat(matt);
	
	std::cout<<"\n c = ab";
	SparseMatrix<int> matc = matx * matt;
	
	printSpMat(matc);
}

int main(int argc, char **argv)
{        
    std::cout<<"\n test sparse linear math";
	testSpVec();
	testSpMat();
	std::cout<<"\ndone.\n";
    exit(0);
}
