#include <iostream>
#include <gpr/PCAReduction.h>
#include <math/generate_data.h>
#include <time.h> 

using namespace aphid;

int main(int argc, char *argv[])
{
    std::cout<<"\n test dimensionality reduction PCA ";
	srand(std::time(NULL) );
	DenseMatrix<float> X;
	const int n = 10;
	const int d = 45;
	const int np = d/3;
	
	PCAReduction<float> pca;
	pca.createX(d, n);
	for(int i=0;i<n;++i) {
		if(i<2) generate_data<float>("swiss_roll", X, np, 0.01);
		for(int j=0;j<np;++j) {
			pca.setXiCompj(X.column(j)[0] + 0.01 * RandomFn11(),i, j*3);
			pca.setXiCompj(X.column(j)[1] + 0.01 * RandomFn11(),i, j*3+1);
			pca.setXiCompj(X.column(j)[2] + 0.01 * RandomFn11(),i, j*3+2);
		}
	}
	
	DenseMatrix<float> redX;
	pca.compute(redX);
	
	std::cout<<" redx "<<redX;
	
    return 1;
}
