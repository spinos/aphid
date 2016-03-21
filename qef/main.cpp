#include <iostream>
#include <cmath>
#include <MersenneTwister.h>
#include <Vector3F.h>
#include <QuadraticErrorFunction.h>

/// reference https://inst.eecs.berkeley.edu/~ee127a/book/login/def_pseudo_inv.html

void printMatrix( char* desc, int m, int n, double* a ) 
{
	int i, j;
	std::cout<<"\n "<<desc;
	for(i=0; i< m; i++) {
        std::cout<<"\n| ";
        for(j=0; j< n; j++) {
            std::cout<<" "<<a[j*m + i];
        }
        std::cout<<" |";
    }
	std::cout<<"\n";
}

void testSVD3()
{
	double a[3*8]= {
            -0.008275, 0.01513, 0.0228, -0.02298, 0.007532, -0.04152, -0.01905, -0.01169,
0.064, 0.06322, 0.06318, 0.0007675, -4.113e-05, -0.06099, -0.06174, -0.06178,
0.1215, 0.002071, -0.1193, 0.1173, -0.1192, 0.1121, -0.002552, -0.1191
		};
		
	lfr::DenseMatrix<double> P(8, 3);
	int i, j;
	
	for(j=0;j<3;++j) {
		for(i=0;i<8;++i) {
			P.column(j)[i] = a[j*8+i];
		}
	}
	
/// P is changed after svd, keep a copy
	lfr::DenseMatrix<double> A(8, 3);
	A.copy(P);
	std::cout<<"\n A  "<<A;
	
	lfr::SvdSolver<double> slv;
	slv.compute(P);
	
	std::cout<<" s"<<slv.S();
	std::cout<<" u"<<slv.U();
	std::cout<<" v"<<slv.Vt();
	
	std::cout<<"\n proof A = U S V^T";
	lfr::DenseMatrix<double> Sd(8, 3);
	Sd.setZero();
	for(i=0;i<3;++i)
		Sd.column(i)[i] = slv.S()[i];
		
	lfr::DenseMatrix<double> USd(8, 3); 
	slv.U().mult(USd, Sd);
	
	lfr::DenseMatrix<double> Atau(8, 3); 
	USd.mult(Atau, slv.Vt() );
	
	std::cout<<"\n A' "<< Atau
	<<"\n A  "<<A;

/// reference https://inst.eecs.berkeley.edu/~ee127a/book/login/exa_pinv_4by5.html	
	std::cout<<"\n proof left inverse of A \n A* = V S^-1 U^T";
	lfr::DenseMatrix<double> S1(8, 3);
	S1.setZero();
	for(i=0;i<3;++i)
		S1.column(i)[i] = 1.0 / slv.S()[i];
		
	lfr::DenseMatrix<double> VS1(3, 3);
	slv.Vt().transMult(VS1, S1);

	lfr::DenseMatrix<double> Astar(3, 8);
	VS1.multTrans(Astar, slv.U() );
	std::cout<<"\n A* "<< Astar;
	
	lfr::DenseMatrix<double> I(3, 3);
	Astar.mult(I, A);
	std::cout<<"\n A* A = I n "<< I;
}

void testQE()
{
	std::cout<<"\n begin test qe";
	aphid::QuadraticErrorFunction<float, 3> qf;
	qf.create(5);
	
	aphid::Vector3F Ns[5];
	Ns[0].set(-.454f, 0.001f, .454f);
	Ns[1].set(-.454f, 0.001f, .454f);
	Ns[2].set( .454f, -0.015f, .454f);
	Ns[3].set( .454f, -0.0f, .454f);
	Ns[4].set( .0f, 0.0f, .54f);
	
	Ns[0].normalize();
	Ns[1].normalize();
	Ns[2].normalize();
	Ns[3].normalize();
	Ns[4].normalize();
	
	aphid::Vector3F Ps[4];
	Ps[0].set(0.f, 0.f, 0.f);
	Ps[1].set(0.f, 1.f, 0.f);
	Ps[2].set(1.f, 0.f, 0.f);
	Ps[3].set(1.f, 1.f, 0.f);
	Ps[4].set(.5f, 0.f, .5f);
	
	float ndp;
	int i=0;
	for(;i<5;++i) {
		qf.copyARow(i, (const float *)&Ns[i]);
		ndp = Ns[i].dot(Ps[i] ); 
		std::cout<<"\n b["<<i<<"] "<<ndp;
		qf.copyB(i, &ndp );
	}
	
	qf.compute();
	
	std::cout<<"\n x "<< qf.x();
	
	std::cout<<"\n end test qe";
}

int main()
{ 
    testSVD3();
	testQE();
	return 1;
}
