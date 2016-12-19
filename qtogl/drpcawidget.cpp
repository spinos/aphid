#include <QtGui>
#include <GeoDrawer.h>
#include <QtOpenGL>
#include "drpcawidget.h"
#include <gpr/PCAReduction.h>
#include <math/generate_data.h>
#include <math/transform_data.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
/// http://matlab.izmiran.ru/help/techdoc/ref/cov.html	
/// A = [-1 1 2 ; -2 3 1 ; 4 0 3]
/// C = 
///		10.3333   -4.1667    3.0000
///   -4.1667    2.3333   -1.5000
///    3.0000   -1.5000    1.0000

	DenseMatrix<float> A(3,3);
	
	A.column(0)[0] = -1;
	A.column(1)[0] = 1;
	A.column(2)[0] = 2; 
	A.column(0)[1] = -2;
	A.column(1)[1] = 3;
	A.column(2)[1] = 1; 
	A.column(0)[2] = 4; 
	A.column(1)[2] = 0;
	A.column(2)[2] = 3;
	
	std::cout<<"A"<<A;
	
	center_data(A, 1, 3.f);

	DenseMatrix<float> At = A.transposed();
	
	DenseMatrix<float> C(3,3);
	A.AtA(C);

/// normalizes by N-1 where N is the number of observations	
	C.scale(1.f/2);
	
	std::cout<<"C"<<C;
	
/// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/index.htm#_geev.htm
/// SSYEV Example Program Results 
/// Eigenvalues
/// -11.07  -6.23   0.86   8.87  16.09
/// Eigenvectors (stored columnwise)
///  -0.30  -0.61   0.40  -0.37   0.49
///  -0.51  -0.29  -0.41  -0.36  -0.61
///  -0.08  -0.38  -0.66   0.50   0.40
///   0.00  -0.45   0.46   0.62  -0.46
///  -0.80   0.45   0.17   0.31   0.16
	float a[5*5] = {
            1.96f,  0.00f,  0.00f,  0.00f,  0.00f,
           -6.49f,  3.80f,  0.00f,  0.00f,  0.00f,
           -0.47f, -6.39f,  4.17f,  0.00f,  0.00f,
           -7.20f,  1.50f, -1.51f,  5.70f,  0.00f,
           -0.65f, -6.34f,  2.67f,  1.80f, -7.10f};
	A.resize(5,5);
	A.copyData(a);
	
	EigSolver<float> eig;
	eig.computeSymmetry(A);
	std::cout<<"\n eigenvalues"<<eig.S()
			<<"\n eigenvectors"<<eig.V();
			
	m_D = 480;
	int np = m_D/3;
	m_N = 36;
	int nlg2 = 6;
	
	for(int j=0;j<nlg2;++j) {
		for(int i=0;i<nlg2;++i) {
			int k = j*nlg2+i;
			m_data[k] = new DenseMatrix<float>(3, np);
			if(k==0) {
				generate_data<float>("swiss_roll", *m_data[k], np, 0.f);
				
				Matrix33F mrot;
				mrot.rotateEuler(PI * RandomF01(),
							PI * RandomF01(),
							PI * RandomF01() );
				Vector3F vpos(5,5,10);
				
				transform_data<float>(*m_data[k], mrot, vpos, 0.1f);
				
			} else if(k==1) {
				generate_data<float>("helix", *m_data[k], np, 0.1f);
				Matrix33F mrot;
				Vector3F vpos(10,10,-5);
				
				transform_data<float>(*m_data[k], mrot, vpos, 0.1f);
				
			} else {
				m_data[k]->copy(*m_data[k&1]);
				Matrix33F mrot;
				mrot.rotateEuler(0.1 * RandomF01(),
							0.0,
							0.1 * RandomF01() );
				Vector3F vpos;
				
				transform_data<float>(*m_data[k], mrot, vpos, 0.2f);
				
			}
		}
	}
	
	DenseMatrix<float> X;
	
	PCAReduction<float> pca;
	pca.createX(m_D, m_N);
	for(int i=0;i<m_N;++i) {
		const float * src = m_data[i]->column(0);
		
		for(int j=0;j<m_D;++j) {
			pca.setXiCompj(src[j], i, j);
		}
	}
	
	DenseMatrix<float> redX;
	pca.compute(redX);
	std::cout<<" reduced x "<<redX;
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
    getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(0.f, .85f, .55f);

	for(int i=0;i<m_N;++i) {
	    
	    glEnableClientState(GL_VERTEX_ARRAY);
        
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_data[i]->column(0) );
        glDrawArrays(GL_POINTS, 0, m_data[i]->numCols() );
        
        glDisableClientState(GL_VERTEX_ARRAY);
	}
}
