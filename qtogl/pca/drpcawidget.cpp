#include <QtGui>
#include <GeoDrawer.h>
#include <QtOpenGL>
#include "drpcawidget.h"
#include <ogl/DrawDop.h>
#include <math/generate_data.h>
#include <math/transform_data.h>
#include <gpr/PCASimilarity.h>
#include <gpr/PCAFeature.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
#if 0
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
	
	DenseVector vm;
	center_data(A, 1, 3.f, vm);

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
#endif

	m_features = new PCASimilarity<float, PCAFeature<float, 3> >;
			
	m_D = 1200;
	int np = m_D/3;
	m_N = 64;
	int nlg2 = 8;
	int ndv2 = 32;
	
	for(int j=0;j<nlg2;++j) {
		for(int i=0;i<nlg2;++i) {
			int k = j*nlg2+i;
			m_data[k] = new DenseMatrix<float>(3, np);
			if(k==0) {
				generate_data<float>("swiss_roll", *m_data[k], np, 0.f);
				
				Matrix33F mrot;
				Vector3F vpos;
				Vector3F vsca(1.f, 1.f, 1.f);
				
				transform_data<float>(*m_data[k], vsca, mrot, vpos, 0.0f);
				
				m_features->begin(*m_data[k], 1);
				
			} else if(k==1) {
				generate_data<float>("helix", *m_data[k], np, 0.f);
				Matrix33F mrot;
				Vector3F vpos;
				Vector3F vsca(1.f, 1.f, 1.f);
				
				transform_data<float>(*m_data[k], vsca, mrot, vpos, 0.0f);
				
				m_features->select(*m_data[k], 1);
				
			} else {
#if 1
				m_data[k]->copy(*m_data[k>ndv2]);
#else
				m_data[k]->copy(*m_data[rand() & 1]);
#endif
				Matrix33F mrot;
						
				mrot.rotateEuler(0.5f * PI * RandomFn11(),
							0.99f * PI * RandomFn11(),
							0.5f * PI * RandomF01() );
				
				Vector3F vpos(9.f * i + 0.f * RandomFn11() + 30 * (k>ndv2),
							9.f * j + 0.f * RandomFn11(),
							0.f);
/// same scale all axises 
				float nsca = RandomFlh(0.5, 1.5);
				Vector3F vsca(nsca, nsca, nsca);
				
				transform_data<float>(*m_data[k], vsca, mrot, vpos, 0.03f);
				
				m_features->select(*m_data[k], 1);
			}
		}
	}
	
	std::cout<<"\n n feature "<<m_features->numFeatures();
	m_features->separateFeatures();
	std::cout.flush();
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
#if 0
	getDrawer()->setColor(0.f, .0f, .55f);

	for(int i=0;i<m_N;++i) {
	    
	    glEnableClientState(GL_VERTEX_ARRAY);
        
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_data[i]->column(0) );
        glDrawArrays(GL_POINTS, 0, m_data[i]->numCols() );
        
        glDisableClientState(GL_VERTEX_ARRAY);
	}
#endif	
	drawFeatures();
}

void GLWidget::drawFeatures()
{
	getDrawer()->m_wireProfile.apply();
	
	const float groupCol[2][3] = {
		{.85f, .45f, 0.f},
		{0.f, .85f, .45f}
	};
	
	float mm[16] = 
	{1, 0, 0, 0, 
	0, 1, 0, 0, 
	0, 0, 1, 0, 
	0, 0, 0, 1};
	
/// stored columnwise
	DenseMatrix<float> pdr(3, m_features->featureDim() );
	BoundingBox box;
/// center at zero and no rotation
	AOrientedBox ob;
	DrawDop dd;
	
	const int n = m_features->numFeatures();
	for(int i=0;i<n;++i) {
		m_features->getFeatureSpace(mm, i);
		
		glPushMatrix();
		glMultMatrixf((const GLfloat*)mm);
		
		getDrawer()->coordsys();
#if 1
		m_features->getFeaturePoints(pdr, i, 1);
		
		getDrawer()->setColor(.95f, .85f, .85f);
		
		glEnableClientState(GL_VERTEX_ARRAY);
        
		glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)pdr.column(0) );
        glDrawArrays(GL_POINTS, 0, pdr.numCols() );
        
        glDisableClientState(GL_VERTEX_ARRAY);
#endif
/// bounding box is columnwise
		m_features->getFeatureBound((float *)&box, i, 1);
		
		const float * gcol = groupCol[m_features->groupIndices()[i]];
		getDrawer()->setColor(gcol[0], gcol[1], gcol[2]);
		getDrawer()->boundingBox(box);
		
		ob.calculateCenterExtents(pdr.column(0), pdr.numCols() );
		dd.update8DopPoints(ob);

		glEnableClientState(GL_VERTEX_ARRAY);
        dd.drawAWireDop();
		glDisableClientState(GL_VERTEX_ARRAY);
		
		glPopMatrix();
	}
}
