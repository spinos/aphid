#include <QtGui>
#include "sampleKernelDialog.h"
#include "sampleKernelWidget.h"
#include <math/gsamp.h>

namespace aphid {
namespace gpr {

SampleKernelDialog::SampleKernelDialog(const Covariance<float, RbfKernel<float> > & covar,
										QWidget *parent)
    : QDialog(parent)
{
#if 0
	DenseMatrix<float> A(covar.K().numRows(), covar.K().numCols() );
	A.copy(covar.K() );
/// K * K^-1 is I if add diag
/// otherwise is symmetric
	// A.addDiagonal(0.1f);
	DenseMatrix<float> KKi(A.numRows(), A.numColumns() );
	A.mult(KKi, covar.Kinv() );
	 
	std::cout<<"\n K*K^-1"<<KKi;
#endif
	 
    DenseMatrix<float> smps;
    SvdSolver<float> svder;
    gsamp(smps, covar.K(), 1, &svder);
	
#if 0
	std::cout<<"\n K"<<covar.K();
#endif
	
	m_wig = new SampleKernelWidget(smps, this);
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_wig);
	layout->setStretch(2, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    setWindowTitle(tr("Simple Interpolation") );
    resize(480, 480);
}

}
}
