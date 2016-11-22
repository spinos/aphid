#include <QtGui>
#include "sampleKernelDialog.h"
#include "sampleKernelWidget.h"
#include "gsamp.h"

namespace aphid {
namespace gpr {

SampleKernelDialog::SampleKernelDialog(const Covariance<float, RbfKernel<float> > & covar,
										QWidget *parent)
    : QDialog(parent)
{
	    
    qDebug()<<"\n sample gaussian distribution";
    lfr::DenseMatrix<float> smps;
    lfr::SvdSolver<float> svder;
    gsamp(smps, covar.K(), 1, &svder);
	
	m_wig = new SampleKernelWidget(smps, this);
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_wig);
	layout->setStretch(2, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    setWindowTitle(tr("Sample Gaussian distribution") );
    resize(480, 480);
}

}
}
