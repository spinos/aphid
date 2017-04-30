#include <QtGui>
#include "viewKernelDialog.h"
#include "kernelWidget.h"

namespace aphid {
namespace gpr {

ViewKernelDialog::ViewKernelDialog(const DenseMatrix<float > * K,
										QWidget *parent)
    : QDialog(parent)
{
	m_kernView = new KernelWidget(this);
    m_kernView->plotK(K);

	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_kernView);
	layout->setStretch(2, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    setWindowTitle(tr("View Kernel") );
    resize(480, 480);
}

}
}
