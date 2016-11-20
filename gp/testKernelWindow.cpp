#include <QtGui>
#include "testKernelWindow.h"
#include "kernelWidget.h"
#include <iostream>
#include "RbfKernel.h"
#include "Covariance.h"
#include "linspace.h"

using namespace lfr;

namespace aphid {
namespace gpr {

TestKernelWindow::TestKernelWindow()
{
    std::cout<<" test kernel ";
    std::cout<<"\n build X";
    
    DenseMatrix<float> X(25,1);
    linspace<float>(X.column(0), -1.f, 1.f, 25);
    
    
    std::cout<<"\n build kernel";
    RbfKernel<float> rbf(0.3);
    
    Covariance<float, RbfKernel<float> > cov;
    cov.create(X, rbf);
    
    m_kernView = new KernelWidget(this);
    m_kernView->plotK(&cov.K());
    setCentralWidget(m_kernView);
    
	setWindowTitle(tr("RBF Kernel"));
	
	createActions();
    createMenus();
	
	qRegisterMetaType<QImage>("QImage");
    //connect(m_thread, SIGNAL(sendInitialDictionary(QImage)),
      //      this, SLOT(recvInitialDictionary(QImage)));
}

TestKernelWindow::~TestKernelWindow()
{
	qDebug()<<"closing main window";
}

void TestKernelWindow::createMenus()
{}

void TestKernelWindow::createActions()
{}

void TestKernelWindow::recvInitialDictionary(const QImage &image)
{}

void TestKernelWindow::keyPressEvent(QKeyEvent *e)
{
	// if (e->key() == Qt::Key_Escape)
       // close();

	QWidget::keyPressEvent(e);
}

}
}

