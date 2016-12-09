#include <QtGui>
#include <iostream>
#include "testKernelWindow.h"
#include "kernelWidget.h"
#include "Covariance.h"
#include "sampleKernelDialog.h"
#include "RbfKernel.h"
#include "linspace.h"

namespace aphid {
namespace gpr {

TestKernelWindow::TestKernelWindow()
{
    std::cout<<" test kernel ";
    std::cout<<"\n build X";
    
    DenseMatrix<float> X(30,1);
    linspace<float>(X.column(0), -1.f, 1.f, 30);
    
    std::cout<<"\n build kernel";
    RbfKernel<float> rbf(0.33);
    
    Covariance<float, RbfKernel<float> > cov;
    cov.create(X, rbf);
    
    m_kernView = new KernelWidget(this);
    m_kernView->plotK(&cov.K());
    setCentralWidget(m_kernView);
    
    QDateTime local(QDateTime::currentDateTime());
    qDebug() << "Local time is:" << local;
    srand (local.toTime_t() );
    
    m_smpdlg = new SampleKernelDialog(cov, 
        this);
	
	setWindowTitle(tr("RBF Kernel"));
	
	createActions();
    createMenus();
	
	qRegisterMetaType<QImage>("QImage");
    //connect(m_thread, SIGNAL(sendInitialDictionary(QImage)),
      //      this, SLOT(recvInitialDictionary(QImage)));
      
    m_smpdlg->show();
	m_smpdlg->move(0,0);
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

