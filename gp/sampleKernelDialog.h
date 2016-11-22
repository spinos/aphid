#ifndef APH_GPR_SAMPLE_KERNEL_DIALOG_H
#define APH_GPR_SAMPLE_KERNEL_DIALOG_H
#include <QDialog>
#include "RbfKernel.h"
#include "Covariance.h"

namespace aphid {
namespace gpr {

class SampleKernelWidget;

class SampleKernelDialog : public QDialog
{
    Q_OBJECT

public:
    SampleKernelDialog(const Covariance<float, RbfKernel<float> > & covar,
						QWidget *parent = 0);

protected:
    
public slots:
   
private:
	
private:
	SampleKernelWidget * m_wig;
	
};

}
}
#endif