#ifndef APH_GPR_VIEW_KERNEL_DIALOG_H
#define APH_GPR_VIEW_KERNEL_DIALOG_H
#include <QDialog>

namespace aphid {

template<typename T>
class DenseMatrix;

namespace gpr {

class KernelWidget;

class ViewKernelDialog : public QDialog
{
    Q_OBJECT

public:
    ViewKernelDialog(const DenseMatrix<float > * K,
						QWidget *parent = 0);

protected:
    
public slots:
   
private:
	
private:
	KernelWidget * m_kernView;
	
};

}
}
#endif