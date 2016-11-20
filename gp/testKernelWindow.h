#ifndef TEST_KERNEL_WINDOW_H
#define TEST_KERNEL_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace aphid {
namespace gpr {
    
class KernelWidget;

class TestKernelWindow : public QMainWindow
{
    Q_OBJECT

public:
    TestKernelWindow();
	virtual ~TestKernelWindow();
	
protected:
    void keyPressEvent(QKeyEvent *event);
	
private:
	void createActions();
    void createMenus();
	
private slots:
	void recvInitialDictionary(const QImage &image);

private:
	QMenu * windowMenu;
	QMenu * generateMenu;
	KernelWidget * m_kernView;
	
};

}
}
#endif
