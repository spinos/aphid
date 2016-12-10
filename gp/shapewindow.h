#ifndef INST_WINDOW_H
#define INST_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace aphid {
namespace gpr {
class ViewKernelDialog;
}
}

class GLWidget;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();
    ~Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:

    GLWidget *glWidget;
	aphid::gpr::ViewKernelDialog * m_kernDlg;
    
};
#endif
