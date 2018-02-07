#ifndef SDFT_WINDOW_H
#define SDFT_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
QT_END_NAMESPACE

namespace aphid {
class SuperformulaControl;
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
	aphid::SuperformulaControl* m_superformulaControl;
	
};
#endif
