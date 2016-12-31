#ifndef NACA_4_WINDOW_H
#define NACA_4_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
QT_END_NAMESPACE

class GLWidget;
class ParamDialog;

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
    ParamDialog * m_xDlg;
	
};
#endif
