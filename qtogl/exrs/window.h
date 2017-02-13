#ifndef EXRS_WINDOW_H
#define EXRS_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
QT_END_NAMESPACE

class GLWidget;

namespace exrs {
class Parameter;
}

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(const exrs::Parameter * param);
    ~Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:

    GLWidget *glWidget;

};
#endif
