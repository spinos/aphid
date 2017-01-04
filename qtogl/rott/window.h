#ifndef ROTT_WINDOW_H
#define ROTT_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
QT_END_NAMESPACE

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

};
#endif
