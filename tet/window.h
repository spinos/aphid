#ifndef TTG_WINDOW_H
#define TTG_WINDOW_H

#include <QMainWindow>
#include "Parameter.h"
QT_BEGIN_NAMESPACE

QT_END_NAMESPACE
//! [0]
class GLWidget;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(const ttg::Parameter * param);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    GLWidget *glWidget;
};
//! [0]

#endif
