#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>
#include <Base3DView.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(int argc, char *argv[]);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    aphid::Base3DView *glWidget;
};
//! [0]

#endif
