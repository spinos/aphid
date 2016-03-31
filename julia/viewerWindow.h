#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QWidget;
QT_END_NAMESPACE

namespace jul {

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(int argc, char *argv[]);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    QWidget * m_widget;
};

}

#endif
