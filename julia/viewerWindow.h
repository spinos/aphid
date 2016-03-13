#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>
#include <Parameter.h>
QT_BEGIN_NAMESPACE
class QWidget;
QT_END_NAMESPACE

namespace jul {

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(const Parameter * param);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    QWidget * m_widget;
};

}

#endif
