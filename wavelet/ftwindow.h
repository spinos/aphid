#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class Ft1dWidget;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    Ft1dWidget * m_plot;
	
};
//! [0]

#endif
