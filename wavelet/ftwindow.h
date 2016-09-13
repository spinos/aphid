#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

#include <Plot2DWidget.h>

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    aphid::Plot2DWidget * m_plot;
	
};
//! [0]

#endif
