#ifndef HALTON_WINDOW_H
#define HALTON_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class HaltonWidget;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    HaltonWidget * m_plot;
	
};
//! [0]

#endif
