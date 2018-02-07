#ifndef LINREG_WINDOW_H
#define LINREG_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class LinregWidget;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    LinregWidget * m_plot;
	
};
//! [0]

#endif
