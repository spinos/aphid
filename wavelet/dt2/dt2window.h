#ifndef DT_FT_2D_WINDOW_H
#define DT_FT_2D_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class Dt2Widget;

class dt2Window : public QMainWindow
{
    Q_OBJECT

public:
    dt2Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    Dt2Widget * m_plot;
	
};
//! [0]

#endif
