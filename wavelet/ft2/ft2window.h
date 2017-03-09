#ifndef FT_2D_WINDOW_H
#define FT_2D_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class Ft2Widget;

class Ft2Window : public QMainWindow
{
    Q_OBJECT

public:
    Ft2Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    Ft2Widget * m_plot;
	
};
//! [0]

#endif
