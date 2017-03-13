/*
 *   gauwindow.h
 *
 */
 
#ifndef GAU_WINDOW_H
#define GAU_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class GauWidget;

class GauWindow : public QMainWindow
{
    Q_OBJECT

public:
    GauWindow(int argc, char *argv[]);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    GauWidget * m_plot;
	
};
//! [0]

#endif
