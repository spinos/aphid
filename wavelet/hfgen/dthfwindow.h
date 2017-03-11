/*
 *   dthfwindow.h
 *
 */
 
#ifndef DTHF_WINDOW_H
#define DTHF_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class DthfWidget;
class SynthControl;

class DthfWindow : public QMainWindow
{
    Q_OBJECT

public:
    DthfWindow(int argc, char *argv[]);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    DthfWidget * m_plot;
	SynthControl * m_control;
	
};
//! [0]

#endif
