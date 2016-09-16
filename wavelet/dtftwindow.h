#ifndef DT_FT_WINDOW_H
#define DT_FT_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class DtFtWidget;

class dtftWindow : public QMainWindow
{
    Q_OBJECT

public:
    dtftWindow();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    DtFtWidget * m_plot;
	
};
//! [0]

#endif
