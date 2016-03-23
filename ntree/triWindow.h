#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE
//! [0]
class TriWidget;

class TriWindow : public QMainWindow
{
    Q_OBJECT

public:
    TriWindow(const std::string & filename);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    TriWidget *glWidget;
};
//! [0]

#endif
