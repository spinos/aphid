#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
QT_END_NAMESPACE

class GLWidget;
class PhysicsControl;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
    void createActions();
    void createMenus();
    
private slots:
    void receiveTurnOffCaching();
    
private:
    GLWidget *glWidget;
    PhysicsControl * m_physicsControl;
    QMenu * windowMenu;
    QAction * showPhysicsControlAct;
    QMenu * cachingMenu;
    QAction * enableCachingAct;
};
//! [0]

#endif
