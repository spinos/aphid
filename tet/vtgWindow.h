#ifndef TTG_WINDOW_H
#define TTG_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
QT_END_NAMESPACE

namespace ttg {

class vtgWidget;

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
	
private:
    vtgWidget *glWidget;
	
};

}
#endif
