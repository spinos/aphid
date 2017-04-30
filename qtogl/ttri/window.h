#ifndef TTG_WINDOW_H
#define TTG_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
QT_END_NAMESPACE

namespace tti {
class Parameter;
}

class GLWidget;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(const tti::Parameter * param);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
	void createActions();
	void createMenus();
	
private:
    GLWidget *glWidget;
	
};
#endif
