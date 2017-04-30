#ifndef TTG_WINDOW_H
#define TTG_WINDOW_H

#include <QMainWindow>
#include "vdfParameter.h"
QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
QT_END_NAMESPACE

namespace ttg {

class vdfWidget;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(const ttg::Parameter * param);

protected:
    void keyPressEvent(QKeyEvent *event);

private:
	void createActions(Parameter::Operation opt);
	void createMenus(Parameter::Operation opt);
	
private:
    vdfWidget *glWidget;
	
};

}
#endif
