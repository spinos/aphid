#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class GLWidget;

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
    GLWidget *glWidget;
	QMenu * fileMenu;
    QAction * importTriangleAct;
	QAction * importCurveAct;
	QAction * importPatchAct;
};
#endif
