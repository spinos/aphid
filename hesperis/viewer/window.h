#ifndef GAR_WINDOW_H
#define GAR_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
class QStackedWidget;
class QGraphicsView;
QT_END_NAMESPACE

class GLWidget;
class ToolBox;

namespace aphid {
class HesScene;
}

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();
	~Window();
	
	void showAssets();

protected:
    void keyPressEvent(QKeyEvent *event);

private:
	void createActions();
	void createMenus();
	void changeView(int x);
	
private slots:
	void recvToolAction(int x);
	void toggleAssetDlg(bool x);
	void recvAssetDlgClose();
	void performLoad(bool x);
	void recvDspState(int x);
	
private:
    aphid::HesScene* m_scene;
	QStackedWidget * m_centerStack;
    GLWidget *glWidget;
	ToolBox * m_tools;
	QAction * m_assetAct;
	QAction * m_loadAct;
	QMenu * m_fileMenu;
    //QMenu * m_windowMenu;
    
};
#endif
