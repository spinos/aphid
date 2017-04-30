#ifndef WBG_WINDOW_H
#define WBG_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
QT_END_NAMESPACE

class GLWidget;
class ToolBox;
class AssetDlg;

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
	void toggleAssetDlg(bool x);
	void recvAssetDlgClose();
	
private:
    GLWidget *glWidget;
	ToolBox * m_tools;
	AssetDlg * m_assets;
	QAction * m_assetAct;
    QMenu * m_windowMenu;
    
};
#endif
