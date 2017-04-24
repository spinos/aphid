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
class AssetDlg;
class ShrubScene;
class ShrubChartView;
class VegetationPatch;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();
	~Window();

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
	
private:
	QStackedWidget * m_centerStack;
    GLWidget *glWidget;
	ToolBox * m_tools;
	AssetDlg * m_assets;
	QAction * m_assetAct;
    QMenu * m_windowMenu;
    ShrubScene * m_scene;
	ShrubChartView * m_chartView;
	VegetationPatch * m_vege;

};
#endif
