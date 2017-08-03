#ifndef GAR_WINDOW_H
#define GAR_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
QT_END_NAMESPACE

class GLWidget;
class ToolBox;
class AssetDlg;
class AttribDlg;
class ShrubScene;
class ChartDlg;
class ShrubChartView;
class Vegetation;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window();
	~Window();
	
/// initial states
	void showDlgs();
	
protected:
    void keyPressEvent(QKeyEvent *event);

private:
	void createActions();
	void createMenus();
	void singleSynth();
	void multiSynth();
	
private slots:
	void recvToolAction(int x);
	void toggleAssetDlg(bool x);
	void recvAssetDlgClose();
	void performExport(bool x);
	void recvDspState(int x);
	void recvChartDlgClose();
	void recvAttribDlgClose();
	void toggleChartDlg(bool x);
	void toggleAttribDlg(bool x);
	
private:
	GLWidget *glWidget;
	ToolBox * m_tools;
	AssetDlg * m_assets;
	ChartDlg* m_chart;
	AttribDlg* m_attrib;
	QAction * m_assetAct;
	QAction * m_graphAct;
	QAction * m_attribAct;
	QAction * m_exportAct;
	QMenu * m_fileMenu;
    QMenu * m_windowMenu;
    ShrubScene * m_scene;
	ShrubChartView * m_chartView;
	Vegetation * m_vege;

};
#endif
