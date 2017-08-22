#ifndef GAR_WINDOW_H
#define GAR_WINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QMenu;
class QAction;
QT_END_NAMESPACE

class GLWidget;
class ToolBox;
class AttribDlg;
class ShrubScene;
class ChartDlg;
class TexcoordDlg;
class ShrubChartView;
class AboutGardenDlg;
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
	void performExport(bool x);
	void recvDspState(int x);
	void recvChartDlgClose();
	void recvAttribDlgClose();
	void recvTexcoordDlgClose();
	void toggleChartDlg(bool x);
	void toggleAttribDlg(bool x);
	void toggleTexcoordDlg(bool x);
	void shoAboutWin();
	
private:
	GLWidget *glWidget;
	ToolBox * m_tools;
	ChartDlg* m_chart;
	AttribDlg* m_attrib;
	TexcoordDlg* m_texcoord;
	QAction * m_graphAct;
	QAction * m_attribAct;
	QAction * m_exportAct;
	QAction * m_texcoordAct;
	QAction * m_aboutAct;
	QMenu * m_fileMenu;
    QMenu * m_windowMenu;
    QMenu * m_helpMenu;
    ShrubScene * m_scene;
	ShrubChartView * m_chartView;
	AboutGardenDlg* m_aboutDlg;
	Vegetation * m_vege;
	
};
#endif
