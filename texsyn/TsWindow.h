#ifndef TS_Window_H
#define TS_Window_H

#include <QMainWindow>
#include <LfMachine.h>
#include <LfThread.h>
#include <StatisticDialog.h>
#include <DictionaryDialog.h>
#include <ImageDialog.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

using namespace lfr;

namespace tss {

class TsWindow : public QMainWindow
{
    Q_OBJECT

public:
    TsWindow(LfMachine * world);
	virtual ~TsWindow();
	
protected:
    void keyPressEvent(QKeyEvent *event);
	
private:
	void createActions();
    void createMenus();
	
private slots:
	void recvInitialDictionary(const QImage &image);

private:
	LfThread * m_thread;
    StatisticDialog * m_statistics;
	DictionaryDialog * m_dictionary;
	ImageDialog * m_image;
	QMenu * windowMenu;
	QMenu * generateMenu;
	QAction * generateAct;
	QAction * shoImageAct;
	QAction * shoDictionaryAct;
	QAction * statisticAct;
	
};

}
#endif
