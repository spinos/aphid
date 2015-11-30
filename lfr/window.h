#ifndef WINDOW_H
#define WINDOW_H

#include <QMainWindow>
#include "LfWorld.h"
#include "LfThread.h"
#include "LfWidget.h"
#include "StatisticDialog.h"
QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

using namespace lfr;

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window(LfWorld * world);
	
protected:
    void keyPressEvent(QKeyEvent *event);
	
private:
	void createActions();
    void createMenus();
	
private slots:
	void recvInitialDictionary(const QImage &image);

private:
	LfThread * m_thread;
    LfWidget * m_mainWidget;
	StatisticDialog * m_statistics;
	QMenu * windowMenu;
	QAction * statisticAct;
};
//! [0]

#endif
