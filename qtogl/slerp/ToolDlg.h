/*
 *  ToolDlg.h
 *  slerp
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SLERP_TOOL_DLG_H
#define SLERP_TOOL_DLG_H

#include <QDialog>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace aphid {
class ContextToolGroup;
}

class ToolDlg : public QDialog
{
    Q_OBJECT

public:
    ToolDlg(QWidget *parent = 0);
	
protected:
	virtual void 	closeEvent ( QCloseEvent * e );

signals:
	void onToolDlgClose();
	void toolSelected(int x);
		
public slots:

private slots:
	void recvToolSelected(QPair<int, int> x);

private:
	aphid::ContextToolGroup* m_toolbox;
	
};

#endif