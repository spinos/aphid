/*
 *  ToolBox.h
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef TOOLBOX_H
#define TOOLBOX_H

#include <QToolBar>
#include <ToolContext.h>

class ContextIconFrame;
class ToolBox : public QToolBar, public ToolContext {
Q_OBJECT

public:	
	ToolBox(QWidget *parent = 0);
    ~ToolBox();
private:
	std::vector<ContextIconFrame *> m_contextFrames;
	
signals:
	void contextChanged(int c);
	
public slots:
	void contextFrameChanged(int c);
};

#endif