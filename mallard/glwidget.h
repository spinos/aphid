/*
 *  glWidget.h
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <ScenePort.h>

class BezierDrawer;
class MlDrawer;
class MlEngine;
//! [0]
class GLWidget : public ScenePort
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
	virtual void doClear();
	virtual void doClose();
	
	void rebuildFeather();
	void clearFeather();
	void bakeFrames();
	void testRender();
	
signals:
	void sendFeatherEditBackground(QString name);
	void renderResChanged(QSize s);
	void renderEngineChanged(QString name);
	void renderStarted(QString name);
	
public slots:
	void updateOnFrame(int x);
	void exportBake();
	void receiveCancelRender();
	
protected:
	virtual void clientDraw();
	virtual void clientSelect(QMouseEvent *event);
	virtual void clientMouseInput(QMouseEvent *event);
	virtual void importBody(const std::string & fileName);
	virtual void afterOpen();
private:
	virtual char selectFeather();
	virtual char floodFeather();
	void beginBaking();
	void endBaking();
	bool isBaking() const;
	QString renderName() const;
	void drawFeather();

private:
	BezierDrawer * m_bezierDrawer;
	MlDrawer * m_featherDrawer;
	MlEngine * m_engine;
};
//! [3]

#endif
