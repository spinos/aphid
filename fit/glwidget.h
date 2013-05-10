/*
 *  glwidget.h
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <SingleModelView.h>

class FitDeformer;

//! [0]
class GLWidget : public SingleModelView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
    
    virtual bool removeLastAnchor(unsigned & idx);
	virtual bool removeActiveAnchor(unsigned & idx);

	virtual void clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit);
	virtual void buildTree();
	virtual void loadMesh(std::string filename);
	
	void startDeform();
	void setTarget(AnchorGroup * src, KdTree * tree);
	void fit();
	void fitAnchors(Anchor * src, Anchor * dst);
	
protected:
    void keyPressEvent(QKeyEvent *event);
    
signals:
     void needTargetRedraw();
    
private:
	FitDeformer * m_deformer;
	AnchorGroup * m_targetAnchors;
private slots:
	
public slots:
};
//! [3]

#endif
