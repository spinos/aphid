/*
 *  TriWidget.h
 *  
 *
 *  Created by jian zhang on 3/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TRIWidget_H
#define TRIWidget_H

#include <Base3DView.h>
#include <IntersectionContext.h>
#include "Container.h"

class TriWidget : public Base3DView
{
    Q_OBJECT
	
public:
    
    TriWidget(const std::string & filename, QWidget *parent = 0);
    ~TriWidget();
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
	
private:
    void drawTriangle();
    void drawTree();
	void testIntersect(const Ray * incident);
	void testTriangleIntersection(const Ray * incident);
	void testVoxelIntersection(const Ray * incident);
	void drawIntersect();
	void drawVoxelIntersect();
	void drawActiveSource(const unsigned & iLeaf);
	void drawActiveVoxel(const unsigned & iLeaf);
	void drawVoxel();
	void drawVoxelTree();
	BoundingBox getFrameBox();
	
private slots:
	
private:
	IntersectionContext m_intersectCtx;
	Container<cvx::Triangle > m_container;
	enum PickTreeType {
		tTriangle = 0,
		tVoxel = 1
	};
	
	PickTreeType m_pickTree;
};

#endif
