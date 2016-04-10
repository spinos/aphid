/*
 *  voxWidget.h
 *  
 *
 *  Created by jian zhang on 4/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef VOX_Widget_H
#define VOX_Widget_H

#include <Base3DView.h>
#include <VoxelEngine.h>
#include <ConvexShape.h>

class VoxWidget : public aphid::Base3DView 
{
	Q_OBJECT
	
public:
    
    VoxWidget(QWidget *parent = 0);
    virtual ~VoxWidget();

protected:    
    virtual void clientInit();
    virtual void clientDraw();
	virtual void clientMouseInput(QMouseEvent *event);
    virtual void keyPressEvent(QKeyEvent *event);
	
private:
	void buildTests();
	void drawGrids();
	void drawTriangles();
	void drawFronts();
	aphid::cvx::Triangle createTriangle(const aphid::Vector3F & p0,
								const aphid::Vector3F & p1,
								const aphid::Vector3F & p2,
								const aphid::Vector3F & c0,
								const aphid::Vector3F & c1,
								const aphid::Vector3F & c2);
	
private:
	aphid::VoxelEngine<aphid::cvx::Triangle > m_engine;
	aphid::VoxelEngine<aphid::cvx::Triangle > m_engine1;
	aphid::VoxelEngine<aphid::cvx::Triangle > m_engine2;
	aphid::VoxelEngine<aphid::cvx::Triangle > m_engine3;
	aphid::VoxelEngine<aphid::cvx::Triangle > m_engine4;
};
#endif