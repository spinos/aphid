/*
 *  MlUVView.h
 *  mallard
 *
 *  Created by jian zhang on 10/2/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Base2DView.h>
class MlFeatherCollection;
class MlFeather;
class MlUVView : public Base2DView {
Q_OBJECT

public:
    MlUVView(QWidget *parent = 0);
    ~MlUVView();
	
	virtual void clientDraw();
	virtual void clientSelect();
	virtual void clientMouseInput();
    
	static MlFeatherCollection * FeatherLibrary;

private:
	bool pickupFeather(const Vector2F & p);
	void drawFeather(MlFeather * f);
	void drawControlVectors(MlFeather * f);
	void drawActiveBound();
private:
	int m_activeId;
};