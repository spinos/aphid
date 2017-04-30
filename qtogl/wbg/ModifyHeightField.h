/*
 *  ModifyHeightField.h
 *  
 *
 *  Created by jian zhang on 3/25/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WBG_MODIFY_HEIGHT_FIELD_H
#define WBG_MODIFY_HEIGHT_FIELD_H

#include <math/Vector2F.h>

namespace aphid {

class Ray;
class Plane;

}

class DrawHeightField;

class ModifyHeightField {

	int m_heightFieldToolFlag;
	DrawHeightField * m_heightDrawer;
	aphid::Plane * m_plane;
	aphid::Vector2F m_lastWorldP;
	aphid::Vector2F m_lastLocalP;
	aphid::Vector2F m_worldCenterP;
	bool m_isActive;
	
public:
	ModifyHeightField();
	virtual ~ModifyHeightField();
	
protected:
	void setHeightFieldToolFlag(int x);
	const int & heightFieldToolFlag() const;
	void selectHeightField(int x);
	void drawHeightField();
	
	void beginModifyHeightField(const aphid::Ray * incident);
	void endModifyeHeightField();
	void doMoveHeightField(const aphid::Ray * incident);
	void doRotateHeightField(const aphid::Ray * incident);
	void doResizeHeightField(const aphid::Ray * incident);
	
private:
	bool updateActiveState(aphid::Vector2F & wldv, aphid::Vector2F & objv,
				const aphid::Ray * incident);
	
};
#endif