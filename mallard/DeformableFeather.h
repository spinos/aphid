/*
 *  DeformableFeather.h
 *  mallard
 *
 *  Created by jian zhang on 1/6/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "TexturedFeather.h"
class DeformableFeather : public TexturedFeather {
public:
	struct BindCoord {
		Vector3F _objP;
		float _taper;
		short _u, _v, _rgt;
	};
	
	DeformableFeather();
	virtual ~DeformableFeather();
	virtual void computeTexcoord();
	virtual void createVanes();
	
	short numBind(short seg) const;
	BindCoord * getBind(short seg, short idx) const;
protected:

private:
	void computeBinding();
	void bindVane(BaseVane * vane, short rgt);
private:
	struct BindGroup {
		BindGroup() {_bind = 0;}
		~BindGroup() {
			if(_bind) delete _bind;
		}
		BindCoord * _bind;
		short _numBind;
	};
	
	BindGroup * m_group;
};