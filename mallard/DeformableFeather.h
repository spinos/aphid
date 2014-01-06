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
	DeformableFeather();
	virtual ~DeformableFeather();
	virtual void computeTexcoord();
	
	short numBind(short seg) const;
	Vector3F getBind(short seg, short idx, short & u, short & v, short & side, float & taper) const;
protected:

private:
	void computeBinding();
	void bindVane(BaseVane * vane, short rgt);
private:
	struct BindCoord {
		Vector3F _objP;
		float _taper;
		short _u, _v, _rgt;
	};
	
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