/*
 *  ALTransform.h
 *  helloAbc
 *
 *  Created by jian zhang on 10/31/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Alembic/AbcGeom/OXform.h>
#include <Alembic/AbcGeom/XformOp.h>

class ALTransform {
public:
	ALTransform(Alembic::Abc::OObject &parent, const std::string &name);
	~ALTransform();
	
	void addTranslate(const double &tx, const double &ty, const double &tz, Alembic::Util::uint8_t hint);
	void addTranslate(const double &tx, const double &ty, const double &tz);
	void addScalePivotTranslate(const double &tx, const double &ty, const double &tz);
	void addRotatePivotTranslate(const double &tx, const double &ty, const double &tz);
	void addScalePivot(const double &tx, const double &ty, const double &tz);
	void addRotatePivot(const double &tx, const double &ty, const double &tz);
	void addRotate(const double &rx, const double &ry, const double &rz, const int &order);
	void addScale(const double &sx, const double &sy, const double &sz);
	void write();
private:
	Alembic::AbcGeom::OXformSchema m_schema;
    Alembic::AbcGeom::XformSample m_sample;
};