/*
 *  ALTransform.cpp
 *  helloAbc
 *
 *  Created by jian zhang on 10/31/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "ALTransform.h"

ALTransform::ALTransform(Alembic::AbcGeom::OXform &obj) 
{
	m_schema = obj.getSchema();
}

ALTransform::~ALTransform() {}

void ALTransform::addTranslate(const double &tx, const double &ty, const double &tz, Alembic::Util::uint8_t hint)
{
	Alembic::AbcGeom::XformOp op(Alembic::AbcGeom::kTranslateOperation, hint);
	op.setChannelValue(0, tx);
	op.setChannelValue(1, ty);
	op.setChannelValue(2, tz);
	m_sample.addOp(op);
}

void ALTransform::addTranslate(const double &tx, const double &ty, const double &tz)
{
	addTranslate(tx, ty, tz, Alembic::AbcGeom::kTranslateHint);
}

void ALTransform::addScalePivotTranslate(const double &tx, const double &ty, const double &tz)
{
	addTranslate(tx, ty, tz, Alembic::AbcGeom::kScalePivotTranslationHint);
}

void ALTransform::addRotatePivotTranslate(const double &tx, const double &ty, const double &tz)
{
	addTranslate(tx, ty, tz, Alembic::AbcGeom::kRotatePivotTranslationHint);
}

void ALTransform::addScalePivot(const double &tx, const double &ty, const double &tz)
{
	addTranslate(tx, ty, tz, Alembic::AbcGeom::kScalePivotPointHint);
}

void ALTransform::addRotatePivot(const double &tx, const double &ty, const double &tz)
{
	addTranslate(tx, ty, tz, Alembic::AbcGeom::kRotatePivotPointHint);
}

void ALTransform::addRotate(const double &rx, const double &ry, const double &rz, const int &order)
{
	int iXYZ[3] = {0, 1, 2};
	if(order == 1) {
		iXYZ[0] = 1;
		iXYZ[1] = 2;
		iXYZ[2] = 0;
	}
	else if(order == 2) {
		iXYZ[0] = 2;
		iXYZ[1] = 0;
		iXYZ[2] = 1;
	}
	else if(order == 3) {
		iXYZ[0] = 0;
		iXYZ[1] = 2;
		iXYZ[2] = 1;
	}
	else if(order == 4) {
		iXYZ[0] = 1;
		iXYZ[1] = 0;
		iXYZ[2] = 2;
	}
	else if(order == 5) {
		iXYZ[0] = 2;
		iXYZ[1] = 1;
		iXYZ[2] = 0;
	}
	
	double rotVal[3] = {rx, ry, rz};
	
	static const Alembic::AbcGeom::XformOperationType rotDir[3] = {
         Alembic::AbcGeom::kRotateXOperation,
         Alembic::AbcGeom::kRotateYOperation,
         Alembic::AbcGeom::kRotateZOperation
    };
	
	for(int i = 2; i > -1; i--) {
		int idx = iXYZ[i];
		double val = rotVal[idx];
		Alembic::AbcGeom::XformOperationType dir = rotDir[idx];
		
		Alembic::AbcGeom::XformOp op(dir, Alembic::AbcGeom::kRotateHint);
        op.setChannelValue(0, Alembic::AbcGeom::RadiansToDegrees(val));
		m_sample.addOp(op);
	}
}

void ALTransform::addScale(const double &sx, const double &sy, const double &sz)
{
	Alembic::AbcGeom::XformOp op(Alembic::AbcGeom::kScaleOperation, Alembic::AbcGeom::kScaleHint);
	op.setChannelValue(0, sx);
	op.setChannelValue(1, sy);
	op.setChannelValue(2, sz);
	m_sample.addOp(op);
}
	
void ALTransform::write() 
{
	m_schema.set(m_sample);
}