#pragma once
#include <Alembic/AbcGeom/IXform.h>
#include <Alembic/AbcGeom/XformOp.h>

class ALITransform {
public:
	ALITransform(Alembic::AbcGeom::IXform &obj);
	~ALITransform();
	/*
	void addTranslate(const double &tx, const double &ty, const double &tz, Alembic::Util::uint8_t hint);
	void addTranslate(const double &tx, const double &ty, const double &tz);
	void addScalePivotTranslate(const double &tx, const double &ty, const double &tz);
	void addRotatePivotTranslate(const double &tx, const double &ty, const double &tz);
	void addScalePivot(const double &tx, const double &ty, const double &tz);
	void addRotatePivot(const double &tx, const double &ty, const double &tz);
	void addRotate(const double &rx, const double &ry, const double &rz, const int &order);
	void addScale(const double &sx, const double &sy, const double &sz);
	void write();*/
	void verbose();
private:
	Alembic::AbcGeom::IXformSchema m_schema;
    Alembic::AbcGeom::XformSample m_sample;
};
