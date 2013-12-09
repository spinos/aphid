/*
 *  CalamusSkin.h
 *  mallard
 *
 *  Created by jian zhang on 12/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <CollisionRegion.h>
class MlCalamus;
class MlCalamusArray;
class MlCluster;
class CalamusSkin : public CollisionRegion {
public:
	struct FloodTable {
		FloodTable() {}
		FloodTable(unsigned i) {
			reset(i);
		}
		
		void reset(unsigned i) {
			faceIdx = i;
			dartBegin = dartEnd = 0;
		}
		
		unsigned faceIdx, dartBegin, dartEnd;
	};
	
	CalamusSkin();
	virtual ~CalamusSkin();
	
	void cleanup();
	
	void getPointOnBody(MlCalamus * c, Vector3F &p) const;
	void getNormalOnBody(MlCalamus * c, Vector3F &p) const;
	
	void tangentSpace(MlCalamus * c, Matrix33F & frm) const;
	void rotationFrame(MlCalamus * c, const Matrix33F & tang, Matrix33F & frm) const;
	void calamusSpace(MlCalamus * c, Matrix33F & frm) const;
	MlCalamusArray * getCalamusArray() const;
	MlCalamus * getCalamus(unsigned idx) const;
	
	void clearFeather();
	void setNumFeathers(unsigned num);
	unsigned numFeathers() const;
	void addFeather(MlCalamus & ori);
	void zeroFeather();
	void reduceFeather(unsigned num);
	
	void clearFaceVicinity();
	void createFaceVicinity();
	void resetFaceVicinity();
	void setFaceVicinity(unsigned idx, float val);
	float faceVicinity(unsigned idx) const;
	
	void createFaceCluster();
	void conputeFaceClustering();
	void getClustering(unsigned idx, std::vector<Vector3F> & dst);
	
	void createFaceCalamusIndirection();
	void resetFaceCalamusIndirection();
	void computeFaceCalamusIndirection();
	void faceCalamusBeginEnd(unsigned faceIdx, unsigned & begin, unsigned & end) const;
protected:
	bool isPointTooCloseToExisting(const Vector3F & pos, float minDistance);

private:
	MlCalamusArray * m_calamus;
	FloodTable * m_faceCalamusTable;
	MlCluster * m_perFaceCluster;
	float * m_perFaceVicinity;
	unsigned m_numFeather;
};