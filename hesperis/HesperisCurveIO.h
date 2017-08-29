/*
 *  HesperisCurveIO.h
 *  hesperis
 *
 *  Created by jian zhang on 7/12/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisIO.h"
namespace aphid {
    
class HesperisCurveCreator {
public:
	static MObject create(CurveGroup * data, MObject & parentObj,
                       const std::string & nodeName);
	static bool CheckExistingCurves(CurveGroup * geos, MObject &target);
	static bool CreateACurve(Vector3F * pos, unsigned nv, int curveDegree, 
						MObject &target = MObject::kNullObj);
/// rebuild curve to value [0,1] if true
	static bool DoRebuildCurve;
};

class HesperisCurveIO : public HesperisIO {
public:
	static bool IsCurveValid(const MDagPath & path);
	static bool CreateCurveGroup(const MDagPathArray & paths, 
								CurveGroup * dst);
	static bool UpdateCurveGroup(const MDagPathArray & paths, 
								CurveGroup * dst);
	static bool WriteCurves(const MDagPathArray & paths, 
							HesperisFile * file, 
							const std::string & parentName = "");
	static bool ReadCurves(MObject &target = MObject::kNullObj);

};

}
