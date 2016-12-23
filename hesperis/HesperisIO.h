/*
 *  HesperisIO.h
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MMatrix.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MObject.h>
#include <maya/MObjectArray.h>
#include <AllMath.h>
#include <BaseTransform.h>
#include <HTransform.h>
#include <foundation/SHelper.h>
#include <string>
#include <map>
#include "HesperisFile.h"
#include "H5IO.h"

namespace aphid {
    
class CurveGroup;
class ATriangleMeshGroup;

class HesperisTransformCreator {
public:
    static MObject create(BaseTransform * data, MObject & parentObj,
                       const std::string & nodeName);
};

class HesperisIO : public H5IO {
public:
	static bool WriteTransforms(const MDagPathArray & paths, HesperisFile * file );
    static bool AddTransform(const MDagPath & path, HesperisFile * file );
	static bool WriteMeshes(const MDagPathArray & paths, 
							HesperisFile * file, 
							const std::string & parentName = "");
    static bool CreateMeshGroup(const MDagPathArray & paths, 
								ATriangleMeshGroup * dst);
	static void LsChildren(MObjectArray & dst, 
	            const int & maxCount,
	            const MObject & oparent);
    static bool FindNamedChild(MObject & dst, 
                const std::string & name,
                const MObject & oparent = MObject::kNullObj);
    static bool GetTransform(BaseTransform * dst, const MDagPath & path);
    static bool LsCurves(std::vector<std::string > & dst);
    static bool LsMeshes(std::vector<std::string > & dst);
    static bool LsTransforms(std::vector<std::string > & dst);
	static Matrix33F::RotateOrder GetRotationOrder(MTransformationMatrix::RotationOrder x);

	static std::string H5PathNameTo(const MDagPath & path);
	static std::string H5PathNameTo(const MObject & node);
	static std::string CurrentHObjectPath;
    static MPoint GlobalReferencePoint;
    
protected:
    template<typename Th, typename Td, typename Tc>
    static bool ReadTransformAnd(HBase * parent, MObject &target)
    {
        std::vector<std::string > tmNames;
        parent->lsTypedChild<HTransform>(tmNames);
        std::vector<std::string>::const_iterator it = tmNames.begin();
        
        for(;it!=tmNames.end();++it) {
            std::string nodeName = *it;
            SHelper::behead(nodeName, parent->pathToObject());
            SHelper::behead(nodeName, "/");

            HTransform child(*it);
            
			BaseTransform dtrans;
            child.load(&dtrans);
            MObject otm = HesperisTransformCreator::create(&dtrans, target, nodeName);
            
            ReadTransformAnd<Th, Td, Tc>(&child, otm);
            child.close();
        }
        
        std::vector<std::string > polyNames;
        parent->lsTypedChild<Th>(polyNames);
        std::vector<std::string>::const_iterator ita = polyNames.begin();
        
        for(;ita !=polyNames.end();++ita) {
            std::string nodeName = *ita;
            SHelper::behead(nodeName, parent->pathToObject());
            SHelper::behead(nodeName, "/");
            
            Th child(*ita);
            CurrentHObjectPath = child.pathToObject();
            
            Td data;
            child.load(&data);
            Tc::create(&data, target, nodeName);
            child.close();
        }
        return true;
    }
    
};

}
