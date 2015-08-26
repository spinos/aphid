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
#include <maya/MMatrix.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MObject.h>
#include <AllMath.h>
#include <HBase.h>
#include <BaseTransform.h>
#include <HTransform.h>
#include <SHelper.h>
#include <string>
#include <map>

class HesperisFile;
class CurveGroup;
class ATriangleMeshGroup;

class HesperisTransformCreator {
public:
    static MObject create(BaseTransform * data, MObject & parentObj,
                       const std::string & nodeName);
};

class HesperisIO {
public:
	static bool WriteTransforms(const MDagPathArray & paths, HesperisFile * file, const std::string & beheadName = "");
    static bool AddTransform(const MDagPath & path, HesperisFile * file, const std::string & beheadName = "");
    static bool IsCurveValid(const MDagPath & path);
	static bool WriteCurves(const std::map<std::string, MDagPath > & paths, 
							HesperisFile * file, 
							const std::string & parentName = "");
	static bool WriteMeshes(const std::map<std::string, MDagPath > & paths, 
							HesperisFile * file, 
							const std::string & parentName = "");
	static bool CreateCurveGroup(const std::map<std::string, MDagPath > & paths, 
								CurveGroup * dst);
    static bool CreateMeshGroup(const std::map<std::string, MDagPath > & paths, 
								ATriangleMeshGroup * dst);
    static MMatrix GetParentTransform(const MDagPath & path);
    static MMatrix GetWorldTransform(const MDagPath & path);
    static bool FindNamedChild(MObject & dst, const std::string & name, MObject & oparent = MObject::kNullObj);
    static bool GetTransform(BaseTransform * dst, const MDagPath & path);
    static bool LsCurves(std::vector<std::string > & dst);
    static bool LsMeshes(std::vector<std::string > & dst);
    static bool LsTransforms(std::vector<std::string > & dst);
	static Matrix33F::RotateOrder GetRotationOrder(MTransformationMatrix::RotationOrder x);

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
            Td data;
            child.load(&data);
            Tc::create(&data, target, nodeName);
            child.close();
        }
        return true;
    }
};