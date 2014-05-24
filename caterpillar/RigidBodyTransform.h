#pragma once
#include <maya/MPxTransform.h>
#include <maya/MPxTransformationMatrix.h>
#include <maya/MTransformationMatrix.h>

namespace caterpillar {

class RigidBodyTransformMatrix : public MPxTransformationMatrix
{
	public:
		RigidBodyTransformMatrix();
		static void *creator();
		
		virtual MMatrix asMatrix() const;
		//virtual MMatrix	asMatrix(double percent) const;
		//virtual MMatrix	asRotateMatrix() const;
		 void	setRockInX( float space[16]);
		
		static	MTypeId	id;
	protected:		
		typedef MPxTransformationMatrix ParentClass;
		float fm[4][4];
};

class RigidBodyTransformNode : public MPxTransform 
{
	public:
		RigidBodyTransformNode();
		RigidBodyTransformNode(MPxTransformationMatrix *);
		virtual ~RigidBodyTransformNode();

		virtual MPxTransformationMatrix *createTransformationMatrix();
			
		virtual void postConstructor();

		virtual MStatus validateAndSetValue(const MPlug& plug,
			const MDataHandle& handle, const MDGContext& context);
		
		virtual void  resetTransformation (MPxTransformationMatrix *);
		virtual void  resetTransformation (const MMatrix &);
					
		RigidBodyTransformMatrix *getRigidBodyTransformMatrix();
				
		const char* className();
		static	void * 	creator();
		static  MStatus	initialize();

		static	MTypeId	id;
		static MObject a_objectId;
		static MObject a_inSolver;
	protected:
		typedef MPxTransform ParentClass;
};

}
