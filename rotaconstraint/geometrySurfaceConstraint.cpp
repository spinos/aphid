#include <geometrySurfaceConstraint.h>

MTypeId     geometrySurfaceConstraint::id( 0x33f77896 );
MObject     geometrySurfaceConstraint::compoundTarget;        
MObject     geometrySurfaceConstraint::targetGeometry;       
MObject     geometrySurfaceConstraint::targetWeight;       
MObject     geometrySurfaceConstraint::constraintParentInverseMatrix;       
MObject     geometrySurfaceConstraint::constraintGeometry;       

geometrySurfaceConstraint::geometrySurfaceConstraint() 
{
	weightType = rotaBase::kLargestWeight;
}

geometrySurfaceConstraint::~geometrySurfaceConstraint() 
{
}

void geometrySurfaceConstraint::postConstructor()
{
}

MStatus geometrySurfaceConstraint::compute( const MPlug& plug, MDataBlock& block )
{	
	MStatus returnStatus;
 
	if ( plug == geometrySurfaceConstraint::constraintGeometry )
	{
		//
		block.inputValue(constraintParentInverseMatrix);
		//
		MArrayDataHandle targetArray = block.inputArrayValue( compoundTarget );
		unsigned int targetArrayCount = targetArray.elementCount();
		double weight,selectedWeight = 0;
		if ( weightType == rotaBase::kSmallestWeight )
			selectedWeight = FLT_MAX;
		MObject selectedMesh;
		unsigned int i;
		for ( i = 0; i < targetArrayCount; i++ )
		{
			MDataHandle targetElement = targetArray.inputValue();
			weight = targetElement.child(targetWeight).asDouble();
			if ( !equivalent(weight,0.0))
			{
				if ( weightType == rotaBase::kLargestWeight )
				{
					if ( weight > selectedWeight )
					{
						MObject mesh = targetElement.child(targetGeometry).asMesh();
						if ( !mesh.isNull() )
						{
							selectedMesh = mesh;
							selectedWeight =  weight;
						}
					}
				}
				else
				{
					if  ( weight < selectedWeight )
					{
						MObject mesh = targetElement.child(targetGeometry).asMesh();
						if ( !mesh.isNull() )
						{
							selectedMesh = mesh;
							selectedWeight =  weight;
						}
					}
				}
			}
			targetArray.next();
		}
		//
		if ( selectedMesh.isNull() )
		{
			block.setClean(plug);
		}
		else
		{
			// The transform node via the geometry attribute will take care of
			// updating the location of the constrained geometry.
			MDataHandle outputConstraintGeometryHandle = block.outputValue(constraintGeometry);
			outputConstraintGeometryHandle.setMObject(selectedMesh);
		}
	} 
	else 
	{
		return MS::kUnknownParameter;
	}

	return MS::kSuccess;
}

const MObject geometrySurfaceConstraint::weightAttribute() const
{
	return geometrySurfaceConstraint::targetWeight;
}

const MObject geometrySurfaceConstraint::targetAttribute() const
{
	return geometrySurfaceConstraint::compoundTarget;
}

void geometrySurfaceConstraint::getOutputAttributes(MObjectArray& attributeArray)
{
	attributeArray.clear();
	attributeArray.append( geometrySurfaceConstraint::constraintGeometry );
}

void* geometrySurfaceConstraint::creator()
{
	return new geometrySurfaceConstraint();
}

MStatus geometrySurfaceConstraint::initialize()
{
	MFnNumericAttribute nAttr;
	MStatus				status;

	// constraint attributes

	{	// Geometry: mesh, readable, not writable, delete on disconnect
		MFnTypedAttribute typedAttrNotWritable;
		geometrySurfaceConstraint::constraintGeometry =
			typedAttrNotWritable.create( "constraintGeometry", "cg", MFnData::kMesh, &status ); 	
		if (!status) { status.perror("typedAttrNotWritable.create:cgeom"); return status;}
		status = typedAttrNotWritable.setReadable(true);
		if (!status) { status.perror("typedAttrNotWritable.setReadable:cgeom"); return status;}
		status = typedAttrNotWritable.setWritable(false);
		if (!status) { status.perror("typedAttrNotWritable.setWritable:cgeom"); return status;}
		status = typedAttrNotWritable.setDisconnectBehavior(MFnAttribute::kDelete);
		if (!status) { status.perror("typedAttrNotWritable.setDisconnectBehavior:cgeom"); return status;}
	}
	{	// Parent inverse matrix: delete on disconnect
		MFnTypedAttribute typedAttr;
		geometrySurfaceConstraint::constraintParentInverseMatrix =
			typedAttr.create( "constraintPim", "ci", MFnData::kMatrix, &status ); 	
		if (!status) { status.perror("typedAttr.create:matrix"); return status;}
		status = typedAttr.setDisconnectBehavior(MFnAttribute::kDelete);
		if (!status) { status.perror("typedAttr.setDisconnectBehavior:cgeom"); return status;}

		// Target geometry: mesh, delete on disconnect
		geometrySurfaceConstraint::targetGeometry =
			typedAttr.create( "targetGeometry", "tg", MFnData::kMesh, &status ); 	
		if (!status) { status.perror("typedAttr.create:tgeom"); return status;}
		status = typedAttr.setDisconnectBehavior(MFnAttribute::kDelete);
		if (!status) { status.perror("typedAttr.setDisconnectBehavior:cgeom"); return status;}
	}
	{	// Target weight: double, min 0, default 1.0, keyable, delete on disconnect
		MFnNumericAttribute typedAttrKeyable;
		geometrySurfaceConstraint::targetWeight 
			= typedAttrKeyable.create( "weight", "wt", MFnNumericData::kDouble, 1.0, &status );
		if (!status) { status.perror("typedAttrKeyable.create:weight"); return status;}
		status = typedAttrKeyable.setMin( (double) 0 );
		if (!status) { status.perror("typedAttrKeyable.setMin"); return status;}
		status = typedAttrKeyable.setKeyable( true );
		if (!status) { status.perror("typedAttrKeyable.setKeyable"); return status;}
		status = typedAttrKeyable.setDisconnectBehavior(MFnAttribute::kDelete);
		if (!status) { status.perror("typedAttrKeyable.setDisconnectBehavior:cgeom"); return status;}
	}
	{	// Compound target(geometry,weight): array, delete on disconnect
		MFnCompoundAttribute compoundAttr;
		geometrySurfaceConstraint::compoundTarget = 
			compoundAttr.create( "target", "tgt",&status );
		if (!status) { status.perror("compoundAttr.create"); return status;}
		status = compoundAttr.addChild( geometrySurfaceConstraint::targetGeometry );
		if (!status) { status.perror("compoundAttr.addChild"); return status;}
		status = compoundAttr.addChild( geometrySurfaceConstraint::targetWeight );
		if (!status) { status.perror("compoundAttr.addChild"); return status;}
		status = compoundAttr.setArray( true );
		if (!status) { status.perror("compoundAttr.setArray"); return status;}
		status = compoundAttr.setDisconnectBehavior(MFnAttribute::kDelete);
		if (!status) { status.perror("typedAttrKeyable.setDisconnectBehavior:cgeom"); return status;}
	}

	status = addAttribute( geometrySurfaceConstraint::constraintParentInverseMatrix );
	if (!status) { status.perror("addAttribute"); return status;}
	status = addAttribute( geometrySurfaceConstraint::constraintGeometry );
	if (!status) { status.perror("addAttribute"); return status;}
	status = addAttribute( geometrySurfaceConstraint::compoundTarget );
	if (!status) { status.perror("addAttribute"); return status;}

	status = attributeAffects( compoundTarget, constraintGeometry );
	if (!status) { status.perror("attributeAffects"); return status;}
	status = attributeAffects( targetGeometry, constraintGeometry );
	if (!status) { status.perror("attributeAffects"); return status;}
	status = attributeAffects( targetWeight, constraintGeometry );
	if (!status) { status.perror("attributeAffects"); return status;}
	status = attributeAffects( constraintParentInverseMatrix, constraintGeometry );
	if (!status) { status.perror("attributeAffects"); return status;}

	return MS::kSuccess;
}
