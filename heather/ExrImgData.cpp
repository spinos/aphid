#include "ExrImgData.h"

void* ExrImgData::creator()
{
    return new ExrImgData;
}


void ExrImgData::copy ( const MPxData& other )
{
	_pDesc = ((const ExrImgData&)other)._pDesc;
}

MTypeId ExrImgData::typeId() const
{
	return ExrImgData::id;
}

MString ExrImgData::name() const
{ 
	return ExrImgData::typeName;
}

MStatus ExrImgData::readASCII(  const MArgList& args,
                                unsigned& lastParsedElement )
{
    return MS::kSuccess;
}

MStatus ExrImgData::writeASCII( ostream& out )
{
    //out << fValue << " ";
    return MS::kSuccess;
}

MStatus ExrImgData::readBinary( istream& in, unsigned )
{
    //in.read( (char*) &fValue, sizeof( fValue ));
    //return in.fail() ? MS::kFailure : MS::kSuccess;
    return MS::kSuccess;
}

MStatus ExrImgData::writeBinary( ostream& out )
{
    //out.write( (char*) &fValue, sizeof( fValue));
    //return out.fail() ? MS::kFailure : MS::kSuccess;
    return MS::kSuccess;;
}

//
// this is the unique type id used to identify the new user-defined type.
//
const MTypeId ExrImgData::id( 0x17c98f0b );
const MString ExrImgData::typeName( "heatherExrImgData" );
