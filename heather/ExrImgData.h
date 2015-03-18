#pragma once

#include <maya/MPxData.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <StripeCompressedImage.h>
class ExrImgData : public MPxData
{
public:
	struct DataDesc
	{
	    StripeCompressedRGBAZImage * _compressedImg;
		bool _isValid;
	};
						ExrImgData() {}
    virtual					~ExrImgData() {}

    virtual MStatus         readASCII( const MArgList&, unsigned& lastElement );
    virtual MStatus         readBinary( istream& in, unsigned length );
    virtual MStatus         writeASCII( ostream& out );
    virtual MStatus         writeBinary( ostream& out );

	virtual void			copy( const MPxData& );
	MTypeId                 typeId() const; 
	MString                 name() const;

    	DataDesc * getDesc() const { return _pDesc; }
    	void setDesc(DataDesc * pDesc) { _pDesc = pDesc; }

	static const MString	typeName;
    static const MTypeId    id;
	static void*            creator();

private:
    DataDesc * _pDesc; 
};

