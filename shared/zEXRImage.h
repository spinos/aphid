#ifndef ZFN_EXR_H
#define ZFN_EXR_H
#include <BaseImage.h>
#include <vector>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfArray.h>

#include <OpenEXR/ImfBoxAttribute.h>
#include <OpenEXR/ImfChannelListAttribute.h>
#include <OpenEXR/ImfCompressionAttribute.h>
#include <OpenEXR/ImfChromaticitiesAttribute.h>
#include <OpenEXR/ImfFloatAttribute.h>
#include <OpenEXR/ImfEnvmapAttribute.h>
#include <OpenEXR/ImfDoubleAttribute.h>
#include <OpenEXR/ImfIntAttribute.h>
#include <OpenEXR/ImfLineOrderAttribute.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfOpaqueAttribute.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <OpenEXR/ImfVecAttribute.h>

class ZEXRImage;

class ZEXRSampler
{
public:
	ZEXRSampler();
	~ZEXRSampler();
	void setPixels(ZEXRImage * src);
	void reduceFrom(ZEXRSampler * src);
	void setWidth(int w);
	half* getPixels() const;
	int getWidth() const;
	void sample(float u, float v, int count, float * dst) const;
	half *_pixels;
	int _width, _rank;
};

class ZEXRImage : public BaseImage
{
public:
	ZEXRImage();
	ZEXRImage(const char* filename);
	~ZEXRImage(void);
	
	virtual bool doRead(const std::string & filename);
	virtual void doClear();
	virtual const char * formatName() const;
	
	virtual void allWhite();
	virtual void allBlack();
	
	virtual void sample(float u, float v, int count, float * dst) const;
	
	void sample(float u, float v, float level, int count, float * dst) const;
	
	int getNumMipmaps() const;
	
	virtual void applyMask(BaseImage * another);
	
	static bool isAnOpenExrFile(const std::string& filename);
	
	half *_pixels;
	
	std::vector<ZEXRSampler*>_mipmaps;
	int _numMipmaps;
	
private:
	void readPixels(Imf::InputFile& file);
	void setupMipmaps();
};
#endif
