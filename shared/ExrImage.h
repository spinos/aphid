#ifndef APHID_EXR_IMAGE_H
#define APHID_EXR_IMAGE_H

#include "BaseImage.h"

namespace aphid {
class ExrImage : public BaseImage
{
    char *_pixels;
    
public:
	ExrImage();
	virtual ~ExrImage();
	
	virtual IFormat formatName() const;
	
	bool getTile(float * dst, const int ind, int tileSize, int rank = 3) const;
	
	static bool IsOpenExrFile(const std::string& filename);

protected:
	virtual bool readImage(const std::string & filename);
		
private:
    void tileCoord(const int ind, const int tileSize, int & x, int & y) const;
	void clear();
	
};
}
#endif        //  #ifndef APHID_EXR_IMAGE_H

