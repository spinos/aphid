#ifndef EXRIMAGE_H
#define EXRIMAGE_H

#include "BaseImage.h"

namespace aphid {
class ExrImage : public BaseImage
{
    char *_pixels;
    
public:
	ExrImage();
	virtual ~ExrImage();
	
	virtual bool doRead(const std::string & filename);
	virtual void doClear();
	virtual const char * formatName() const;
	
	
	bool getTile(float * dst, const int ind, int tileSize, int rank = 3) const;
	
	static bool IsOpenExrFile(const std::string& filename);
	
private:
    void tileCoord(const int ind, const int tileSize, int & x, int & y) const;
	
};
}
#endif        //  #ifndef EXRIMAGE_H

