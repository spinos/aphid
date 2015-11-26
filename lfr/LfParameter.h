#include <string>
#include <vector>

/// forward declaration 
/// include openexr headers now will cause macros conflictions
class ZEXRImage;

namespace lfr {

class LfParameter {

    struct ImageInd {
		ZEXRImage * _image;
		int _ind;
	};
	
#define MAX_NUM_OPENED_IMAGES 16 
	ImageInd m_openedImages[MAX_NUM_OPENED_IMAGES];
	int m_currentImage;
	
	std::vector<std::string > m_imageNames;
	std::string m_imageName;
	int m_atomSize;
	int m_dictionaryLength;
	int m_numPatches;
	bool m_isValid;
public:
	LfParameter(int argc, char *argv[]);
	virtual ~LfParameter();
	
	bool isValid() const;
	int atomSize() const;
	int dictionaryLength() const;
	int numPatches() const;
	int dimensionOfX() const;
	int randomImageInd() const;
	std::string imageName(int i) const;
	
	bool isImageOpened(const int ind, int & idx) const;
	ZEXRImage *openImage(const int ind);
	
	static void PrintHelp();
protected:

private:
	bool searchImagesIn(const char * dirname);
	void countPatches();
};

}
