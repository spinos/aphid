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
/// n x n atom
	int m_atomSize;
/// num predictors
	int m_dictionaryLength;
/// total num patches
	int m_numPatches;
/// dictionary image size
	int m_dictWidth, m_dictHeight;
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
	
	void getDictionaryImageSize(int & x, int & y) const;
	static void PrintHelp();
protected:

private:
	bool searchImagesIn(const char * dirname);
	void countPatches();
};

}
