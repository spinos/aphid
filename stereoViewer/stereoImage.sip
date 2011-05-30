// Define the SIP wrapper to the word library.

%Module StereoImage 0

class StereoImage {

%TypeHeaderCode
#include <stereoImage.h>
%End

public:
    StereoImage();
    
    void setSize(int w, int h);
    void setLeftImage(void *d);
    void setRightImage(void *d);

    void *display();
};
