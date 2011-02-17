// Define the SIP wrapper to the word library.

%Module FooImage 0

class FooImage {

%TypeHeaderCode
#include <FooImage.h>
%End

public:
    FooImage(const char *);
    
    void setSize(int w, int h);
    void addColor(void *d);

    void *display() const;
    
    void setRed(float red);
};
