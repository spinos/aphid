// Define the SIP wrapper to the word library.


%Module hoatzin 0

class Hoatzin {

%TypeHeaderCode
#include <hoatzin.h>
%End

public:
    Hoatzin();
    int set_attribute_int(const char *widgetName, const char *attributeName, int value);

};
