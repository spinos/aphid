need xmlrpc-c

http://sourceforge.net/projects/xmlrpc-c/files/

To build shotgunAPI.lib

Edit CMakeList.txt for proper definitions

set (SG_DEFAULT_URL "localhost")
set (SG_AUTHENTICATION_NAME "default")
set (SG_AUTHENTICATION_KEY "unknown")

Copy CMakeList.txt to apiRoot/lib/Shotgun
Edit Shotgun.h to add following line

#include "shotgunAPIConfig.h"

CMake 
set source to apiRoot/lib/Shotgun 
set binary to musket
gen and build

To build plugin

Edit sip/Shotgun.sip to remove const after deleteEntity()
Edit mappedTypes.sip to replace 27000100 to 0100, replace time_t related long to long long
