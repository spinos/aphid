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
