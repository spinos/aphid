need xmlrpc-c

http://sourceforge.net/projects/xmlrpc-c/files/

To build shotgunAPI.lib

copy all lib source into /Shotgun
Edit shotgunAPIConfig.h.in for proper authorization
Edit Shotgun.h to add #include <shotgunAPIConfig.h>
Edit utils.h to #include <algorithm>
Edit exceptions.cpp to add #include<Shotgun/..>
To build plugin

need latest sip

goto plugins/python add src dir
python configure.py
cd src to make then make install
Edit sip/Shotgun.sip to remove const after deleteEntity()
Edit mappedTypes.sip to replace 27000100 to 0100, replace time_t related long to long long

Note:
SG_DEFAULT_URL should be http://whatever/api3/
