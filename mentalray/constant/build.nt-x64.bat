cl /TP -c /GR /Zp8 /W3 /GS- /GR /GF /Ox /MD /nologo /W3 /Zc:forScope /EHsc /Ob2 -DQMC -DMI_MODULE= -DMI_PRODUCT_RAY -DWIN_NT -DEVIL_ENDIAN -D_WIN32_WINNT=0x0400 -DNV_CG -D_SECURE_SCL=0 -DBIT64 -DHYPERTHREAD -DX86 -I../../include  illumconstant.cpp
link /delayload:opengl32.dll /nologo /nodefaultlib:LIBC.LIB /MAP:mapfile /OPT:NOREF /INCREMENTAL:NO /LIBPATH:..\..\lib /STACK:0x400000 ws2_32.lib user32.lib mpr.lib /DLL /OUT:constant.dll illumconstant.obj ../../nt-x64/lib/shader.lib
mt.exe -nologo -manifest constant.dll.manifest -outputresource:constant.dll;2
