TARGET = foo
SOURCES       = main.cpp 
OBJECTS       = main.o 

CC = cc
CXX = c++
LINK = c++
DEFINES =
CPPFLAGS = -DNDEBUG $(DEFINES) -I. -I/Users/jianzhang/Library/CHOLMOD/Include -I/Users/jianzhang/Library/SuiteSparse_config
CFLAGS = -pipe -fPIC -Os -Wall -W $(DEFINES)
CXXFLAGS = -pipe -fPIC -Os -Wall -W
LFLAGS = 
LIBS = -L/Users/jianzhang/Library/SuiteSparse_config -lsuitesparseconfig \
        -L/Users/jianzhang/Library/AMD/Lib -lamd \
        -L/Users/jianzhang/Library/CAMD/Lib -lcamd \
        -L/Users/jianzhang/Library/COLAMD/Lib -lcolamd \
        -L/Users/jianzhang/Library/CCOLAMD/Lib -lccolamd \
        -L/Users/jianzhang/Library/CHOLMOD/Lib -lcholmod
        
.SUFFIXES: .c .o .cpp .cc .cxx .C


.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.C.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

all: $(TARGET)

$(TARGET):  $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

clean:
	-rm -f $(OBJECTS)
	-rm -f $(TARGET)
