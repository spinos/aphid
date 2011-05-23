PRODUCT = constant
CC = gcc
CXX = g++
INCPATH = /Applications/Autodesk/maya2008/devkit/mentalray/include
LINK = g++
SRCS := illumconstant.cpp
OBJS := $(patsubst %.cpp,%.o,$(filter %.cpp, $(SRCS)))

all: $(PRODUCT).so

$(OBJS): %.o: %.cpp
	$(CXX) -I$(INCPATH) -c $< -o $@

$(PRODUCT).so: illumconstant.o
	$(LINK) -flat_namespace -undefined suppress -dynamiclib -o $(PRODUCT).so  $(OBJS)

clean:
	@-rm $(OBJS)
	@-rm $(PRODUCT).so

