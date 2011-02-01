#include <Python.h>
#include <iostream>
using namespace std;

int
main() 
  {    
	Py_Initialize();

	cout<<"version "<<Py_GetVersion()<<endl;

// equals to
// >>> import os
// >>> os.path.expanduser('~')	
	PyObject *osPath = PyImport_AddModule("os.path");
	PyObject *pathHome = PyObject_CallMethod(osPath, "expanduser", "(s)", "~");

// extract string
	cout<<"$Home="<<PyString_AsString(pathHome)<<endl;

// equals to
// >>> import random
// >>> random.random()	
	PyObject *randomName = PyString_FromString("random");
	PyObject *randomModule = PyImport_Import(randomName);
	PyObject *randomRandom = PyObject_CallMethod(randomModule, "random", NULL);
	
// extract double floating number
	cout<<"random random is "<<PyFloat_AsDouble(randomRandom)<<endl;	

// >>> random.uniform(9.0, 9.1)	
	float min = 9.f;
	float max = 9.1f;
	PyObject *pMin = Py_BuildValue("f", min); 
	PyObject *pMax = Py_BuildValue("f", max); 
	
	PyObject *randomUniform = PyObject_CallMethodObjArgs(randomModule, PyString_FromString("uniform"), pMin, pMax, NULL);
	cout<<"random uniform(9, 9.1) is "<<PyFloat_AsDouble(randomUniform)<<endl;

// equals to
// >>> execfile('foo.py')
// >>> a = foo()
// >>> a.getAge()
// >>> a.add(37, 59)
// >>> a.log(141, 'Tom', 1961)	

#ifdef WIN32
	FILE *fin = fopen("D:/aphid/pyobject/foo.py", "r+");
#else
        FILE *fin = fopen("/Users/jianzhang/aphid/pyobject/foo.py", "r+");
#endif
	PyRun_SimpleFile(fin, "foo");
	fclose(fin);
	
	PyObject *mainModule = PyImport_Import(PyString_FromString("__main__"));
	PyObject *mainDict = PyModule_GetDict(mainModule);

	PyObject *fooClass = PyDict_GetItemString(mainDict, "foo");
	PyObject *fooInstance = PyObject_CallObject(fooClass, NULL);
	
	PyObject *fooAge = PyObject_CallMethod(fooInstance, "getAge", NULL);
// extract integer number	
	cout<<"age: "<<PyInt_AsLong(fooAge)<<endl;
	
	PyObject *fooAdd = PyObject_CallMethod(fooInstance, "add", "ii", 37, 59);
	cout<<"add: "<<PyInt_AsLong(fooAdd)<<endl;
    
	PyObject *fooLog = PyObject_CallMethod(fooInstance, "log", "isi", 141, "Tom", 1961);
// extract string
	cout<<"log: "<<PyString_AsString(fooLog)<<endl;

// equals
// >>> l = loaf(13, 31)
// >>> l.minus()

// get name of class to create
	PyObject *loafName = PyDict_GetItemString(mainDict, "inst_name");
	cout<<"class name: "<<PyString_AsString(loafName)<<endl;
// get arg list
	PyObject *loafArgv = PyDict_GetItemString(mainDict, "inst_argv");

	PyObject *loafClass = PyDict_GetItem(mainDict, loafName);
	PyObject *loafInstance = PyObject_CallObject(loafClass, loafArgv);
	
// check 
	PyObject *loafMinus = PyObject_CallMethod(loafInstance, "minus", NULL);	
	cout<<"13 - 31 equals "<<PyInt_AsLong(loafMinus)<<endl;
	
	PyErr_Print();

	Py_Finalize();
    
    return 0;
  };
