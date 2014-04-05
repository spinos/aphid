// Define the SIP wrapper to the word library.

%Module (name=word, language="C++")

%Exception MY::Ooops(SIP_Exception) /PyName=StdException/
{
%TypeHeaderCode
#include <exception>
#include <iostream>
%End
%RaiseCode
        const char *detail = sipExceptionRef.what();
        SIP_BLOCK_THREADS
        PyErr_SetString(PyExc_RuntimeError, detail);
        SIP_UNBLOCK_THREADS
%End
};

%MappedType std::string
{
%TypeHeaderCode
#include <string>
#include <iostream>
%End

%ConvertFromTypeCode
    // convert an std::string to a Python string
    // I removed the part that converts to a Python unicode string.
    PyObject* newstring;
    newstring = PyUnicode_FromString(sipCpp->c_str());
    
    return newstring;
%End

%ConvertToTypeCode
    // Allow a Python string (or a unicode string) whenever a string is
    // expected.
    // If argument is a Unicode string, just decode it to UTF-8
    // If argument is a Python string, assume it's UTF-8
    if (sipIsErr == NULL)
    {
        return PyUnicode_Check(sipPy);
    }
    if (sipPy == Py_None)
    {
        *sipCppPtr = new std::string;
        return 1;
    }
    if (PyUnicode_Check(sipPy))
    {
        *sipCppPtr = new std::string(PyUnicode_AsUTF8(sipPy));
        Py_DECREF(sipPy);
        return 1;
    }

    return 0;
%End
};

%MappedType std::vector<std::string>
{
%TypeHeaderCode
#include <vector>
%End

%ConvertFromTypeCode
    PyObject *pList = NULL;

    // Create the Python list of the correct length.
    if ((pList = PyList_New(sipCpp->size())) == NULL)
        return NULL;

    // Go through each element in the C++ instance and convert it to the
    // corresponding Python object.
    for (SIP_SSIZE_T i = 0; i < SIP_SSIZE_T(sipCpp->size()); ++i)
    {
        PyObject *pObj;

        if ((pObj = PyUnicode_FromString((*sipCpp)[i].c_str())) == NULL)
        {
            Py_DECREF(pList);

            return NULL;
        }

        PyList_SET_ITEM(pList, i, pObj);
    }

    // Return the Python list.
    return pList;
%End

%ConvertToTypeCode
    if (sipIsErr == NULL)
    {
        return (PySequence_Check(sipPy) && PySequence_Size(sipPy) >= 0);
    }

    std::vector<std::string> *stdVector = new std::vector<std::string>;
    SIP_SSIZE_T len = PySequence_Size(sipPy);

    for (SIP_SSIZE_T i = 0; i < len; ++i)
    {
        PyObject *itm = PySequence_ITEM(sipPy, i);

        if (!itm)
        {
            delete stdVector;
            *sipIsErr = 1;

            return 0;
        }

        stdVector->push_back(PyUnicode_AsUTF8(itm));

        Py_DECREF(itm);
    }

    // Return the instance on the heap.
    *sipCppPtr = stdVector;

    // Apply the normal transfer.
    return sipGetState(sipTransferObj);
%End
};

namespace MY {
class Word {

%TypeHeaderCode
#include <word.h>
#include <exception>
%End

public:
    Word();

    std::string reversed(const std::string & w) const;
    std::string str(const std::string & name) const;
    std::string strvec(const std::vector<std::string> & name) const;
    int someerr() const throw(MY::Ooops);
};
};
