file (GLOB Math_SRC
    "../../shared/math/*.h"
    "../../shared/math/*.cpp"
)

ADD_LIBRARY (aphidMath ${LIB_TYP} 
${Math_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
