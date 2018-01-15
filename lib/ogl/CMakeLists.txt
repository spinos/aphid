file (GLOB Ogl_SRC
    "../../shared/ogl/*.h"
    "../../shared/ogl/*.cpp"
)

ADD_LIBRARY (aphidOgl ${LIB_TYP}
${Ogl_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
