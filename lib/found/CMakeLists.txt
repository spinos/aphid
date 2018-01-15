file (GLOB Foundation_SRC
"../../shared/foundation/*.h"
    "../../shared/foundation/*.cpp"
)

ADD_LIBRARY (aphidFoundation ${LIB_TYP}
${Foundation_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
