file (GLOB Gpr_SRC
    "../../shared/gpr/*.h"
    "../../shared/gpr/*.cpp"
)

ADD_LIBRARY (aphidGp ${LIB_TYP}
${Gpr_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
