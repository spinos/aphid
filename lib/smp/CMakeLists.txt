file (GLOB smp_SRC
    "../../shared/smp/*.h"
    "../../shared/smp/*.cpp"
)

ADD_LIBRARY (aphidSmp ${LIB_TYP}
${smp_SRC})

SET (LIBRARY_OUTPUT_PATH ../)
