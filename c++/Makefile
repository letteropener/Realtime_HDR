################################################################################
# SaveToAvi Makefile
################################################################################
PROJECT_ROOT=../../
OPT_INC = ${PROJECT_ROOT}/common/make/common_spin.mk
-include ${OPT_INC}

################################################################################
# Key paths and settings
################################################################################
ifeq ($(wildcard ${OPT_INC}),)
CFLAGS += -std=c++11
CXX = g++ ${CFLAGS}
ODIR  = .obj/build${D}
SDIR  = .
MKDIR = mkdir -p
endif

OUTPUTNAME = SaveToAvi${D}
OUTDIR = ../../bin

################################################################################
# Dependencies
################################################################################
# Spinnaker deps
SPINNAKER_LIB = -L../../lib -lSpinnaker${D} ${SPIN_DEPS}

################################################################################
# Master inc/lib/obj/dep settings
################################################################################
_OBJ = SaveToAvi.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
INC = -I../../include -I/usr/include/spinnaker
LIB += -Wl,-Bdynamic ${SPINNAKER_LIB}
LIB += -Wl,-rpath-link=../../lib 

CV_INC=-I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/
CV_FLAGS=  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching
################################################################################
# Rules/recipes
################################################################################
# Final binary
${OUTPUTNAME}: ${OBJ}
	${CXX}   -o ${OUTPUTNAME} ${OBJ} ${LIB} ${CV_INC} ${CV_FLAGS}
	

# Intermediate object files
${OBJ}: ${ODIR}/%.o : ${SDIR}/%.cpp
	@${MKDIR} ${ODIR}
	${CXX} ${CFLAGS} ${INC} ${CV_INC} ${CV_FLAGS} -Wall -D LINUX -c $< -o $@

# Clean up intermediate objects
clean_obj:
	rm -f ${OBJ}
	@echo "intermediate objects cleaned up!"

# Clean up everything.
clean: clean_obj
	rm -f ${OUTDIR}/${OUTPUTNAME}
	@echo "all cleaned up!"
