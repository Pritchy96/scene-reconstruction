CC := g++
CXXFLAGS := -Wall -std=c++11 -g

SRC_DIR := ./src
TEST_DIR := ./test
BUILD_DIR := ./build
TARGET_DIR := ./bin
LIBRARY_DIR := ./lib

LIBS := -lGL -lGLU -lglfw -lX11 -lXxf86vm -lXrandr -lpthread -lXi -lGLEW -lopencv_stitching\
 -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired\
  -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo\
   -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash\
    -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency\
	 -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching\
	  -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot\
	   -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_calib3d\
	    -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect\
		 -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core -lhdf5\
		  -lboost_system -lboost_filesystem -lgl_framework -lcvsba -lvtkCommonCore -lpcl_common -lpcl_kdtree\
		   -lpcl_octree -lpcl_search -lpcl_recognition -lvtkCommonDataModel -lvtkCommonMath -lvtkRenderingCore\
		   -lqhull -lpcl_surface -lpcl_sample_consensus -lpcl_io -lpcl_filters -lpcl_features -lpcl_keypoints\
		    -lpcl_registration -lpcl_segmentation  -lpcl_visualization -lpcl_people -lpcl_outofcore\
			 -lpcl_tracking -lvtkRenderingLOD


MAINS := $(BUILD_DIR)/main.o $(TEST_DIR)/gl_framework_test.o
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(filter-out $(MAINS), $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES)))

all: $(MAINS) #This currently makes './build/blah.o', not 'blah'

# Requires a bit more thought to do nicely.
# gl_framework_test: $(OBJ_FILES) $(BUILD_DIR)/gl_framework_test.o
# 	$(CC) $(CXXFLAGS) -o $(TARGET_DIR)/$@ $(LIBS) $^
# 	cp -r $(SRC_DIR)/shaders $(TARGET_DIR)


main: $(OBJ_FILES) $(BUILD_DIR)/main.o
	$(CC) $(CXXFLAGS) -o $(TARGET_DIR)/$@ $(LIBS) $^
	cp -r $(SRC_DIR)/shaders $(TARGET_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm $(BUILD_DIR)/*.o

-include $(OBJ_FILES:.o=.d)