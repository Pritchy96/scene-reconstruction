CC := g++
CXXFLAGS := -Wall -std=c++11 -MMD -MP

SRC_DIR := ./src
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
		  -lboost_system -lboost_filesystem -L$(LIBRARY_DIR) -lgl_framework


MAINS := $(BUILD_DIR)/main.o
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(filter-out $(MAINS), $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES)))

all: $(MAINS) #This currently makes './build/blah.o', not 'blah'

main: $(OBJ_FILES) $(BUILD_DIR)/main.o
	$(CC) $(CXXFLAGS) -o $(TARGET_DIR)/$@ $(LIBS) $^
	cp -r $(SRC_DIR)/shaders $(TARGET_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm $(BUILD_DIR)/*.o

-include $(OBJ_FILES:.o=.d)