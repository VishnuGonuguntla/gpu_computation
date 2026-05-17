CXX = g++
CXXFLAGS = -g -O3
SRC = $(wildcard src/*.cpp)

bin/molDyn: $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f bin/molDyn