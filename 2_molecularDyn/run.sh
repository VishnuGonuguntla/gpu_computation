g++ -std=c++2a -Wall -Wextra -O3 -Isrc -c src/Helper.cpp -o obj/Helper.o
g++ -std=c++2a -Wall -Wextra -O3 -Isrc -c src/Solver.cpp -o obj/Solver.o
# g++ -std=c++20 -Wall -Wextra -O3 -I/home/vishnu-gonuguntla/packages/eigen-3.4.0 -Isrc -c src/DataReader.cpp -o obj/dataReader.o
# g++ -std=c++20 -Wall -Wextra -O3 -I/home/vishnu-gonuguntla/packages/eigen-3.4.0 -Isrc -c src/NeuralNet.cpp -o obj/neuralNet.o

g++ -std=c++2a -Wall -Wextra -O3 obj/Helper.o obj/Solver.o -Isrc src/molDyn-base.cpp -o bin/molDyn

# g++ -std=c++20 -Wall -Wextra -O3 -I/home/vishnu-gonuguntla/packages/eigen-3.4.0 obj/dataReader.o obj/helper.o -Isrc src/read_dataset_images.cpp -o bin/image_parser
# g++ -std=c++20 -Wall -Wextra -O3 -I/home/vishnu-gonuguntla/packages/eigen-3.4.0 obj/dataReader.o obj/helper.o -Isrc src/read_dataset_labels.cpp -o bin/label_parser

# ./bin/image_parser data/train-images.idx3-ubyte results/train_images.txt 100
# ./bin/label_parser data/train-labels.idx1-ubyte results/train_labels.txt 100
./bin/molDyn molDyn.par