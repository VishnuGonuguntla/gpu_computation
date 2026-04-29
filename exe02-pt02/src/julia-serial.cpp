#include <iostream>
#include "lodepng.h"
#include <complex>
#include <vector>


int main(){
    // define image 
    int image_width = 800;
    int image_height = 800;

    unsigned char r, g, b;
    std::complex<double> c(-0.525, -0.525);
    std::complex<double> z;
    std::vector<unsigned char> image (image_height * image_width * 4); 

    //int temp = norm(c);
    int threshold = 10;

    for (int i = 0; i < image_width; ++i){
        for (int j = 0; j < image_height; ++j){
            double real = -2 + double((i * 4.0) / image_width); 
            double imaginary = -2 + double((j * 4.0) / image_height);
            
            z = {real, imaginary};
            size_t iter = 0; 
            size_t max_iter = 200;

            while (std::abs(z) < threshold*threshold && iter < max_iter){
                z = z * z + c;
                iter += 1;
            }

            if (iter == max_iter){
                r=static_cast<unsigned char> ((iter * 255/max_iter)), g=0, b=0;
            }
            else{
                r=0, g=0, b=static_cast<unsigned char> ((iter * 255/max_iter)); 
            }

            int index = 4 * (j * image_width + i);

            image[index] = r;
            image[index + 1] = g;
            image[index + 2] = b;
            image[index + 3] = 255;

        }
    }

    unsigned error = lodepng::encode("julia_set.png", image, image_width, image_height);

    if(error){
        std::cout << "encoder error" <<error<<": "<< lodepng_error_text(error) << std::endl;
    }
    std::cout << "image saved" << std::endl;


}




