#include <iostream>
#include "png.h"

__host__ int main(void)
{
	std::cout << "Hello, world!\n" << std::endl;

	std::cout << "Reading file..." << std::endl;
	read_png_file("test.png");

	std::cout << "Writing file..." << std::endl;
	write_png_file("out.png");

	std::cout << "Done." << std::endl;
	return 0;
}
