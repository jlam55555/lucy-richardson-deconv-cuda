#ifndef PNGIOH
#define PNGIOH

#include <png.h>

extern int width, height;
extern png_byte color_type;
extern png_byte bit_depth;

extern png_structp png_ptr;
extern png_infop info_ptr;
extern int number_of_passes;
extern png_bytep * row_pointers;

void read_png_file(const char* file_name);
void write_png_file(const char* file_name);

#endif // PNGIOH
