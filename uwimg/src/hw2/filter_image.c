#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
    float norm, sum;
    sum = 0.0;
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                sum += get_pixel(im, k, j, i);
            }
        }
    }
    
    if (sum != 0.0) {
        for (int i = 0; i < im.c; i++) {
            for (int j = 0; j < im.h; j++) {
                for (int k = 0; k < im.w; k++) {
                    norm = get_pixel(im, k, j, i);
                    set_pixel(im, k, j, i, (float) norm / sum);
                }
            }
        }
    }
}

image make_box_filter(int w)
{
    image im;
    float val;
    im = make_image(w, w, 1);
    val = (float) 1 / (w * w);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            set_pixel(im, j, i, 0, val);
        }
    }
    return im;
}

image convolve_image(image im, image filter, int preserve)
{
    assert(filter.c == 1 || filter.c == im.c);
    image convolved;
    float conv, sum;
    int filterX, filterY;
    filterX = filter.w / 2;
    filterY = filter.h / 2;
    if (preserve == 1) {
        convolved = make_image(im.w, im.h, im.c);
        if (filter.c != 1) {
            for (int i = 0; i < im.c; i++) {
                for (int j = 0; j < im.h; j++) {
                    for (int k = 0; k < im.w; k++) {
                        sum = 0;
                        for (int x = -filterX; x <= filterX; x++) {
                            for (int y = -filterY; y <= filterY; y++) {
                                conv = get_pixel(filter, x + filterX, y + filterY, i);
                                sum += conv * get_pixel(im, k + x, j + y, i);
                            }
                        }
                        set_pixel(convolved, k, j, i, sum);
                    }
                }
            }
        } else {
            for (int i = 0; i < im.c; i++) {
                for (int j = 0; j < im.h; j++) {
                    for (int k = 0; k < im.w; k++) {
                        sum = 0;
                        for (int x = -filterX; x <= filterX; x++) {
                            for (int y = -filterY; y <= filterY; y++) {
                                conv = get_pixel(filter, x + filterX, y + filterY, 0);
                                sum += conv * get_pixel(im, k + x, j + y, i);
                            }
                        }
                        set_pixel(convolved, k, j, i, sum);
                    }
                }
            }
        }
        
    } else {
        convolved = make_image(im.w, im.h, 1);
        if (filter.c != 1) {
            for (int j = 0; j < im.h; j++) {
                for (int k = 0; k < im.w; k++) {
                    sum = 0;
                    for (int i = 0; i < im.c; i++) {
                        for (int x = -filterX; x <= filterX; x++) {
                            for (int y = -filterY; y <= filterY; y++) {
                                conv = get_pixel(filter, x + filterX, y + filterY, i);
                                sum += conv * get_pixel(im, k + x, j + y, i);
                            }
                        }
                    }
                    set_pixel(convolved, k, j, 0, sum);
                }
            }
        } else {
            for (int j = 0; j < im.h; j++) {
                for (int k = 0; k < im.w; k++) {
                    sum = 0;
                    for (int i = 0; i < im.c; i++) {
                        for (int x = -filterX; x <= filterX; x++) {
                            for (int y = -filterY; y <= filterY; y++) {
                                conv = get_pixel(filter, x + filterX, y + filterY, 0);
                                sum += conv * get_pixel(im, k + x, j + y, i);
                            }
                        }
                    }
                    set_pixel(convolved, k, j, 0, sum);
                }
            }
        }
    }
    return convolved;
}

image make_highpass_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 1, 0, 0, -1);
    set_pixel(filter, 2, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 1, 1, 0, 4);
    set_pixel(filter, 2, 1, 0, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 1, 2, 0, -1);
    set_pixel(filter, 2, 2, 0, 0);
    return filter;
}

image make_sharpen_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 1, 0, 0, -1);
    set_pixel(filter, 2, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 1, 1, 0, 5);
    set_pixel(filter, 2, 1, 0, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 1, 2, 0, -1);
    set_pixel(filter, 2, 2, 0, 0);
    return filter;
}

image make_emboss_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -2);
    set_pixel(filter, 1, 0, 0, -1);
    set_pixel(filter, 2, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 1, 1, 0, 1);
    set_pixel(filter, 2, 1, 0, 1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 1, 2, 0, 1);
    set_pixel(filter, 2, 2, 0, 2);
    return filter;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Since the sharpen and emboass filters are both in color, we would want to preserve the channels when we run the convolution
// so that the color channels stay intact.
// Since the highpass filter is just for edge detection, preserving isn't necessary because that can be done in black and white.

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// The highpass filter might need some post-processing if we are not preserving the channels. Since we combine all our channels
// into one channel, we might need to do some more processing for the black and white verison of the image.

image make_gaussian_filter(float sigma)
{
    image filter;
    float val, pow;
    int filterSize;
    filterSize = (int) ceil(6.0 * sigma);
    if (filterSize % 2 == 0) {
        filterSize++;
    }
    filter = make_image(filterSize, filterSize, 1);
    for (int x = 0; x < filterSize; x++) {
        for (int y = 0; y < filterSize; y++) {
            pow = (float) ((x - filterSize / 2) * (x - filterSize / 2)) + ((y - filterSize / 2) * (y - filterSize / 2));
            pow = (float) -1 * pow / (2 * sigma * sigma);
            pow = exp(pow);
            val = pow / (TWOPI * sigma * sigma);
            set_pixel(filter, x, y, 0, val);
        }
    }
    l1_normalize(filter);
    return filter;
}

image add_image(image a, image b)
{
    assert(a.h == b.h && a.w == b.w && a.c == b.c);
    image added;
    float sum;
    added = make_image(a.w, a.h, a.c);
    for (int i = 0; i < a.c; i++) {
        for (int j = 0; j < a.h; j++) {
            for (int k = 0; k < a.w; k++) {
                sum = get_pixel(a, k, j, i);
                sum += get_pixel(b, k, j, i);
                set_pixel(added, k, j, i, sum);
            }
        }
    }
    return added;
}

image sub_image(image a, image b)
{
    assert(a.h == b.h && a.w == b.w && a.c == b.c);
    image sub;
    float diff;
    sub = make_image(a.w, a.h, a.c);
    for (int i = 0; i < a.c; i++) {
        for (int j = 0; j < a.h; j++) {
            for (int k = 0; k < a.w; k++) {
                diff = get_pixel(a, k, j, i);
                diff -= get_pixel(b, k, j, i);
                set_pixel(sub, k, j, i, diff);
            }
        }
    }
    return sub;
}

image make_gx_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -1);
    set_pixel(filter, 1, 0, 0, 0);
    set_pixel(filter, 2, 0, 0, 1);
    set_pixel(filter, 0, 1, 0, -2);
    set_pixel(filter, 1, 1, 0, 0);
    set_pixel(filter, 2, 1, 0, 2);
    set_pixel(filter, 0, 2, 0, -1);
    set_pixel(filter, 1, 2, 0, 0);
    set_pixel(filter, 2, 2, 0, 1);
    return filter;
}

image make_gy_filter()
{
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -1);
    set_pixel(filter, 1, 0, 0, -2);
    set_pixel(filter, 2, 0, 0, -1);
    set_pixel(filter, 0, 1, 0, 0);
    set_pixel(filter, 1, 1, 0, 0);
    set_pixel(filter, 2, 1, 0, 0);
    set_pixel(filter, 0, 2, 0, 1);
    set_pixel(filter, 1, 2, 0, 2);
    set_pixel(filter, 2, 2, 0, 1);
    return filter;
}

void feature_normalize(image im)
{
    // get min max
    float min, max, val, range;
    min = get_pixel(im, 0, 0, 0);
    max = min;
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                val = get_pixel(im, k, j, i);
                if (val < min) {
                    min = val;
                } else if (val > max) {
                    max = val;
                }
            }
        }
    }
    range = max - min;
    if (range == 0) {
        for (int i = 0; i < im.c; i++) {
            for (int j = 0; j < im.h; j++) {
                for (int k = 0; k < im.w; k++) {
                    set_pixel(im, k, j, i, 0);
                }
            }
        }
    } else {
        for (int i = 0; i < im.c; i++) {
            for (int j = 0; j < im.h; j++) {
                for (int k = 0; k < im.w; k++) {
                    val = get_pixel(im, k, j, i);
                    set_pixel(im, k, j, i, (float) (val / range));
                }
            }
        }
    }
}

image *sobel_image(image im)
{
    image* img;
    image gx, gy, imx, imy;
    float xval, yval, grad, theta;

    img = calloc(2, sizeof(image));
    img[0] = make_image(im.w, im.h, 1);
    img[1] = make_image(im.w, im.h, 1);

    gx = make_gx_filter();
    gy = make_gy_filter();
    imx = convolve_image(im, gx, 0);
    imy = convolve_image(im, gy, 0);

    for (int i = 0; i < im.h; i++) {
        for (int j = 0; j < im.w; j++) {
            xval = get_pixel(imx, j, i, 0);
            yval = get_pixel(imy, j, i, 0);
            // sobel: gradient = sqrt(x^2 + y^2)
            // theta: arctan(y / x)
            grad = sqrt((xval * xval) + (yval * yval));
            theta = atan2(yval, xval);
            set_pixel(img[0], j, i, 0, grad);
            set_pixel(img[1], j, i, 0, theta);
        }
    }

    free_image(gx);
    free_image(gy);
    free_image(imx);
    free_image(imy);
    return img;
}

image colorize_sobel(image im)
{
    image color;
    float mag, ang;
    image* sobel = sobel_image(im);
    feature_normalize(sobel[0]);
    feature_normalize(sobel[1]);
    color = make_image(im.w, im.h, 3);
    for (int i = 0; i < im.h; i++) {
        for (int j = 0; j < im.w; j++) {
            mag = get_pixel(sobel[0], j, i, 0);
            ang = get_pixel(sobel[1], j, i, 0);
            // mag: sat and value
            set_pixel(color, j, i, 1, mag);
            set_pixel(color, j, i, 2, mag);
            // ang: hue
            set_pixel(color, j, i, 0, ang);
        }
    }
    hsv_to_rgb(color);
    return color;
}
