#include <math.h>
#include "image.h"

// round float to int, return 0 if negative or certain max if over that max
static int roundNearest(float, int);

float nn_interpolate(image im, float x, float y, int c)
{
    int nearestX, nearestY;
    nearestX = roundNearest(x, im.w - 1);
    nearestY = roundNearest(y, im.h - 1);
    return get_pixel(im, nearestX, nearestY, c);
}

static int roundNearest(float coord, int max) {
    if (coord < 0 ) {
        return 0;
    } else if (coord > max) {
        return max;
    } else {
        return (int) (coord + 0.5);
    }
}

image nn_resize(image im, int w, int h)
{
    float scaleX, scaleY, offsetX, offsetY, val;
    scaleX = (float) im.w / w;
    scaleY = (float) im.h / h;
    // match -0.5 to -0.5
    offsetX = (float) (scaleX - 1) / 2;
    offsetY = (float) (scaleY - 1) / 2;
    image resized = make_image(w, h, im.c);
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                val = nn_interpolate(im, k * scaleX + offsetX, j * scaleY + offsetY, i);
                set_pixel(resized, k, j, i, val);
            }
        }
    }
    return resized;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    float q1, q2, d1, d2, d3, d4, q;
    int minX, maxX, minY, maxY;
    minX = floor(x);
    minY = floor(y);
    maxX = ceil(x);
    maxY = ceil(y);
    d1 = (float) (x - minX);
    d2 = (float) (maxX - x);
    d3 = (float) (y - minY);
    d4 = (float) (maxY - y);
    q1 = get_pixel(im, minX, minY, c) * d2 + get_pixel(im, maxX, minY, c) * d1;
    q2 = get_pixel(im, minX, maxY, c) * d2 + get_pixel(im, maxX, maxY, c) * d1;
    q = q1 * d4 + q2 * d3;
    return q;
}

image bilinear_resize(image im, int w, int h)
{
    float scaleX, scaleY, offsetX, offsetY, val;
    scaleX = (float) im.w / w;
    scaleY = (float) im.h / h;
    // match -0.5 to -0.5
    offsetX = (float) (scaleX - 1) / 2;
    offsetY = (float) (scaleY - 1) / 2;
    image resized = make_image(w, h, im.c);
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                val = bilinear_interpolate(im, k * scaleX + offsetX, j * scaleY + offsetY, i);
                set_pixel(resized, k, j, i, val);
            }
        }
    }
    return resized;
}

