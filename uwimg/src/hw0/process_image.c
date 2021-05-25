#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

// get index of specific pixel, assuming bounds checking was done
static int get_idx(int, int, int, int, int);

float get_pixel(image im, int x, int y, int c)
{
    int idx;
    assert(c >= 0);
    assert(c < im.c);
    x = (x < 0) ? 0 : (x >= im.w ? im.w - 1 : x);
    y = (y < 0) ? 0 : (y >= im.h ? im.h - 1 : y);
    idx = get_idx(im.w, im.h, x, y, c);
    return im.data[idx];
}

static int get_idx(int w, int h, int x, int y, int c)
{
    return c * w * h + x + y * w;
}

void set_pixel(image im, int x, int y, int c, float v)
{
    if (x >= 0 && x < im.w && y >= 0 && y < im.h && c >= 0 && c < im.c) {
        im.data[get_idx(im.w, im.h, x, y, c)] = v;
    }
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, sizeof(float) * im.w * im.h * im.c);
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    int idx;
    float r, g, b;
    image gray = make_image(im.w, im.h, 1);
    for (int i = 0; i < im.w; i++) {
        for (int j = 0; j < im.h; j++) {
          idx = get_idx(im.w, im.h, i, j, 0);
          r = im.data[idx];
          g = im.data[get_idx(im.w, im.h, i, j, 1)];
          b = im.data[get_idx(im.w, im.h, i, j, 2)];
          gray.data[idx] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    int start;
    int end;
    start = get_idx(im.w, im.h, 0, 0, c);
    end = start + (im.w * im.h);
    for (int i = start; i < end; i++) {
        im.data[i] += v;
    }
}

void clamp_image(image im)
{
    for (int i = 0; i < im.w * im.h * im.c; i++) {
        im.data[i] = MAX(im.data[i], 0);
        im.data[i] = MIN(im.data[i], 1);
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    int idx0, idx1, idx2;
    float r, g, b, v, s, h, h1, c;
    for (int i = 0; i < im.w; i++) {
        for (int j = 0; j < im.h; j++) {
            idx0 = get_idx(im.w, im.h, i, j, 0);
            idx1 = get_idx(im.w, im.h, i, j, 1);
            idx2 = get_idx(im.w, im.h, i, j, 2);
            r = im.data[idx0];
            g = im.data[idx1];
            b = im.data[idx2];

            v = three_way_max(r, g, b);
            c = v - three_way_min(r, g, b);

            s = (v == 0) ? 0 : c / v;
            h1 = (c == 0) ? 0 : (v == r) ? ((g - b) / c) : (v == g) ?
                 (((b - r) / c) + 2) : (((r - g) / c) + 4);
            h = (h1 < 0) ? ((h1 / 6) + 1) : h1 / 6;
            im.data[idx0] = h;
            im.data[idx1] = s;
            im.data[idx2] = v;
        }
    }
}

void hsv_to_rgb(image im)
{
    float c, h, s, v, min, r, g, b, x, h1;
    int idx0, idx1, idx2;
    for (int i = 0; i < im.w; i++) {
        for (int j = 0; j < im.h; j++) {
            idx0 = get_idx(im.w, im.h, i, j, 0);
            idx1 = get_idx(im.w, im.h, i, j, 1);
            idx2 = get_idx(im.w, im.h, i, j, 2);
            h = im.data[idx0];
            s = im.data[idx1];
            v = im.data[idx2];
            c = s * v;
            min = v - c;

            // algorithm from wikipedia page linked in hw spec
            h *= 6;
            h1 = h;
            while (h1 >= 2) {
                h1 -= 2;
            }
            h1 = h1 - 1;

            x = (h1 < 0) ? c * (1 + h1) : c * (1 - h1);
            if (h >= 0 && h <= 1) {
                r = c;
                g = x;
                b = 0;
            } else if (h <= 2) {
                r = x;
                g = c;
                b = 0;
            } else if (h <= 3) {
                r = 0;
                g = c;
                b = x;
            } else if (h <= 4) {
                r = 0;
                g = x;
                b = c;
            } else if (h <= 5) {
                r = x;
                g = 0;
                b = c;
            } else if (h <= 6) {
                r = c;
                g = 0;
                b = x;
            }
            r += min;
            g += min;
            b += min;
            
            im.data[idx0] = r;
            im.data[idx1] = g;
            im.data[idx2] = b;
        }
    }
}

void scale_image(image im, int c, float v) {
    int start;
    int end;
    if (c >= 0 && c < im.c) {
        start = get_idx(im.w, im.h, 0, 0, c);
        end = get_idx(im.w, im.h, im.w - 1, im.h - 1, c);
        for (int i = start; i <= end; i++) {
            im.data[i] *= v;
        }
    }
}
