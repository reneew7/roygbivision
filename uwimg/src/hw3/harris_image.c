#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    image filter;
    float val, pow;
    int filterSize;
    filterSize = (int) ceil(6.0 * sigma);
    if (filterSize % 2 == 0) {
        filterSize++;
    }
    filter = make_image(filterSize, 1, 1);
    for (int x = 0; x < filterSize; x++) {
        pow = (float) ((x - filterSize / 2) * (x - filterSize / 2));
        pow = (float) -1 * pow / (2 * sigma * sigma);
        pow = exp(pow);
        val = pow / (TWOPI * sigma * sigma);
        set_pixel(filter, x, 0, 0, val);
    }
    l1_normalize(filter);
    return filter;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    /* if(1){
        image g = make_gaussian_filter(sigma);
        image s = convolve_image(im, g, 1);
        free_image(g);
        return s;
    } */
    image gx = make_1d_gaussian(sigma);
    image gy = make_image(1, gx.w, 1);
    float val;
    for (int i = 0; i < gy.h; i++) {
        val = get_pixel(gx, i, 0, 0);
        set_pixel(gy, 0, i, 0, val);
    }
    image s = convolve_image(im, gx, 1);
    image res = convolve_image(s, gy, 1);
    free_image(gx);
    free_image(gy);
    free_image(s);
    return res;
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    image S, gx, gy, ix, iy, res;
    float ix2, iy2, ixy;
    S = make_image(im.w, im.h, 3);
    gx = make_gx_filter();
    gy = make_gy_filter();

    ix = convolve_image(im, gx, 0);
    iy = convolve_image(im, gy, 0);

    for (int i = 0; i < im.h; i++) {
        for (int j = 0; j < im.w; j++) {
            ix2 = get_pixel(ix, j, i, 0);
            iy2 = get_pixel(iy, j, i, 0);

            ixy = ix2 * iy2;
            ix2 = ix2 * ix2;
            iy2 = iy2 * iy2;
            set_pixel(S, j, i, 0, ix2);
            set_pixel(S, j, i, 1, iy2);
            set_pixel(S, j, i, 2, ixy);
        }
    }

    res = smooth_image(S, sigma);
    free_image(S);
    free_image(gx);
    free_image(gy);
    free_image(ix);
    free_image(iy);
    return res;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    float trace, det, ix2, iy2, ixy, r;

    for (int i = 0; i < S.h; i++) {
        for (int j = 0; j < S.w; j++) {
            ix2 = get_pixel(S, j, i, 0);
            iy2 = get_pixel(S, j, i, 1);
            ixy = get_pixel(S, j, i, 2);
            trace = ix2 + iy2;
            det = (ix2 * iy2) - (ixy * ixy);
            // We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
            r = (float) det - (0.06 * trace * trace);
            set_pixel(R, j, i, 0, r);
        }
    }
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    // for every pixel in the image:
    //     for neighbors within w:
    //         if neighbor response greater than pixel response:
    //             set response to be very low (I use -999999 [why not 0??])
    image r;
    float val, self;
    int window, set;
    r = copy_image(im);
    window = w / 2;
    for (int i = 0; i < im.c; i++) {
        for (int j = 0; j < im.h; j++) {
            for (int k = 0; k < im.w; k++) {
                set = 0;
                self = get_pixel(im, k, j, i);
                set_pixel(r, k, j, i, self);
                for (int m = j - window; m <= j + window; m++) {
                    for (int n = k - window; n <= k + window; n++) {
                        val = get_pixel(im, n, m, i);
                        if (val > self) {
                            set_pixel(r, k, j, i, -999999);
                            set = 1;
                            break;
                        }
                    }
                    if (set) {
                        break;
                    }
                }
            }
        }
    }
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);

    // count number of responses over threshold
    int count = 0;
    float val;
    for (int i = 0; i < Rnms.h; i++) {
        for (int j = 0; j < Rnms.w; j++) {
            val = get_pixel(Rnms, j, i, 0);
            if (val > thresh) {
                count++;
            }
        }
    }

    
    *n = count; // <- set *n equal to number of corners in image.
    descriptor *d = calloc(count, sizeof(descriptor));
    int idx = 0;
    int idxPix;
    // fill in array *d with descriptors of corners, use describe_index.
    for (int i = 0; i < Rnms.h; i++) {
        for (int j = 0; j < Rnms.w; j++) {
            val = get_pixel(Rnms, j, i, 0);
            if (val > thresh) {
                idxPix = i * im.w + j;
                    d[idx] = describe_index(im, idxPix);
                    idx++;
            }
        }
    }
    free_image(S);
    free_image(R);
    free_image(Rnms);
    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
