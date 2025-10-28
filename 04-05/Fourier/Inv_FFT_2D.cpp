
//IFFT polynominal representation of human body
// Witten by Taisei Shimomura (c201504026@tokushima-u.ac.jp)
// Haga's Laboratory
// Ver.1, July. 8, 2018



#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define  PI  3.14159265358979

void read_data(char *, float *, int);
void write_data(char *, float *, int);
void fft2d(int, float *, float *, int, int);
void FFTInit(int, float *, float *, unsigned short *);
void FFT(int, int, float *, float *, float *, float *, unsigned short *);

typedef struct {
	char   f1[50];      /* input real part file name */
  	char   f2[50];      /* input imaginary part file name */
	char   f3[50];      /* output real part file name */
  //char   f4[50];      /* output imaginary part file name */
	float  *fr;         /* real part data */
        float  *fi;     /* imaginary part data */
	int    nx ;         /* number of width  */
	int    ny ;         /* number of height */
	int    ir;          /* forward or inverse */
} Param;


int main(void)
{
	Param   *pm;
	int   i;

	/* default parameter value */
	pm = (Param *)malloc(sizeof(Param));
	sprintf( pm->f1, "FFT_2D_real.raw");
	sprintf( pm->f2, "FFT_2D_imaginary.raw");
	sprintf( pm->f3, "CT_image_reproduced_lowpass.raw");
	pm->nx = 512;
	pm->ny = 512;
	pm->ir = -1;  //FFT = 1 IFFT = -1
	pm->fr = (float *)malloc((unsigned long)pm->nx*pm->ny*sizeof(float));
	pm->fi = (float *)malloc((unsigned long)pm->nx*pm->ny*sizeof(float));

	/// Read Image data  ///
	printf(" *** Read Image data   ***\n");
	read_data(pm->f1, pm->fr,pm->nx*pm->ny);
    read_data(pm->f2, pm->fi,pm->nx*pm->ny);

    int nskip = 0;
    int k = 0;
    double RR = 250; // Lowpass
    for(int iy=0; iy<pm->ny; iy++)
        for(int ix=0; ix<pm->ny; ix++)
        {
            if(abs(iy - pm->ny/2) > pm->ny/2-nskip || abs(ix - pm->nx/2) > pm->nx/2-nskip)
            {
                pm->fr[k] = 0;
                pm->fi[k] = 0;
            }
            double rr2 = (iy - pm->ny/2)*(iy - pm->ny/2)+(ix - pm->nx/2)*(ix - pm->nx/2);
            if(rr2 > RR*RR)
            {
                pm->fr[k] = 0;
                pm->fi[k] = 0;
            }
            k++;
        }
    
    
	/// 2-D fourier transform ///
	printf(" *** 2-D fourier transform   ***\n");
	fft2d(pm->ir, pm->fr, pm->fi, pm->nx, pm->ny);

	///Write Image data  ///
	printf(" *** Write Image data   ***\n");
	write_data(pm->f3, pm->fr, pm->nx*pm->ny);

	free(pm->fr);
	free(pm->fi);
	free(pm);

	return 0;
}

//Here is technical part

void read_data(char *fi, float *img, int size)
{
	FILE   *fp;

	/* open file and write data */
	if((fp = fopen(fi, "rb")) == NULL) {
		fprintf( stderr," Error : file open [%s].\n", fi);
		exit(1);
	}
	fread(img, sizeof(float), size, fp);
	fclose(fp);
}

void write_data(char *fi, float *img, int size)
{
  FILE   *fp;

	/* open file and write data */
	if((fp = fopen(fi, "wb")) == NULL) {
		fprintf( stderr," Error : file open [%s].\n", fi);
		exit(1);
	}
	fwrite(img, sizeof(float), size, fp);
	fclose(fp);
}

void bitrev(int nx, float *xr, float *xi, unsigned short *brv)
{
	int     i, j;
	float   a, b;
	for(i = 0; i < nx; i++){
	    j = brv[i];
	    if(i < j){
	      a = xr[i]; 
	      b = xi[i];
	        xr[i] = xr[j];
	        xi[i] = xi[j];
	        xr[j] = a;
	        xi[j] = b;
	    }
	}
}


void FFT(int ir, int nx, float *xr, float *xi, float *si, float *co, unsigned short *brv)
{
	int     i, j, n1, n2=nx, j3, j4, k, l, ll, d=1, g;
	float   a, b, c, s;

	for(l = 1; l <= nx/2; l *= 2, d += d) {
	    g  = 0;
	    ll = n2;
	    n2 /= 2;
	    for(k = 1; k <= n2; k++)
        {
	        n1 = k-ll;
	        c  = co[g];
	        s  = -ir*si[g];
	        g += d;
	        for(j = ll; j <= nx; j += ll)
            {
                j3 = j+n1-1;
	            j4 = j3+n2;
	            a  = xr[j3]-xr[j4]; 
	            b  = xi[j3]-xi[j4];
	            xr[j3] += xr[j4];
                xi[j3] += xi[j4];
	            xr[j4] =  c*a+s*b;
                xi[j4] =  c*b-s*a;
	        }
	    }
	}

	bitrev(nx, xr, xi, brv);
	if(ir == -1)
	    for(i = 0; i < nx; i++) {
	        xr[i] /= nx;
	        xi[i] /= nx;
	    }
}

int br(int nx, unsigned nn)
{
	unsigned        r,c;
	r=0;
	for(c = 1; c <= (unsigned)nx/2; c <<= 1) {
	  r <<= 1;
	    if((nn&c) != 0)
	         r++;
	}
	return(r);
}

void FFTInit(int nx, float *si, float *co, unsigned short *brv)
{
	double  d=2.0*PI/nx;
	int     i;
	int     br(int, unsigned);
	for(i = 0; i < nx/4; i++) {
	    si[i] = (float)sin(d*i);
	    co[i+nx/4] = -si[i];
	}
	for(i = nx/4; i < nx/2; i++) {
	    si[i] = (float)sin(d*i);
	    co[i-nx/4] = si[i];
	}
	for(i = 0; i < nx; i++)
	  brv[i] = br(nx, (unsigned)i);
}

void fft2d(int ir, float *fr, float *fi, int nx, int ny)
{
	int   i, j;
	float *gr, *gi, *si, *co;
	unsigned short *br;

	gr = (float *)malloc((unsigned long)nx*sizeof(float));
	gi = (float *)malloc((unsigned long)nx*sizeof(float));
	si = (float *)malloc((unsigned long)nx/2*sizeof(float));
	co = (float *)malloc((unsigned long)nx/2*sizeof(float));
	br = (unsigned short *)malloc((unsigned long)nx*sizeof(unsigned short));
	FFTInit(nx, si, co, br);
	for(i = 0 ; i < ny ; i++) {
		for(j = 0 ; j < nx/2 ; j++)
        {
            gr[j] = fr[i*nx+j+nx/2];
			gr[j+nx/2] = fr[i*nx+j];
			gi[j] = fi[i*nx+j+nx/2];
			gi[j+nx/2] = fi[i*nx+j];
		}
		FFT(ir, nx, gr, gi, si, co, br);
		for(j = 0 ; j < nx/2 ; j++)
        {
            fr[i*nx+j+nx/2] = gr[j];
			fr[i*nx+j] = gr[j+nx/2];
			fi[i*nx+j+nx/2] = gi[j];
			fi[i*nx+j] = gi[j+nx/2];
		}
	}
	free(gr);
	free(gi);
	free(si);
	free(co);
	free(br);

	gr = (float *)malloc((unsigned long)ny*sizeof(float));
	gi = (float *)malloc((unsigned long)ny*sizeof(float));
	si = (float *)malloc((unsigned long)ny/2*sizeof(float));
	co = (float *)malloc((unsigned long)ny/2*sizeof(float));
	br = (unsigned short *)malloc((unsigned long)ny*sizeof(unsigned short));
	FFTInit(ny, si, co, br);
	for(j = 0 ; j < nx ; j++) {
		for(i = 0 ; i < ny/2 ; i++) {
			gr[i] = fr[(i+ny/2)*nx+j];
			gr[i+ny/2] = fr[i*nx+j];
			gi[i] = fi[(i+ny/2)*nx+j];
			gi[i+ny/2] = fi[i*nx+j];
		}
		FFT(ir, ny, gr, gi, si, co, br);
		for(i = 0 ; i < ny/2 ; i++) {
			fr[(i+ny/2)*nx+j] = gr[i];
			fr[i*nx+j] = gr[i+ny/2];
			fi[(i+ny/2)*nx+j] = gi[i];
			fi[i*nx+j] = gi[i+ny/2];
		}
	}
	free(gr);
	free(gi);
	free(si);
	free(co);
	free(br);
}
