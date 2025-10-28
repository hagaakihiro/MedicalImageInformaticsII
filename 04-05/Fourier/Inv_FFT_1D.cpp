//FFT for one dimension
// Witten by Taisei Shimomura (Haga's Laboratory)
// Modified by Akihiro Haga
// Ver.1, Oct. 27, 2019

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


#define  PI  3.14159265358979

void read_data(char *, double *, int, int);
void write_data(char *, double *, int);
void fft1d(int, double *, double *, int);
void FFTInit(int, double *, double *, unsigned short *);
void FFT(int, int, double *, double *, double *, double *, unsigned short *);

typedef struct {
	char   f1[50];      /* input real part file name */
	char   f3[50];      /* output real part file name */
	char   f4[50];      /* output imaginary part file name */
    char   f5[50];      /* output inversed real data file name */
	double  *fr;        /* real part data */
    double  *fi;        /* imaginary part data */
	int    nx;          /* number of data  */
	int    ir;          /* forward or inverse */
} Param;


int main(void)
{
	Param   *pm;
	int   i;

	/* default parameter value */
	pm = (Param *)malloc(sizeof(Param));
	//sprintf( pm->f1, "ocsillation.csv");
   	sprintf( pm->f3, "ocsillation_FFT_real.csv");
	sprintf( pm->f4, "ocsillation_FFT_imaginary.csv");
    sprintf( pm->f5, "ocsillation_reproduced.csv");
    
    // Number of data
	pm->nx = 512;
    int nskip = 0;
    //real part (nx size)
    pm->fr = (double *)malloc((unsigned long)pm->nx*sizeof(double));
    //imaginary (nx size)
    pm->fi = (double *)malloc((unsigned long)pm->nx*sizeof(double));
    
	/// Read Image data  ///
	printf(" *** Read real part csv data   ***\n");
	read_data(pm->f3, pm->fr, pm->nx, nskip);  //
    printf(" *** Read imaginary part csv data   ***\n");
    read_data(pm->f4, pm->fi, pm->nx, nskip);  //
    
    /// 1-D inverse fourier transform ///
    printf(" *** 1D inverse fourier transform   ***\n");
    pm->ir = -1;  //FFT = 1 IFFT = -1
    fft1d(pm->ir, pm->fr, pm->fi, pm->nx);
    
    //for(i = 0 ; i < pm->nx ; i++){
    //    printf("%lf %lf \n", f_orig[i],pm->fi[i]);
    //}
    
    ///Write inversed data  ///
    printf(" *** Write reproduced data in csv format ***\n");
    write_data(pm->f5, pm->fr, pm->nx);
    //write_data(pm->f5, f_orig, pm->nx);
    
	free(pm->fr);
	free(pm->fi);
	free(pm);

	return 0;
}

//Here is technical part

void read_data(char *fi, double *img, int size, int nskip)
{
	FILE   *fp;
    int ret;
    char buf[4][128];
    double data[4];
    
	/* open file and write data */
	if((fp = fopen(fi, "r")) == NULL) {
		fprintf( stderr," Error : file open [%s].\n", fi);
		exit(1);
	}
    
    //for(int i=0; i< nskip; i++) fscanf(fp, "%lf", &data[0]);
    //printf("%s %s %s %s\n",buf[0], buf[1], buf[2], buf[3]);
    //exit(0);
    int k = 0;
    for(int i=0; i<size; i++)
    {
        fscanf(fp, "%lf", &data[0]);
        img[k] = data[0];
        if(i < nskip || i > size-nskip) img[k]=0;
        //printf("%d %lf \n", i, img[k]);
        k++;
    }
    if(k != size)
    {
        printf("size is different ! %d %d \n", k, size);
        exit(1);
    }
    //exit(0);
    fclose(fp);
}

void write_data(char *fi, double *img, int size)
{
  FILE   *fp;

	/* open file and write data */
	if((fp = fopen(fi, "w")) == NULL) {
		fprintf( stderr," Error : file open [%s].\n", fi);
		exit(1);
	}
    for(int k = 0; k < size; k++)
    {
        fprintf(fp,"%lf \n", img[k]);
    }
	fclose(fp);
}

void bitrev(int nx, double *xr, double *xi, unsigned short *brv)
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


void FFT(int ir, int nx, double *xr, double *xi, double *si, double *co, unsigned short *brv)
{
	int     i, j, n1, n2=nx, j3, j4, k, l, ll, d=1, g;
	double   a, b, c, s;

	for(l = 1; l <= nx/2; l *= 2, d += d) { // l*2 ,d=1,2,4,8
	    g  = 0;
	    ll = n2;
	    n2 /= 2;
	    for(k = 1; k <= n2; k++)
        {
	        n1 = k-ll; //1-512 =-511, 2-512=-510 ,3-512=-509 
	        c  = co[g];//co[0], co[1]
	        s  = -ir*si[g];//-si[0], -si[1]
	        g += d;  //0,0+1=1, 
	        for(j = ll; j <= nx; j += ll)
            {
                j3 = j+n1-1; //n2 + (1-n2) -1=0, n2 -510 -1 =1
	            j4 = j3+n2;  //0 + 256 = 256, 1+256 = 257
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

void FFTInit(int nx, double *si, double *co, unsigned short *brv)
{
	double  d=2.0*PI/nx;
	int     i;
	int     br(int, unsigned);
	for(i = 0; i < nx/4; i++) {
	    si[i] = (double)sin(d*i);
	    co[i+nx/4] = -si[i];
	}
	for(i = nx/4; i < nx/2; i++) {
	    si[i] = (double)sin(d*i);
	    co[i-nx/4] = si[i];
	}
	for(i = 0; i < nx; i++)
	  brv[i] = br(nx, (unsigned)i);
}

void fft1d(int ir, double *fr, double *fi, int nx)
{
	int   i, j;
	double *gr, *gi, *si, *co;
	unsigned short *br;

	gr = (double *)malloc((unsigned long)nx*sizeof(double));
	gi = (double *)malloc((unsigned long)nx*sizeof(double));
	si = (double *)malloc((unsigned long)nx/2*sizeof(double));
	co = (double *)malloc((unsigned long)nx/2*sizeof(double));
	br = (unsigned short *)malloc((unsigned long)nx*sizeof(unsigned short));
	FFTInit(nx, si, co, br);
    for(j = 0 ; j < nx/2 ; j++)
    {
        gr[j] = fr[j+nx/2];
        gr[j+nx/2] = fr[j];
        gi[j] = fi[j+nx/2];
        gi[j+nx/2] = fi[j];
    }
    FFT(ir, nx, gr, gi, si, co, br);
    for(j = 0 ; j < nx/2 ; j++)
    {
        fr[j+nx/2] = gr[j];
        fr[j] = gr[j+nx/2];
        fi[j+nx/2] = gi[j];
        fi[j] = gi[j+nx/2];
    }
	free(gr);
	free(gi);
	free(si);
	free(co);
	free(br);

}
