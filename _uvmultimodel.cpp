/*

#
# UVMULTIFIT - C++ MULTI-THREADED CORE ENGINE.
#
# Copyright (c) Ivan Marti-Vidal 2012. 
#               EU ALMA Regional Center. Nordic node.
# Copyright (c) Ivan Marti-Vidal 2024
#               Universitat de Valencia (Spain)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>,
# or write to the Free Software Foundation, Inc., 
# 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# a. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# b. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
# c. Neither the name of the author nor the names of contributors may 
#    be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
#

*/




#include <Python.h>


// compiler warning that we use a deprecated NumPy API
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define NO_IMPORT_ARRAY
#if PY_MAJOR_VERSION >= 3
#define NPY_NO_DEPRECATED_API 0x0
#endif
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

#include <pthread.h>
#include <gsl/gsl_sf_bessel.h>
#include <stdio.h>  
#include <sys/types.h>
#include <new>
#include <complex>
#include <iostream>
#include <fstream>
//#include <sstream> 
//#include <stdlib.h>  
//#include <string.h>


#if QUINN_FITTER == 0
#include "_QuinnFringe.h"
#endif







// cribbed from SWIG machinery
#if PY_MAJOR_VERSION >= 3
#define PyClass_Check(obj) PyObject_IsInstance(obj, (PyObject *)&PyType_Type)
#define PyInt_Check(x) PyLong_Check(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_FromSize_t(x) PyLong_FromSize_t(x)
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
//#define PyString_AsString(str) PyBytes_AsString(str)
#define PyString_Size(str) PyBytes_Size(str)
#define PyString_InternFromString(key) PyUnicode_InternFromString(key)
#define Py_TPFLAGS_HAVE_CLASS Py_TPFLAGS_BASETYPE
#define PyString_AS_STRING(x) PyUnicode_AS_STRING(x)
#define _PyLong_FromSsize_t(x) PyLong_FromSsize_t(x)
// For PyArray_FromDimsAndData -> PyArray_SimpleNewFromData
//#define INTEGER long
//#define INTEGERCAST  (const npy_intp*)
//#else
// For PyArray_FromDimsAndData -> PyArray_SimpleNewFromData
//#define INTEGER int
//#define INTEGERCAST (long int *)
#endif
// and after some hacking
#if PY_MAJOR_VERSION >= 3
#define PyString_AsString(obj) PyUnicode_AsUTF8(obj)
#endif







/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for least-square visibility fitting.";
static char uvmultimodel_docstring[] =
    "Calculate the residuals and chi square of a multi-component model";
static char clearPointers_docstring[] =
    "Delete the data pointers.";
static char setData_docstring[] =
    "Get the data pointers.";
static char setNspw_docstring[] =
    "Set up the pointers to the data arrays.";
static char setModel_docstring[] =
    "Set up the model components and variable parameters.";
static char setNCPU_docstring[] =
    "Set up the parallelization.";
static char setWork_docstring[] =
    "Allocate memory to compute Hessian and error vector.";
static char unsetWork_docstring[] =
    "Deallocate memory obtained with setWork().";
static char QuinnFF_docstring[] =
    "Perform Fringe Fitting, based on the delay-rate fringe peaks, using the Quinn estimator for the peak.";



/* Available functions */
static PyObject *setNspw(PyObject *self, PyObject *args);
static PyObject *setData(PyObject *self, PyObject *args);
static PyObject *clearPointers(PyObject *self, PyObject *args);
static PyObject *setNCPU(PyObject *self, PyObject *args);
static PyObject *modelcomp(PyObject *self, PyObject *args);
static PyObject *setModel(PyObject *self, PyObject *args);
static PyObject *setWork(PyObject *self, PyObject *args);
static PyObject *unsetWork(PyObject *self, PyObject *args);
static PyObject *QuinnFF(PyObject *self, PyObject *args);


void *writemod(void *work);


/* Module specification */
static PyMethodDef module_methods[] = {
    {"setData", setData, METH_VARARGS, setData_docstring},
    {"setNspw", setNspw, METH_VARARGS, setNspw_docstring},
    {"setModel", setModel, METH_VARARGS, setModel_docstring},
    {"setNCPU", setNCPU, METH_VARARGS, setNCPU_docstring},
    {"modelcomp", modelcomp, METH_VARARGS, uvmultimodel_docstring},
    {"setWork", setWork, METH_VARARGS, setWork_docstring},
    {"unsetWork", unsetWork, METH_VARARGS, unsetWork_docstring},
    {"QuinnFF", QuinnFF, METH_VARARGS, QuinnFF_docstring},
    {"clearPointers", clearPointers, METH_VARARGS, clearPointers_docstring},
    {NULL, NULL, 0, NULL}
};


// normally abort() is called on problems, which breaks CASA.
// here we report and save the error condition which can be
// noticed for a cleaner exit.
//int gsl_death_by = GSL_SUCCESS;
//static void gsl_death(const char * reason, const char * file,
//    int line, int gsl_errno) {
//    // stderr does not end up synchronized with stdout
//    printf("GSL Death by '%s' in file %s at line %d: GSL Error %d\n",
//        reason, file, line, gsl_errno);
//    fflush(stdout); std::cout << std::flush;
//    gsl_death_by = gsl_errno;
//}







typedef std::complex<double> cplx64;


static int Nspw, NCPU, nui, cIF, ncomp, npar, HankelOrder=1, maxnchan=0;
static int mode, Nants, NphClos, NampClos, Nbas;
static double cosDecRef, sinDecRef;
static bool compFixed, isModel, MixedG, isTime, doAmpClos, doPhClos, doClos, onlyClos, doDump; 
static double phClosWgt, ampClosWgt;
static int NparMax = 7; // Degrees of freedom for components (i.e., RA, Dec, Flux, Major, Ratio, PA, Inner).



// ALL POINTERS TO DATA AND METADATA:
struct DATA {
int **ants[2];
int **dtIndex;
double **freqs;
double **uv[3]; 
double **wgt[2]; 
double **dt; 
double **dtArray; 
double **RAshift;
double **Decshift; 
double **Stretch;
cplx64 **ObsVis;
int **fittable;
int **isGain;
int *nnu; 
int *nt; 
double *phaseCenter; 
};


// ALL POINTERS TO MODEL-RELATED STUFF:
struct MODEL {
cplx64 **ModVis;
cplx64 ****Gain;
double **vars;
double **fixp; 
double *Chi2;
double **WorkHess;
double **WorkGrad; 
int **parAnt;
int *nparAnt;
int *models;
double *Hessian;
double *Gradient; 
double *dpar; 
double *muRA; 
double *muDec;
int **BasIdx;
int **phaseClosIdx;
int **ampClosIdx;
double *closBufferWgt;
cplx64 *closBuffer;
cplx64 **closBufferMod;
double *closBufferAbs;
double **closBufferAbsMod;
double *closBufferLog;
double **closBufferLogMod;
int *triSpec;
};


/* Structure to pass, as void cast, to the workers */
struct SHARED_DATA {
int *t0;
int *t1;
int Iam;
};



static SHARED_DATA master;
static SHARED_DATA *worker;

static DATA vis;
static MODEL mod;




/* Initialize the module */

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef pc_module_def = {
    PyModuleDef_HEAD_INIT,
    "_uvmultimodel",         /* m_name */
    module_docstring,       /* m_doc */
    -1,                     /* m_size */
    module_methods,         /* m_methods */
    NULL,NULL,NULL,NULL     /* m_reload, m_traverse, m_clear, m_free */
};
PyMODINIT_FUNC PyInit__uvmultimodel(void)
{
    PyObject *m = PyModule_Create(&pc_module_def);
  //  import_array();
  //  (void)gsl_set_error_handler(gsl_death);

//////////////////////
// Initiate variables with dummy values:
    NCPU = 0; Nspw = 0; npar = -1; Nants = 0; ncomp = -1; Nbas = 0;
    master.t0 = new int[1];
    master.t1 = new int[1];

    return(m);
}



#else

PyMODINIT_FUNC init_uvmultimodel(void)
{
    PyObject *m = Py_InitModule3("_uvmultimodel", module_methods, module_docstring);
    if (m == NULL)
        return;

//////////////////////
// Initiate variables with dummy values:
    NCPU = 0; Nspw = 0; npar = -1; Nants = 0; ncomp = -1; Nbas = 0;
    master.t0 = new int[1];
    master.t1 = new int[1];

////////////////////

    /* Load `numpy` functionality. */
    import_array();
}

#endif




/* Clears pointers to the data
WARNING: In principle, this shouldn't be done, since the
data in these pointers may still be used by Python!!
*/
void clearData(){

/*
      int i,j;
      delete vis.ants[0];
      delete vis.ants[1];
      delete vis.uv[0];
      delete vis.uv[1];
      delete vis.uv[2];
      delete vis.wgt[0];
      delete vis.wgt[1];
      delete vis.freqs;
      delete vis.dt;
      delete vis.dtArray;
      delete vis.dtIndex;
      delete vis.RAshift;
      delete vis.Decshift;
      delete vis.Stretch;
      delete vis.ObsVis;
      delete mod.ModVis;
      delete vis.fittable;
      delete vis.isGain;
    for(i=0;i<Nspw;i++){
      for(j=0;j<Nants;j++){
        delete mod.Gain[i][j];
      };
      delete mod.Gain[i];
    };


if(Nspw > 0){
    delete vis.nnu;
    delete vis.nt;
};
*/

return;

};



/* WARNING! There are pointers here that may still
be used by Python! */
void clearModel(){
  int i;

    if(isModel){
//      delete mod.vars;
//      delete mod.fixp;
//      delete mod.parAnt;
//      delete mod.nparAnt;
//      delete mod.models;
      delete mod.closBufferWgt;
      delete mod.closBuffer;
      delete mod.closBufferAbs;
      delete mod.closBufferLog;
      for(i=0;i<npar+1;i++){
        delete mod.closBufferMod[i];
        delete mod.closBufferAbsMod[i];
        delete mod.closBufferLogMod[i];
      };
      delete mod.closBufferMod;
      delete mod.closBufferAbsMod;
      delete mod.closBufferLogMod;
      for(i=0;i<2;i++){delete mod.BasIdx[i];};
      for(i=0;i<3;i++){delete mod.phaseClosIdx[i];};
      for(i=0;i<4;i++){delete mod.ampClosIdx[i];};
      delete mod.BasIdx; delete mod.phaseClosIdx; delete mod.ampClosIdx;
    };

isModel = false;

return;

};


static PyObject *clearPointers(PyObject *self, PyObject *args)
{

  int i;
  if (!PyArg_ParseTuple(args, "i",&i)){printf("FAILED clearPointers!\n"); fflush(stdout);  PyObject *ret = Py_BuildValue("i",-1); return ret;};


  switch (i){

   case 0: clearData(); break;
   case 1: clearModel(); break;
   case 2: 
    NCPU = -1; Nspw = -1; npar = -1; Nants = -1; ncomp = -1; Nbas = -1;
    master.t0 = new int[1];
    master.t1 = new int[1];
    isModel = false;
  };


//  if (i==0){clearData();} else {clearModel();}
  PyObject *ret = Py_BuildValue("i",0);
  return ret;



};






// Taylor expansion of the Hankel transform, for special models (GaussianRing):
void GridModel(int imod, double UVRad, double *varp, int *currow, double &tempAmp) {


double UVRadP = 1.0;
double UVRadSq = UVRad*UVRad/4.;
int i;

if (imod == 9){
  tempAmp = 0.0;
  for (i=0;i<HankelOrder;i++){
     tempAmp += varp[currow[NparMax+i]]*UVRadP;
     UVRadP *= UVRadSq;
  };


};

};


















///////////////////////////////////////
/* Code for the workers (NOT CALLABLE FROM PYTHON). */

void *writemod(void *work) {

SHARED_DATA *mydata = (SHARED_DATA *)work;


int nu0, nu1  ;
if (nui==-1){nu0=0;nu1=vis.nnu[cIF];}else{nu0=nui;nu1=nui+1;};

int k,i,t,j,m,p, ii, currow[NparMax+HankelOrder];

double phase, uu, vv, ww, UVRad, Ampli, DerivRe[npar], DerivIm[npar];
double SPA, CPA, tA, tB, tempChi, deltat, Ellip, Ellip0;
Ellip0 = 1.; Ellip = 1.;
double wgtcorrA, tempRe[npar+1], tempIm[npar+1], tempAmp;
tempChi = 0.0;

const double deg2rad = 0.017453292519943295;
const double sec2rad = 3.1415926535/180./3600.; 
const double radsec2 = pow(3.1415926535/180./3600.,2.); // 2.3504430539097885e-11;
int Iam = mydata->Iam; 
int widx = (Iam == -1)?0:Iam;
//int pmax = (mode == -1)?npar+1:1;
//int mmax = (mode == 0)?1:ncomp;
int pmax, mmax;
bool write2mod, writeDer;

switch (mode){
  case  0: mmax = 1;      write2mod = true;         // COMPUTE FIXED MODEL
           pmax = 1;      writeDer  = false; break;

  case -1: mmax = ncomp;  write2mod = false;        // HESS & ERRORS & CHISQ
           pmax = npar+1; writeDer = true;   break; 

  case -2: mmax = ncomp;  write2mod = false;        // GET CHISQ
           pmax = 1;      writeDer  = false; break; 

  case -3: mmax = ncomp;  write2mod = true;         // ADD VARMOD TO FIXED
           pmax = 1;      writeDer  = false; break;

  case -4: mmax = ncomp;  write2mod = true;         // ADD VARMOD TO FIXED
           pmax = 1;      writeDer  = false; break;

  case -5: mmax = 0;      write2mod = false;        // CALIBRATE
           pmax = 1;      writeDer  = false; break;

  default: mmax = ncomp;  write2mod = false;        // GET CHISQ
           pmax = 1;      writeDer  = false; break; 



};



FILE* phClosFile;
FILE* apClosFile;

if(doDump){
  if(doPhClos){ phClosFile = fopen("UVMultiFit_Phase_Closures.dat", "wb");};
  if(doAmpClos){apClosFile = fopen("UVMultiFit_Amplitude_Closures.dat", "wb");};
};


//bool write2mod = (mode == -3 || mode == -4 || mode == -5 || mode >= 0);
//bool writeDer = (mode==-1);

bool TriInv;
double tempD, tempR, tempI, cosphase, sinphase, cosphase0, sinphase0, PA; 
cplx64 *totGain;
double wterm, rsh, dsh, wamp, tempres0, tempres1, ll, mm, PBcorr;
totGain = new cplx64[pmax];

int ant1, ant2, pdep, currTIdx, kT;
cplx64 GCplx;
bool calibrate, PBlimit;

///////////////
// Arrays to compute the closures:
int ClosTime = vis.dtIndex[cIF][mydata->t0[cIF]];
int basIdx;

for(j=0;j<Nbas;j++){
  mod.closBuffer[j] = 0.0;
  mod.closBufferWgt[j] = 0.0;
  mod.closBufferAbs[j] = 0.0;
  mod.closBufferLog[j] = 0.0;
  for(i=0;i<npar+1;i++){
    mod.closBufferMod[i][j] = 0.0;
    mod.closBufferAbsMod[i][j] = 0.0;
    mod.closBufferLogMod[i][j] = 0.0;
  };
};


bool NewClosure = true;
bool Observed[Nbas];
int isTriSpec[Nbas];
double auxAmp, auxAmp0, auxAmp1;
cplx64 TriSpec0, TriSpec1, TriSpec;

int tri0,tri1,tri2,other0,other1,other2;
cplx64 BL1, BL2, BL3, auxBL;
cplx64 *BL1m, *BL2m, *BL3m;
BL1m = new cplx64[npar+1];
BL2m = new cplx64[npar+1];
BL3m = new cplx64[npar+1];

///////////////




for(ii=0;ii<Nbas;ii++){
  Observed[ii]=false;
  isTriSpec[ii]=0;
};

for (t = mydata->t0[cIF]; t < mydata->t1[cIF] ; t++) {


 deltat = vis.dt[cIF][t];

 currTIdx = vis.dtIndex[cIF][t];
 NewClosure = currTIdx == ClosTime;
 if (!NewClosure){ClosTime = currTIdx;};
 ant1 = vis.ants[0][cIF][t];
 ant2 = vis.ants[1][cIF][t];

// Figure out baseline index of this visibility:
 for(basIdx=0;basIdx<Nbas;basIdx++){
   if(mod.BasIdx[0][basIdx]==ant1 && mod.BasIdx[1][basIdx]==ant2){break;};
 }; 

// Figure out if a "perfect" antenna is observing here (only one is admitted):
   for(i=0;i<Nants;i++){
     if(mod.triSpec[i]==1){
       if(isTriSpec[basIdx]==0){
         if(ant1==i){isTriSpec[basIdx]=i+1;} else if(ant2==i){isTriSpec[basIdx]=-i-1;};
       };
     };
   };
// };

//if(isTriSpec[basIdx]!=0){printf("TriSpec found for %i-%i (baseline %i)\n",ant1,ant2,basIdx);};


 if (vis.fittable[cIF][t]!=0) {







// Compute closures when we change integration time:

 if ( (!NewClosure) && (doClos) && (doAmpClos || doPhClos) ){

   for(ii=0;ii<Nbas;ii++){
     mod.closBufferAbs[ii] = std::abs(mod.closBuffer[ii]); 
     mod.closBufferLog[ii] = std::log(mod.closBufferAbs[ii]);
     for(j=0; j<npar+1; j++){
       mod.closBufferAbsMod[j][ii] = std::abs(mod.closBufferMod[j][ii]);
       mod.closBufferLogMod[j][ii] = std::log(mod.closBufferAbsMod[j][ii]);
     };
   };



  // if (writeDer) {



// Closure Phases:
    for(ii=0;ii<NphClos;ii++){

      if(Observed[mod.phaseClosIdx[0][ii]] && Observed[mod.phaseClosIdx[1][ii]] && Observed[mod.phaseClosIdx[2][ii]] && mod.closBufferWgt[mod.phaseClosIdx[0][ii]]>0.0 && mod.closBufferWgt[mod.phaseClosIdx[1][ii]] && mod.closBufferWgt[mod.phaseClosIdx[2][ii]]>0.0){

// Error propagation:
        tempD = 1./(mod.closBufferWgt[mod.phaseClosIdx[0][ii]]);
        tempD += 1./(mod.closBufferWgt[mod.phaseClosIdx[1][ii]]);
        tempD += 1./(mod.closBufferWgt[mod.phaseClosIdx[2][ii]]);
// Scale errors (relative to visibilties:
        tempD = phClosWgt/tempD;

// Arrange visibilities in the baseline variables:
        BL1 = mod.closBuffer[mod.phaseClosIdx[0][ii]];
        BL2 = mod.closBuffer[mod.phaseClosIdx[1][ii]];
        BL3 = mod.closBuffer[mod.phaseClosIdx[2][ii]];
        for(p=0;p<npar+1;p++){
          BL1m[p] = mod.closBufferMod[p][mod.phaseClosIdx[0][ii]];
          BL2m[p] = mod.closBufferMod[p][mod.phaseClosIdx[1][ii]];
          BL3m[p] = mod.closBufferMod[p][mod.phaseClosIdx[2][ii]];
        };



        TriSpec = BL1*BL2/BL3;
        TriSpec0 = BL1m[0]*BL2m[0]/BL3m[0];

        // Module of the trispectrum model:
        tempR = mod.closBufferAbsMod[0][mod.phaseClosIdx[0][ii]]*mod.closBufferAbsMod[0][mod.phaseClosIdx[1][ii]]/mod.closBufferAbsMod[0][mod.phaseClosIdx[2][ii]];
        // Module of the observed trispectrum:
        tempI = mod.closBufferAbs[mod.phaseClosIdx[0][ii]]*mod.closBufferAbs[mod.phaseClosIdx[1][ii]]/mod.closBufferAbs[mod.phaseClosIdx[2][ii]];

       // Normalize, to get the closure phases:
        TriSpec0 /= tempR;
        TriSpec /= tempI;

      // Closure residuals:
        tempres0 = (TriSpec.real() - TriSpec0.real());
        tempres1 = (TriSpec.imag() - TriSpec0.imag());

        tempChi += (tempres0*tempres0 + tempres1*tempres1)*tempD;        

        if(doDump){
           fwrite(&mod.BasIdx[0][mod.phaseClosIdx[0][ii]],sizeof(int),1,phClosFile);
           fwrite(&mod.BasIdx[1][mod.phaseClosIdx[0][ii]],sizeof(int),1,phClosFile);
           fwrite(&mod.BasIdx[1][mod.phaseClosIdx[1][ii]],sizeof(int),1,phClosFile);
           fwrite(&deltat,sizeof(double),1,phClosFile);
           tempR = std::arg(TriSpec);
           fwrite(&tempR,sizeof(double),1,phClosFile);
           tempR = std::arg(TriSpec0);
           fwrite(&tempR,sizeof(double),1,phClosFile);
        };


     // Compute all gradients of the closure model w.r.t. fitting parameters:
     if(writeDer){
        for(p=1;p<npar+1;p++){
         // New module of the model trispectrum:
          tempR = mod.closBufferAbsMod[p][mod.phaseClosIdx[0][ii]]*mod.closBufferAbsMod[p][mod.phaseClosIdx[1][ii]]/mod.closBufferAbsMod[p][mod.phaseClosIdx[2][ii]];
         // Perturbated trispectrum:
          TriSpec1 = BL1m[p]*BL2m[p]/BL3m[p];
         // Normalize, to get closure phases:
          TriSpec1/= tempR ; 

      // Compute numerical derivatives and gradient vector:
          DerivRe[p-1] = (TriSpec1.real() - TriSpec0.real())/mod.dpar[p-1];
          DerivIm[p-1] = (TriSpec1.imag() - TriSpec0.imag())/mod.dpar[p-1];
          mod.WorkGrad[widx][p-1] += tempres0*tempD*DerivRe[p-1];
          mod.WorkGrad[widx][p-1] += tempres1*tempD*DerivIm[p-1];
        };


    // Add to the Hessian:
        for(p=0;p<npar;p++){
          for(m=p;m<npar;m++){
          mod.WorkHess[widx][npar*p+m] += tempD*(DerivRe[p]*DerivRe[m]+DerivIm[p]*DerivIm[m]);
        };
      };

     };

//////////////////////////
// FOR TESTING:
//      tempD *= 1000.;
//////////////////////////

///////////////////////////////////////////////////////
   // Find out if this closure is a trispectrum:

   for(p=0;p<Nants;p++){
     if(mod.triSpec[p]==1){
       tri0 = 0; tri1 = 0; tri2 = 0; other0 = -1; other1 = -1; other2=-1;
              if(mod.BasIdx[0][mod.phaseClosIdx[0][ii]]==p){tri0=1;other0=mod.BasIdx[1][mod.phaseClosIdx[0][ii]];
       } else if(mod.BasIdx[1][mod.phaseClosIdx[0][ii]]==p){tri0=2;other0=mod.BasIdx[0][mod.phaseClosIdx[0][ii]];};
              if(mod.BasIdx[0][mod.phaseClosIdx[1][ii]]==p){tri1=1;other1=mod.BasIdx[1][mod.phaseClosIdx[1][ii]];
       } else if(mod.BasIdx[1][mod.phaseClosIdx[1][ii]]==p){tri1=2;other1=mod.BasIdx[0][mod.phaseClosIdx[1][ii]];};
              if(mod.BasIdx[0][mod.phaseClosIdx[2][ii]]==p){tri2=1;other2=mod.BasIdx[1][mod.phaseClosIdx[2][ii]];
       } else if(mod.BasIdx[1][mod.phaseClosIdx[2][ii]]==p){tri2=2;other2=mod.BasIdx[0][mod.phaseClosIdx[2][ii]];};

// Check all possibilities:

//// CASE 1: REFANT APPEARING FIRST IN ALL BASELINES:
       j=0;
       if(tri0==1 && tri1==1){ //if(ii%128){printf("CASE 1-1\n");};
         BL1 = std::conj(BL1);
         for(j=0;j<npar+1;j++){BL1m[j]=std::conj(BL1m[j]);};
         if(other0==mod.BasIdx[1][mod.phaseClosIdx[2][ii]]){BL3=std::conj(BL3); for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=1;
       } else if(tri0==1 && tri2==1) { //if(ii%128){printf("CASE 1-2\n");};
         BL1 = std::conj(BL1); auxBL = BL2; BL2 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL2m[j]; BL2m[j] = BL3m[j]; BL3m[j] = auxBL;};
         for(j=0;j<npar+1;j++){BL1m[j]=std::conj(BL1m[j]);};
         if(other0==mod.BasIdx[1][mod.phaseClosIdx[1][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=2;
       } else if(tri1==1 && tri2==1) { //if(ii%128){printf("CASE 1-3\n");};
         auxBL = BL1; BL1 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL1m[j]; BL1m[j] = BL3m[j]; BL3m[j] = auxBL;};
         BL1 = std::conj(BL1);
         for(j=0;j<npar+1;j++){BL1m[j]=std::conj(BL1m[j]);};
         if(other2==mod.BasIdx[1][mod.phaseClosIdx[0][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=3;
//// CASE 2: REFANT APPEARING FIRST AND SECOND:
       } else if(tri0==1 && tri1==2){ //if(ii%128){printf("CASE 2-1\n");};
         auxBL = BL1; BL1 = BL2; BL2 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL1m[j]; BL1m[j] = BL2m[j]; BL2m[j] = auxBL;};
         if(other1==mod.BasIdx[1][mod.phaseClosIdx[2][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=4;
       } else if(tri0==1 && tri2==2){ //if(ii%128){printf("CASE 2-2\n");};
         auxBL = BL1; BL1 = BL2; BL2 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL1m[j]; BL1m[j] = BL2m[j]; BL2m[j] = auxBL;};
         auxBL = BL1; BL1 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL1m[j]; BL1m[j] = BL3m[j]; BL3m[j] = auxBL;};
         if(other2==mod.BasIdx[1][mod.phaseClosIdx[1][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=5;
       } else if(tri1==1 && tri2==2){ //if(ii%128){printf("CASE 2-3\n");};
         auxBL = BL1; BL1 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL1m[j]; BL1m[j] = BL3m[j]; BL3m[j] = auxBL;};
         if(other2==mod.BasIdx[1][mod.phaseClosIdx[0][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=6;
//// CASE 3: REFANT APPEARING SECOND AND FIRST:
       } else if(tri0==2 && tri1==1){ //if(ii%128){printf("CASE 3-1\n");};
         if(other0==mod.BasIdx[1][mod.phaseClosIdx[2][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=7;
       } else if(tri0==2 && tri2==1){ //if(ii%128){printf("CASE 3-2\n");};
         auxBL = BL2; BL2 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL2m[j]; BL2m[j] = BL3m[j]; BL3m[j] = auxBL;};
         if(other0==mod.BasIdx[1][mod.phaseClosIdx[1][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=8;
       } else if(tri1==2 && tri2==1){ //if(ii%128){printf("CASE 3-3\n");};
         auxBL = BL2; BL2 = BL1; BL1 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL1m[j]; BL1m[j] = BL3m[j]; BL3m[j] = auxBL;};
         auxBL = BL2; BL2 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL2m[j]; BL2m[j] = BL3m[j]; BL3m[j] = auxBL;};
         if(other1==mod.BasIdx[1][mod.phaseClosIdx[0][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=9;
//// CASE 4: REFANT APPEARING SECOND IN ALL BASELINES:
       } else if(tri0==2 && tri1==2){ //if(ii%128){printf("CASE 4-1\n");};
         BL2 = std::conj(BL2);
         for(j=0;j<npar+1;j++){BL2m[j]=std::conj(BL2m[j]);};
         if(other0==mod.BasIdx[1][mod.phaseClosIdx[2][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=10;
       } else if(tri0==2 && tri2==2){ //if(ii%128){printf("CASE 4-2\n");};
         auxBL = BL2; BL2 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL2m[j]; BL2m[j] = BL3m[j]; BL3m[j] = auxBL;};
         BL2 = std::conj(BL2);
         for(j=0;j<npar+1;j++){BL2m[j]=std::conj(BL2m[j]);};
         if(other0==mod.BasIdx[1][mod.phaseClosIdx[1][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=11;
       } else if(tri1==2 && tri2==2){ //if(ii%128){printf("CASE 4-3\n");};
         auxBL = BL1; BL1 = BL3; BL3 = auxBL; for(j=0;j<npar+1;j++){auxBL = BL1m[j]; BL1m[j] = BL3m[j]; BL3m[j] = auxBL;};
         BL2 = std::conj(BL2);
         for(j=0;j<npar+1;j++){BL2m[j]=std::conj(BL2m[j]);};
         if(other2==mod.BasIdx[1][mod.phaseClosIdx[0][ii]]){BL3=std::conj(BL3);for(j=0;j<npar+1;j++){BL3m[j]=std::conj(BL3m[j]);};};
         j=12;
       };

/////////////////////////////
// Add to the fit!
       if(tri0!=0 || tri1!=0 || tri2!=0){
   //     if(j==0){printf("PROBLEM!\n");};
        TriSpec = BL1*BL2/BL3;
        TriSpec0 = BL1m[0]*BL2m[0]/BL3m[0];
        TriInv = std::abs(TriSpec0)>1.0;
        if(TriInv){TriSpec = 1./TriSpec; TriSpec0 = 1./TriSpec0;};
        tempres0 = (TriSpec.real() - TriSpec0.real());
        tempres1 = (TriSpec.imag() - TriSpec0.imag());

    //    printf("TriSpec: %i-%i / %i-%i / %i-%i:  %.4e/%.4e ;   %.4e/%.4e\n",mod.BasIdx[0][mod.phaseClosIdx[0][ii]],mod.BasIdx[1][mod.phaseClosIdx[0][ii]],mod.BasIdx[0][mod.phaseClosIdx[1][ii]],mod.BasIdx[1][mod.phaseClosIdx[1][ii]],mod.BasIdx[0][mod.phaseClosIdx[2][ii]],mod.BasIdx[1][mod.phaseClosIdx[2][ii]],TriSpec.real(),TriSpec0.real(), TriSpec.imag(),TriSpec0.imag());

        tempChi += (tempres0*tempres0 + tempres1*tempres1)*tempD;


       if(writeDer){

        for(j=1;j<npar+1;j++){
          TriSpec1 = BL1m[j]*BL2m[j]/BL3m[j];
          if(TriInv){TriSpec1 = 1./TriSpec1;};
          DerivRe[j-1] = (TriSpec1.real() - TriSpec0.real())/mod.dpar[j-1];
          DerivIm[j-1] = (TriSpec1.imag() - TriSpec0.imag())/mod.dpar[j-1];
          mod.WorkGrad[widx][j-1] += tempres0*tempD*DerivRe[j-1];
          mod.WorkGrad[widx][j-1] += tempres1*tempD*DerivIm[j-1];
        };


    // Add to the Hessian:
        for(j=0;j<npar;j++){
            for(m=j;m<npar;m++){
            mod.WorkHess[widx][npar*j+m] += tempD*(DerivRe[j]*DerivRe[m]+DerivIm[j]*DerivIm[m]);
          };
        };

       };

       };       
/////////////////////////////

     };
   };
///////////////////////////////////////////////////////

    };
   };



// Closure Amplitudes:
    for(ii=0;ii<NampClos;ii++){
      if(Observed[mod.ampClosIdx[0][ii]] && Observed[mod.ampClosIdx[1][ii]] && Observed[mod.ampClosIdx[2][ii]] && Observed[mod.ampClosIdx[3][ii]]){
      //  auxAmp1 = mod.closBufferAbs[mod.ampClosIdx[0][ii]];
        tempD = 1./(mod.closBufferWgt[mod.ampClosIdx[0][ii]]);// *auxAmp1);//*auxAmp1);
      //  auxAmp1 = mod.closBufferAbs[mod.ampClosIdx[1][ii]];
        tempD += 1./(mod.closBufferWgt[mod.ampClosIdx[1][ii]]);// *auxAmp1);//*auxAmp1);
      //  auxAmp1 = mod.closBufferAbs[mod.ampClosIdx[2][ii]];
        tempD += 1./(mod.closBufferWgt[mod.ampClosIdx[2][ii]]);// *auxAmp1);//*auxAmp1);
      //  auxAmp1 = mod.closBufferAbs[mod.ampClosIdx[3][ii]];
        tempD += 1./(mod.closBufferWgt[mod.ampClosIdx[3][ii]]);// *auxAmp1);//*auxAmp1);
        tempD = ampClosWgt/tempD;
        auxAmp = mod.closBufferLog[mod.ampClosIdx[0][ii]]+mod.closBufferLog[mod.ampClosIdx[1][ii]]-(mod.closBufferLog[mod.ampClosIdx[2][ii]]+mod.closBufferLog[mod.ampClosIdx[3][ii]]);
        auxAmp0 = mod.closBufferLogMod[0][mod.ampClosIdx[0][ii]]+mod.closBufferLogMod[0][mod.ampClosIdx[1][ii]]-(mod.closBufferLogMod[0][mod.ampClosIdx[2][ii]]+mod.closBufferLogMod[0][mod.ampClosIdx[3][ii]]);


        tempres0 = (auxAmp-auxAmp0);
        tempChi += tempres0*tempres0*tempD;

        if(doDump){
           fwrite(&mod.BasIdx[0][mod.ampClosIdx[0][ii]],sizeof(int),1,apClosFile);
           fwrite(&mod.BasIdx[1][mod.ampClosIdx[0][ii]],sizeof(int),1,apClosFile);
           fwrite(&mod.BasIdx[0][mod.ampClosIdx[1][ii]],sizeof(int),1,apClosFile);
           fwrite(&mod.BasIdx[1][mod.ampClosIdx[1][ii]],sizeof(int),1,apClosFile);
           fwrite(&deltat,sizeof(double),1,apClosFile);
           fwrite(&auxAmp,sizeof(double),2,apClosFile);
           fwrite(&auxAmp0,sizeof(double),2,apClosFile);
        };


      if(writeDer){

        for(p=1;p<npar+1;p++){
          auxAmp1 = mod.closBufferLogMod[p][mod.ampClosIdx[0][ii]]+mod.closBufferLogMod[p][mod.ampClosIdx[1][ii]]-(mod.closBufferLogMod[p][mod.ampClosIdx[2][ii]]+mod.closBufferLogMod[p][mod.ampClosIdx[3][ii]]);
          DerivRe[p-1] = (auxAmp1-auxAmp0)/mod.dpar[p-1];
          mod.WorkGrad[widx][p-1] += tempres0*tempD*DerivRe[p-1];
        };



        for(p=0;p<npar;p++){
          for(m=p;m<npar;m++){
          mod.WorkHess[widx][npar*p+m] += tempD*(DerivRe[p]*DerivRe[m]);
        };
      };

     };

      };
    };


 // };



   for(j=0;j<Nbas;j++){
     Observed[j] = false;
     isTriSpec[j] = 0;
     mod.closBuffer[j] = 0.0;
     mod.closBufferWgt[j] = 0.0;
     mod.closBufferAbs[j] = 0.0;
     mod.closBufferLog[j] = 0.0;
     for(ii=0;ii<npar+1;ii++){
       mod.closBufferMod[ii][j] = 0.0;
       mod.closBufferAbsMod[ii][j] = 0.0;
       mod.closBufferLogMod[ii][j] = 0.0;
     };
   };


 };   // Comes from if(doClos...)





















  for (i = nu0; i < nu1; i++) {

    j = (nui!=-1)?0:i;
    k = vis.nnu[cIF]*t+i;
    
    kT = MixedG?vis.nnu[cIF]*currTIdx+i:vis.nnu[cIF];
    tempD = vis.wgt[0][cIF][k]*vis.wgt[0][cIF][k];



   if (vis.isGain[cIF][t]!=0) {



    for(p=0;p<pmax;p++){

     calibrate = false;
     GCplx = cplx64(1.0) ;  


     for (pdep=0; pdep<mod.nparAnt[ant1]; pdep++){
       if (mod.parAnt[ant1][pdep]== p){
         if(MixedG){
           GCplx *= mod.Gain[cIF][ant1][pdep][kT];
         } else {
           GCplx *= mod.Gain[cIF][ant1][pdep][i]*mod.Gain[cIF][ant1][pdep][kT+currTIdx];
         };
         calibrate=true;
       };
     };

     if(!calibrate){
       if (MixedG){
         GCplx *= mod.Gain[cIF][ant1][0][kT];
       } else {
         GCplx *= mod.Gain[cIF][ant1][0][i]*mod.Gain[cIF][ant1][0][kT+currTIdx];
       };
     };


     calibrate = false;

     for (pdep=0; pdep<mod.nparAnt[ant2]; pdep++){
       if (mod.parAnt[ant2][pdep]== p){
         if (MixedG){
           GCplx *= std::conj(mod.Gain[cIF][ant2][pdep][kT]);
         } else {
           GCplx *= std::conj(mod.Gain[cIF][ant2][pdep][i]*mod.Gain[cIF][ant2][pdep][kT+currTIdx]);
         };
         calibrate=true;
       };
     };

     if(!calibrate){
       if (MixedG){
         GCplx *= std::conj(mod.Gain[cIF][ant2][0][kT]);
       } else {
         GCplx *= std::conj(mod.Gain[cIF][ant2][0][i]*mod.Gain[cIF][ant2][0][kT+currTIdx]);
       };
     };

     totGain[p] = GCplx;
    };
   }; 



    uu = vis.freqs[cIF][i]*vis.uv[0][cIF][t];
    vv =  vis.freqs[cIF][i]*vis.uv[1][cIF][t];
    ww =  vis.freqs[cIF][i]*vis.uv[2][cIF][t];

    for(p=0;p<pmax;p++){
      tempRe[p]=0.0; tempIm[p]=0.0;
    };

















    for (m=0;m<mmax;m++){

      PBlimit = false;

      if(!isTime){
        for(p=0;p<NparMax+HankelOrder;p++){currow[p] = m*maxnchan*(NparMax+HankelOrder)+j+p*maxnchan;};
      } else {
        for(p=0;p<NparMax+HankelOrder;p++){currow[p] = m*maxnchan*(NparMax+HankelOrder)+currTIdx+p*maxnchan;};
      };

    //   printf("MOD %i; PARAMS: %.3e %.3e %.3e %.3e %.3e %.3e\n",m,mod.vars[0][currow[0]],mod.vars[0][currow[1]],mod.vars[0][currow[2]],mod.vars[0][currow[3]],mod.vars[0][currow[4]],mod.vars[0][currow[5]]);



      for(p=0;p<pmax;p++){

     //  if(!PBlimit && (p==0 || (mod.vars[p][currow[0]] != mod.vars[0][currow[0]] || mod.vars[p][currow[1]] != mod.vars[0][currow[1]]))){ 


       if(!PBlimit){

       if(p==0 || (mod.vars[p][currow[0]] != mod.vars[0][currow[0]] || mod.vars[p][currow[1]] != mod.vars[0][currow[1]])){ 

    // Project RA:
         rsh = (mod.vars[p][currow[0]]+ mod.muRA[m]*deltat)/cosDecRef - vis.RAshift[cIF][t];

    // Project Dec:
         dsh = mod.vars[p][currow[1]] - vis.Decshift[cIF][t] + mod.muDec[m]*deltat;

    // Get source-centered shifts (l,m):
         tempR = vis.Decshift[cIF][t]*sec2rad+vis.phaseCenter[1];
         tempI = tempR + dsh*sec2rad;
         mm = asin(cos(tempR)*sin(tempI)-sin(tempR)*cos(tempI)*cos(rsh*sec2rad))/sec2rad;
         ll = atan(cos(tempI)*sin(rsh*sec2rad)/(cos(tempR)*cos(tempI)*cos(rsh*sec2rad)+sin(tempR)*sin(tempI)))/sec2rad;
         PBcorr = vis.wgt[1][cIF][t]*(mm*mm + ll*ll)*vis.freqs[cIF][i]*vis.freqs[cIF][i];
    //     PBcorr = vis.wgt[1][cIF][m*vis.nt[cIF]+t]*(mm*mm + ll*ll)*vis.freqs[cIF][i]*vis.freqs[cIF][i];

	 wgtcorrA = exp(PBcorr); PBlimit = false;
// ACTIVATE THIS TO AVOID OVER-COMPUTING MODEL VISIBILITIES IN VERY LARGE MOSAICS (i.e. 3-SIGMA PBEAM CUTOFF):
//	 if (PBcorr<-3.0){wgtcorrA = exp(PBcorr); PBlimit = false;} else {wgtcorrA = 0.0; PBlimit=true;};

         phase = ll*uu + mm*vv;
         wamp = sqrt(1. - (ll*ll + mm*mm)*radsec2);
         wterm = ww*(wamp - 1.)*sec2rad;
         cosphase = cos(phase+wterm);
         sinphase = sin(phase+wterm);
         if(p==0){cosphase0 = cosphase; sinphase0 = sinphase;};
       } else {cosphase= cosphase0; sinphase = sinphase0; };


        if(p==0 || (mod.vars[p][currow[5]] != mod.vars[0][currow[5]] || mod.vars[p][currow[4]] != mod.vars[0][currow[4]])){ 
          if (mod.models[m] != 0) {

           PA = mod.vars[p][currow[5]]*deg2rad;
           SPA = sin(PA);
           CPA = cos(PA);

         tA = (uu*CPA - vv*SPA)*mod.vars[p][currow[4]];
         tB = (uu*SPA + vv*CPA);
         Ellip = sqrt(tA*tA+tB*tB);} else {Ellip = 1.0;};
         if(p==0){Ellip0 = Ellip;};


       } else {Ellip = Ellip0;};


//        if( !PBlimit && (p==0 || EllipChanged || (mod.vars[p][currow[3]] != mod.vars[0][currow[3]] || mod.vars[p][currow[6]] != mod.vars[0][currow[6]]) )){
 

         if (mod.models[m] != 0){

         UVRad = Ellip*(mod.vars[p][currow[3]]/2.0);

         tempAmp = 1.0;
         if (mod.models[m] > 8) {GridModel(mod.models[m], UVRad, mod.vars[p], currow, tempAmp);};

         if (UVRad > 0.0) {
          switch (mod.models[m]) {
            case 1: Ampli = exp(-0.3606737602*UVRad*UVRad); break;
            case 2: Ampli = 2.0*gsl_sf_bessel_J1(UVRad)/UVRad; break;
            case 3: Ampli = gsl_sf_bessel_J0(UVRad); break;
            case 4: Ampli = 3.0*(sin(UVRad)-UVRad*cos(UVRad))/(UVRad*UVRad*UVRad); break;
            case 5: Ampli = sin(UVRad)/UVRad; break;
            case 6: Ampli = pow(1.+2.0813689810056077*UVRad*UVRad,-1.5); break;
            case 7: Ampli = 0.459224094*gsl_sf_bessel_K0(UVRad); break;
            case 8: Ampli = exp(-UVRad*1.3047660265); break;
            default: Ampli = tempAmp; 
          };
        } else {vis.wgt[0][cIF][k]=0.0; Ampli=1.0;};

       } else {Ampli = 1.0;};

    Ampli *= wgtcorrA;

   } else {Ampli = 0.0;};




   if (!PBlimit) {
     if (vis.isGain[cIF][t]!=0) {
      tempR = mod.vars[p][currow[2]]*Ampli*cosphase;
      tempI = mod.vars[p][currow[2]]*Ampli*sinphase;
      tempRe[p] += totGain[p].real()*tempR - totGain[p].imag()*tempI;
      tempIm[p] += totGain[p].imag()*tempR + totGain[p].real()*tempI;
     } else {
      tempRe[p] += mod.vars[p][currow[2]]*Ampli*cosphase;
      tempIm[p] += mod.vars[p][currow[2]]*Ampli*sinphase;
     };
   };


  };

};



//////////////


if(compFixed && mode != -5){ // Add from model array (scaling with flux)
 //  if (vis.isGain[cIF][t]!=0) {
      for(p=0;p<pmax;p++){
       if (vis.isGain[cIF][t]!=0) {
         tempR = mod.ModVis[cIF][k].real()*mod.fixp[p][j];
         tempI = mod.ModVis[cIF][k].imag()*mod.fixp[p][j];
         tempRe[p] += totGain[p].real()*tempR - totGain[p].imag()*tempI;
         tempIm[p] += totGain[p].imag()*tempR + totGain[p].real()*tempI;
       } else {
         tempRe[p] += mod.ModVis[cIF][k].real()*mod.fixp[p][j];
         tempIm[p] += mod.ModVis[cIF][k].imag()*mod.fixp[p][j];
       };
     };
 //  };
};


if (write2mod) { // Write to model array:
  mod.ModVis[cIF][k] = cplx64(tempRe[0],tempIm[0]);
}; 



//////////////



// Save residuals instead:
  if (mode == -4){
    mod.ModVis[cIF][k] = vis.ObsVis[cIF][k] - mod.ModVis[cIF][k];
  } else if (mode == -5){
    mod.ModVis[cIF][k] = vis.ObsVis[cIF][k]/totGain[0];
  };


// Save baselines for closures:
  if ((doPhClos || doAmpClos) && doClos){
    Observed[basIdx] = true;
    mod.closBuffer[basIdx] += vis.ObsVis[cIF][k];
    mod.closBufferWgt[basIdx] += tempD;
    for(p=0;p<npar+1;p++){
      mod.closBufferMod[p][basIdx] += cplx64(tempRe[p],tempIm[p]);
    };
  };



  tempres0 = (vis.ObsVis[cIF][k].real()-tempRe[0])*vis.wgt[0][cIF][k] ; 
  tempres1 = (vis.ObsVis[cIF][k].imag()-tempIm[0])*vis.wgt[0][cIF][k] ; 


   if(!onlyClos){

     tempChi += tempres0*tempres0 + tempres1*tempres1;

   if (writeDer) {

    for(p=0;p<npar;p++){
      DerivRe[p] = (tempRe[p+1]-tempRe[0])/mod.dpar[p]; 
      DerivIm[p] = (tempIm[p+1]-tempIm[0])/mod.dpar[p]; 
      mod.WorkGrad[widx][p] += (vis.ObsVis[cIF][k].real()-tempRe[0])*tempD*DerivRe[p];
      mod.WorkGrad[widx][p] += (vis.ObsVis[cIF][k].imag()-tempIm[0])*tempD*DerivIm[p];
    };  

    for(p=0;p<npar;p++){
      for(m=p;m<npar;m++){
        mod.WorkHess[widx][npar*p+m] += tempD*(DerivRe[p]*DerivRe[m] + DerivIm[p]*DerivIm[m]);
      };
    };

   };

  };


  };


  };


};



if (writeDer){
for(p=0;p<npar;p++){
  for(m=p;m<npar;m++){
    mod.WorkHess[widx][npar*m+p] = mod.WorkHess[widx][npar*p+m];
  //  printf("%.4e ",mod.WorkHess[widx][npar*m+p]);
  };
 // printf("\n");
};
};


if(doDump){
  if(doPhClos){fclose(phClosFile);};
  if(doAmpClos){fclose(apClosFile);};
};




if (Iam != -1){mod.Chi2[Iam] = tempChi; pthread_exit((void*) 0);} 
else {mod.Chi2[0] = tempChi; return (void*) 0;};

}
// END OF CODE FOR THE WORKERS
//////////////////////////////////////






















//////////////////////////////////////
// SET THE NUMBER OF IFS (IT REINITIATES ALL SHARED DATA VARIABLES!):
// USAGE FROM PYTHON: setNspw(i) where i is the number of SPW
static PyObject *setNspw(PyObject *self, PyObject *args)
{
    int i;
    if (!PyArg_ParseTuple(args, "i",&i)){printf("FAILED setNspw!\n"); fflush(stdout);  PyObject *ret = Py_BuildValue("i",-1); return ret;}; 


  //  printf("\n     setNspw %i\n\n",i);

// TODO: RUN gc.collect() FROM PYTHON AFTER setNspw.

    if(Nspw>0){clearData();};

    vis.nnu = new int[i];
    vis.nt = new int[i];
    master.t0 = new int[i];
    master.t1 = new int[i];

    vis.freqs = new double*[i];
    
    vis.ants[0] = new int*[i];
    vis.ants[1] = new int*[i];
    vis.dtIndex = new int*[i];
    vis.dtArray = new double*[i];


    vis.uv[0] = new double*[i];
    vis.uv[1] = new double*[i];
    vis.uv[2] = new double*[i];
    vis.ObsVis = new cplx64*[i];
    mod.ModVis = new cplx64*[i];
    vis.fittable = new int*[i];
    vis.isGain = new int*[i];
    vis.wgt[1] = new double*[i];
    vis.wgt[0] = new double*[i];
    vis.dt = new double*[i];
    vis.RAshift = new double*[i];
    vis.Stretch = new double*[i];
    vis.Decshift = new double*[i];
    mod.Gain = new cplx64***[i];

    Nspw = i;

    PyObject *ret = Py_BuildValue("i",0);
    return ret;
}
























/////////////////////////////
// PREPARE PARALLELIZATION (MUST BE RUN AFTER setData)
// USAGE FROM PYTHON: setNCPU(i) where i is the num. of threads allowed
static PyObject *setNCPU(PyObject *self, PyObject *args)
{
    int i, j, k, k0, k1;
    if (!PyArg_ParseTuple(args, "i",&i)){printf("FAILED setNCPU!\n"); fflush(stdout);  PyObject *ret = Py_BuildValue("i",-1); return ret;};


  //  printf("\n     setNCPU %i\n\n",i);


    printf("Preparing memory for %i workers\n",i);

    for(j=0; j<NCPU; j++){
      delete[] worker[j].t0; 
      delete[] worker[j].t1;
    };

    if(NCPU>0){
       delete[] worker;
       delete[] mod.Chi2;
       delete[] mod.WorkHess;
       delete[] mod.WorkGrad;
    };

    worker = new SHARED_DATA[i];
    for(k=0; k<i; k++){
      worker[k].t0 = new int[Nspw];
      worker[k].t1 = new int[Nspw];
      worker[k].Iam = k;
    };


    for (j=0; j<Nspw; j++) {
         int nperproc = (int)((double)vis.nt[j])/((double)i);
         for (k=0; k<i; k++){
            k0 = nperproc*k;
            if (k==i-1){k1=vis.nt[j];}else{k1 = nperproc*(k+1);};
            worker[k].t0[j] = k0;
            worker[k].t1[j] = k1;
         };
    };


    mod.WorkHess = new double*[i];
    mod.WorkGrad = new double*[i];
    mod.Chi2 = new double[i];

    NCPU = i;


    PyObject *ret = Py_BuildValue("i",0);
    return ret;
}
























//////////////////////////////////
// Fill-in the DATA arrays and master SHARED_DATA object
// USAGE FROM PYTHON: setData(IF, arrlist) where arrlist is list of data arrays.
//                    and IF is the IF number (setNspw must be run first!)
static PyObject *setData(PyObject *self, PyObject *args)
{

    PyObject *pu, *pv, *pw, *pwgt, *preal, *poreal, *tArr, *tIdx;
    PyObject *pfreqs, *pfittable, *ant1l, *ant2l;
    PyObject *pwgtcorr, *dtime, *RAoffset, *Decoffset, *Stretchoff, *iG;
    int IF; 



    if (!PyArg_ParseTuple(args, "iOOOOOOOOOOOOOOOOOOi",&IF,&pu,&pv,&pw, &pwgt,&preal,&poreal,&pfreqs,&pfittable,&pwgtcorr, &dtime, &tArr, &tIdx, &RAoffset, &Decoffset, &Stretchoff, &ant1l, &ant2l, &iG, &Nants)){printf("FAILED setData!\n"); fflush(stdout);  PyObject *ret = Py_BuildValue("i",-1); return ret;};


    /* Interprete the input objects as numpy arrays. */


    vis.ants[0][IF] = (int *)PyArray_DATA(ant1l);
    vis.ants[1][IF] = (int *)PyArray_DATA(ant2l);


    vis.dtArray[IF] = (double *)PyArray_DATA(tArr);
    vis.dtIndex[IF] = (int *)PyArray_DATA(tIdx);

    vis.uv[0][IF] = (double *)PyArray_DATA(pu);
    vis.uv[1][IF] = (double *)PyArray_DATA(pv);
    vis.uv[2][IF] = (double *)PyArray_DATA(pw);

    vis.wgt[0][IF] = (double *)PyArray_DATA(pwgt);
    vis.ObsVis[IF] = (cplx64 *)PyArray_DATA(preal);

    mod.ModVis[IF] = (cplx64 *)PyArray_DATA(poreal);

    vis.freqs[IF] = (double *)PyArray_DATA(pfreqs);

    vis.fittable[IF] = (int *)PyArray_DATA(pfittable);
    vis.wgt[1][IF] = (double *)PyArray_DATA(pwgtcorr);

    vis.isGain[IF] = (int *)PyArray_DATA(iG);


    vis.dt[IF] = (double *)PyArray_DATA(dtime);

    vis.RAshift[IF] = (double *)PyArray_DATA(RAoffset);

    vis.Decshift[IF] = (double *)PyArray_DATA(Decoffset);

    vis.Stretch[IF] = (double *)PyArray_DATA(Stretchoff);

    vis.nt[IF] = PyArray_DIM(preal,0);
    vis.nnu[IF] = PyArray_DIM(preal,1);

    Nbas = Nants*(Nants-1)/2;


  //  if(vis.nnu[IF]>maxnchan){maxnchan=vis.nnu[IF];};

    PyObject *ret = Py_BuildValue("i",0);
    return ret;

}












//////////////////////////////////
// Fill-in the MODEL arrays
// USAGE FROM PYTHON: setModel(arrlist) where arrlist is list of data arrays.
PyObject *setModel(PyObject *self, PyObject *args) {

    PyObject *HessArr, *GradArr, *modArr, *VarArr, *FixArr, *tempArr, *dparArr, *propRA, *propDec, *refpos, *parDep, *aG, *triSpec;

    int i, j, k, l, IF,isFixed, isMixed,isTimeInt;

  //  PyObject *ret = Py_BuildValue("i",0);


    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOiiiddO",&modArr,&HessArr,&GradArr,&VarArr,&FixArr,&dparArr, &propRA, &propDec, &refpos, &parDep, &aG,&isFixed, &isMixed, &isTimeInt, &phClosWgt, &ampClosWgt, &triSpec)){printf("FAILED setModel!\n"); fflush(stdout); PyObject *ret = Py_BuildValue("i",-1); return ret;};

  //  printf("\n     setModel \n\n");

  //  delete[] Hessian;
  //  delete[] models;
  //  delete[] Gradient;
  //  delete[] dpar;
  //  delete[] muRA;
  //  delete[] muDec;

    clearModel();
    isModel = true;

    compFixed = isFixed==1;
    MixedG = isMixed==1;
    isTime = isTimeInt==1;
    doAmpClos = ampClosWgt>0.0;
    doPhClos = phClosWgt>0.0;

    if(NCPU>1 && (doAmpClos || doPhClos)){printf("FAILED Closures cannot be used if NCPU>1!\n"); fflush(stdout); PyObject *ret = Py_BuildValue("i",-2); return ret;};


    mod.models = (int *)PyArray_DATA(modArr);
    mod.Hessian = (double *)PyArray_DATA(HessArr);
    mod.Gradient = (double *)PyArray_DATA(GradArr);
    mod.dpar = (double *)PyArray_DATA(dparArr);
    mod.muRA = (double *)PyArray_DATA(propRA);
    mod.muDec = (double *)PyArray_DATA(propDec);
    vis.phaseCenter = (double *)PyArray_DATA(refpos);
    mod.triSpec = (int *)PyArray_DATA(triSpec);

    cosDecRef = cos(vis.phaseCenter[1]);
    sinDecRef = sin(vis.phaseCenter[1]);



    ncomp = PyArray_DIM(modArr,0);
    npar = PyArray_DIM(GradArr,0);
    Nants = (int) PyList_Size(parDep);
    Nbas = Nants*(Nants-1)/2;





    mod.closBuffer = new cplx64[Nbas];
    mod.closBufferWgt = new double[Nbas];
    mod.closBufferAbs = new double[Nbas];
    mod.closBufferLog = new double[Nbas];

    mod.closBufferMod = new cplx64 *[(npar+1)];
    mod.closBufferAbsMod = new double *[(npar+1)];
    mod.closBufferLogMod = new double *[(npar+1)];

    for(i=0;i<npar+1;i++){
       mod.closBufferMod[i] = new cplx64[Nbas];
       mod.closBufferAbsMod[i] = new double[Nbas];
       mod.closBufferLogMod[i] = new double[Nbas];
    };


    for(j=0;j<Nbas;j++){
      mod.closBuffer[j] = 0.0;
      mod.closBufferWgt[j] = 0.0;
      mod.closBufferAbs[j] = 0.0;
      mod.closBufferLog[j] = 0.0;
      for(i=0;i<npar;i++){
        mod.closBufferMod[i][j] = 0.0;
        mod.closBufferAbsMod[i][j] = 0.0;
        mod.closBufferLogMod[i][j] = 0.0;
      };
    };



    if(doPhClos){
      NphClos = Nants*(Nants-1)*(Nants-2)/6;}
    else {
      NphClos = 0;
    };

    if(doAmpClos){
      NampClos = Nants*(Nants-1)*(Nants-2)*(Nants-3)/24;}
    else {
      NampClos = 0;
    };

    mod.BasIdx = new int*[2];
    mod.phaseClosIdx = new int*[3];
    mod.ampClosIdx = new int*[4];
    mod.BasIdx[0] = new int[Nbas];
    mod.BasIdx[1] = new int[Nbas];
    mod.phaseClosIdx[0] = new int[NphClos];
    mod.phaseClosIdx[1] = new int[NphClos];
    mod.phaseClosIdx[2] = new int[NphClos];
    mod.ampClosIdx[0] = new int[NampClos];
    mod.ampClosIdx[1] = new int[NampClos];
    mod.ampClosIdx[2] = new int[NampClos];
    mod.ampClosIdx[3] = new int[NampClos];

    int cIdx1,cIdx2,cIdx3,cIdx4,aux;

    cIdx1 = 0;
    for(i=0; i<Nants-1; i++){
      for(j=i+1;j<Nants; j++){
        mod.BasIdx[0][cIdx1] = i;
        mod.BasIdx[1][cIdx1] = j;
        cIdx1 += 1;
      };
    };


  if(NphClos>0){
    aux=0;
    for(i=0; i<Nants-2; i++){
      for(j=i+1;j<Nants-1; j++){
        for(cIdx1=0; cIdx1<Nbas; cIdx1++){if(mod.BasIdx[0][cIdx1]==i && mod.BasIdx[1][cIdx1]==j){break;};};
        for(k=j+1;k<Nants; k++){
          for(cIdx2=0; cIdx2<Nbas; cIdx2++){if(mod.BasIdx[0][cIdx2]==j && mod.BasIdx[1][cIdx2]==k){break;};};
          for(cIdx3=0; cIdx3<Nbas; cIdx3++){if(mod.BasIdx[0][cIdx3]==i && mod.BasIdx[1][cIdx3]==k){break;};};
          mod.phaseClosIdx[0][aux] = cIdx1;
          mod.phaseClosIdx[1][aux] = cIdx2;
          mod.phaseClosIdx[2][aux] = cIdx3;
          aux += 1;
        };
      };
    };
   };


  if(NampClos>0){
    aux=0;
    for(i=0; i<Nants-3; i++){
      for(j=i+1;j<Nants-2; j++){
        for(cIdx1=0; cIdx1<Nbas; cIdx1++){if(mod.BasIdx[0][cIdx1]==i && mod.BasIdx[1][cIdx1]==j){break;};};
        for(k=j+1;k<Nants-1; k++){
          for(cIdx3=0; cIdx3<Nbas; cIdx3++){if(mod.BasIdx[0][cIdx3]==j && mod.BasIdx[1][cIdx3]==k){break;};};
          for(l=k+1;l<Nants; l++){
             for(cIdx2=0; cIdx2<Nbas; cIdx2++){if(mod.BasIdx[0][cIdx2]==k && mod.BasIdx[1][cIdx2]==l){break;};};
             for(cIdx4=0; cIdx4<Nbas; cIdx4++){if(mod.BasIdx[0][cIdx4]==i && mod.BasIdx[1][cIdx4]==l){break;};};
             mod.ampClosIdx[0][aux] = cIdx1;
             mod.ampClosIdx[1][aux] = cIdx2;
             mod.ampClosIdx[2][aux] = cIdx3;
             mod.ampClosIdx[3][aux] = cIdx4;
             aux += 1;
        };
      };
    };
   };
  };











    mod.nparAnt = new int[Nants];
    mod.parAnt = new int*[Nants];

    for (i=0;i<Nants;i++){
      mod.nparAnt[i] = PyArray_DIM(PyList_GetItem(parDep,i),0);
      mod.parAnt[i] = (int *)PyArray_DATA(PyList_GetItem(parDep,i));
    };



    for (IF=0; IF<Nspw; IF++){

      mod.Gain[IF] = new cplx64**[Nants];

      for (j=0; j<Nants; j++) {

        mod.Gain[IF][j] = new cplx64*[mod.nparAnt[j]];
        for (i=0;i<mod.nparAnt[j];i++){
          tempArr = PyList_GetItem(PyList_GetItem(PyList_GetItem(aG,IF),j),i);
          mod.Gain[IF][j][i] = (cplx64 *)PyArray_DATA(tempArr);
        };
      };

    };






    HankelOrder = PyArray_DIM(PyList_GetItem(VarArr,0),1)-NparMax;
    maxnchan = PyArray_DIM(PyList_GetItem(VarArr,0),2);
//    delete[] mod.vars;
//    delete[] mod.fixp;

    mod.vars = new double*[(npar+1)];
    mod.fixp = new double*[(npar+1)];   


    for (i=0; i<(npar+1); i++) {
      tempArr = PyList_GetItem(VarArr,i);
      mod.vars[i] = (double *)PyArray_DATA(tempArr);
    };


    for (i=0; i<(npar+1); i++) {
      tempArr = PyList_GetItem(FixArr,i);
      mod.fixp[i] = (double *)PyArray_DATA(tempArr);
    };




    PyObject *ret = Py_BuildValue("i",0);
    return ret;



};
















//////////////////////////////////
// Allocate memory for the workers.
// USAGE FROM PYTHON: setWork() with no arguments.
//                    (setNCPU and setModel must be run first!)
static PyObject *setWork(PyObject *self, PyObject *args)
{
  int i;
    for (i=0; i<NCPU; i++) {
      mod.WorkHess[i] = new double[npar*npar];
      mod.WorkGrad[i] = new double[npar];
    };


 //   printf("\n     setWork %i\n\n",NCPU);


    PyObject *ret = Py_BuildValue("i",0);
    return ret;

};












/////////////////////////////////
// Deallocate the memory allocated with setWork.
// USAGE FROM PYTHON: unsetWork() with no arguments.
//                    (obviously, setWork must be run first!)
static PyObject *unsetWork(PyObject *self, PyObject *args)
{
  int i;
  printf("\n UNSET WORK! %i\n",NCPU);
  for(i=0;i<NCPU;i++){
    delete mod.WorkHess[i];
    delete mod.WorkGrad[i];  
  };

  //  delete WorkHess;
  //  delete WorkGrad;

    PyObject *ret = Py_BuildValue("i",0);
    return ret;

};















//////////////////////////////////////////////////
/* Main Python function. It spreads the work through the workers */
// USAGE FROM PYTHON: modelcomp(IF, nui, opts) (see Python code for info).
static PyObject *modelcomp(PyObject *self, PyObject *args)
{
    void *status;
    double totChi=0.0;
    int i,j;
    int nparsq = npar*npar;

    int auxClos, auxClos2,Dump;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "iiiiii",&cIF,&nui,&mode, &auxClos,&auxClos2,&Dump)){printf("FAILED modelcomp!\n"); fflush(stdout);  PyObject *ret = Py_BuildValue("i",-1); return ret;};


    doClos = auxClos == 1;
    onlyClos = auxClos2 == 1;
    doDump = Dump == 1;

// Zero the workers memory:
    for (i=0; i<NCPU; i++) {
      for(j=0;j<nparsq;j++){mod.WorkHess[i][j] = 0.0;};
      for(j=0;j<npar;j++){mod.WorkGrad[i][j] = 0.0;};
    };



if (NCPU>1) {

  /* Code for the case NCPU>1. 
     Define the workers and perform the parallel task. */

  pthread_t MyThreads[NCPU];
  pthread_attr_t attr;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


  for (i=0; i<NCPU; i++){
    pthread_create(&MyThreads[i], &attr, writemod, (void *)&worker[i]);
   };
  pthread_attr_destroy(&attr);

  for(i=0; i<NCPU; i++){
    pthread_join(MyThreads[i], &status);
  };



} else {

/* Case of one single process (NCPU=1). 
   Now, the master will do all the work. */

master.t0[cIF] = 0;
master.t1[cIF] = vis.nt[cIF];
master.Iam = -1;

writemod((void *)&master);

};


  /* Add-up the Chi square, the error vector, and the  Hessian */
    for (i=0 ; i<NCPU; i++) {
      totChi += mod.Chi2[i];
      for (j=0;j<npar;j++){
        mod.Gradient[j] += mod.WorkGrad[i][j];
      };
      for(j=0;j<nparsq;j++){
        mod.Hessian[j] += mod.WorkHess[i][j];
      };
    };




/* Update references and set the return value */

PyObject *ret = Py_BuildValue("d", totChi);

return ret;

}
















static PyObject *QuinnFF(PyObject *self, PyObject *args){

  int IFFit, refant, doModel, doGlobal;
  int ErrStat = 0; // To track errors in GFF (not implemented)

  PyObject *ret, *gainList;

    if (!PyArg_ParseTuple(args, "iiiiO",&IFFit, &refant, &doModel, &doGlobal, &gainList)){printf("FAILED QuinnFringe!\n"); fflush(stdout);  ret = Py_BuildValue("i",-1); return ret;};


#if QUINN_FITTER == 0

if (IFFit >= Nspw){ret = Py_BuildValue("i", -1);printf("\n spw is too high!"); return ret;};

QuinnFringe *FringeFit = new QuinnFringe(Nants,vis.nt[IFFit],vis.nnu[IFFit],vis.ObsVis[IFFit],mod.ModVis[IFFit],vis.ants[0][IFFit], vis.ants[1][IFFit],vis.dt[IFFit],vis.fittable[IFFit],vis.freqs[IFFit],vis.wgt[0][IFFit]); 

int result = FringeFit->GFF(refant, doGlobal, doModel);

if (result != 0){ret = Py_BuildValue("i",result); return ret;};

// Return the gains:

//double *Rates = new double[Nants]; 
//double *Delays = new double[Nants];
//double *Phases = new double[Nants];

double *Delays = (double *)PyArray_DATA(PyList_GetItem(gainList,0));
double *Rates = (double *)PyArray_DATA(PyList_GetItem(gainList,1));
double *Phases = (double *)PyArray_DATA(PyList_GetItem(gainList,2));
double *Bins = new double[2];

ErrStat = FringeFit->getRates(Rates);
ErrStat = FringeFit->getDelays(Delays);
ErrStat = FringeFit->getPhases(Phases);
ErrStat = FringeFit->getBins(Bins);

//PyObject *PyRate, *PyDelay, *PyPhase;

npy_intp Dims[1];
//int *Dims = new int[1];
Dims[0] = Nants;

//printf("\nNants: %i  DIMS:  %i/%.3e/%.3e \n",Nants,Dims[0], Bins[0],Bins[1]);

//PyRate = PyArray_SimpleNewFromData(1, Dims, NPY_FLOAT64, (void *) Rates);
//PyDelay = PyArray_SimpleNewFromData(1, Dims, NPY_FLOAT64, (void *) Delays);
//PyPhase = PyArray_SimpleNewFromData(1, Dims, NPY_FLOAT64, (void *) Phases);

//printf("\n Ant 1: %.4e %.4e %.4e\n", Rates[0], Delays[0], Phases[0]);

//ret = Py_BuildValue("[O,O,O,d,d]",PyDelay,PyRate,PyPhase,Bins[0],Bins[1]);
ret = Py_BuildValue("[d,d]",Bins[0],Bins[1]);

delete FringeFit;

if (ErrStat != 0){ret = Py_BuildValue("i",ErrStat);};

#else

printf("\n QUINN FITTER NOT INSTALLED!");
ret = Py_BuildValue("i", -1);

#endif

return ret;


};

