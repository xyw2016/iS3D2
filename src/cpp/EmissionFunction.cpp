#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <complex>
#include <array>
#include <ctime>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "iS3D.h"
#include "readindata.h"
#include "EmissionFunction.h"
#include "Arsenal.h"
#include "ParameterReader.h"
#include "DeltafData.h"
#include <gsl/gsl_sf_bessel.h> //for modified bessel functions
#include "GaussThermal.h"
#include "SampledParticle.h"

using namespace std;



double compute_detA(Shear_Stress pimunu, double shear_mod, double bulk_mod)
{
  double pixx_LRF = pimunu.pixx_LRF;  double piyy_LRF = pimunu.piyy_LRF;
  double pixy_LRF = pimunu.pixy_LRF;  double piyz_LRF = pimunu.piyz_LRF;
  double pixz_LRF = pimunu.pixz_LRF;  double pizz_LRF = pimunu.pizz_LRF;

  double Axx = 1.0  +  pixx_LRF * shear_mod  +  bulk_mod;
  double Axy = pixy_LRF * shear_mod;
  double Axz = pixz_LRF * shear_mod;
  double Ayy = 1.0  +  piyy_LRF * shear_mod  +  bulk_mod;
  double Ayz = piyz_LRF * shear_mod;
  double Azz = 1.0  +  pizz_LRF * shear_mod  +  bulk_mod;

  // assume Aij is symmetric (need to change this formula if include diffusion)
  double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);

  return detA;
}

bool is_linear_pion0_density_negative(double T, double neq_pion0, double J20_pion0, double bulkPi, double F, double betabulk)
{
  // determine if linear pion0 density goes negative

  double dn_pion0 = bulkPi * (neq_pion0  +  J20_pion0 * F / T / T) / betabulk;

  double nlinear_pion0 = neq_pion0 + dn_pion0;

  if(nlinear_pion0 < 0.0) return true;

  return false;
}

bool does_feqmod_breakdown(double mass_pion0, double T, double F, double bulkPi, double betabulk, double detA, double detA_min, double z, Gauss_Laguerre * laguerre, int df_mode, int fast, double Tavg, double F_avg, double betabulk_avg)
{
  if(df_mode == 3)
  {
    // use the average temperature, df coefficents instead
    if(fast)
    {
      T = Tavg;
      F = F_avg;
      betabulk = betabulk_avg;
    }
    const int laguerre_pts = laguerre->points;
    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];
    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    // calculate linearized pion density
    double mbar_pion0 = mass_pion0 / T;

    double neq_fact = T * T * T / two_pi2_hbarC3;
    double J20_fact = T * neq_fact;

    double neq_pion0 = neq_fact * GaussThermal(neq_int, pbar_root1, pbar_weight1, laguerre_pts, mbar_pion0, 0., 0., -1.);
    double J20_pion0 = J20_fact * GaussThermal(J20_int, pbar_root2, pbar_weight2, laguerre_pts, mbar_pion0, 0., 0., -1.);

    bool pion_density_negative = is_linear_pion0_density_negative(T, neq_pion0, J20_pion0, bulkPi, F, betabulk);

    if(detA <= detA_min || pion_density_negative)
    {
      return true;
    }
  }
  else if(df_mode == 4)
  {
    //if(z < 0.0) printf("Error: z should be positive");

    if(detA <= detA_min || z < 0.0)
    {
      return true;
    }
  }

  return false;
}



// Class EmissionFunctionArray ------------------------------------------
EmissionFunctionArray::EmissionFunctionArray(ParameterReader* paraRdr_in, Table* chosen_particles_in, Table* pT_tab_in,
  Table* phi_tab_in, Table* y_tab_in, Table* eta_tab_in, particle_info* particles_in,
  int Nparticles_in, FO_surf* surf_ptr_in, long FO_length_in, Deltaf_Data * df_data_in)
  {
    // momentum and spacetime rapdity tables
    pT_tab = pT_tab_in;
    phi_tab = phi_tab_in;
    y_tab = y_tab_in;
    eta_tab = eta_tab_in;

    pT_tab_length = pT_tab->getNumberOfRows();
    phi_tab_length = phi_tab->getNumberOfRows();
    y_tab_length = y_tab->getNumberOfRows();
    eta_tab_length = eta_tab->getNumberOfRows();


    // omp parameters
    CORES = 1;

  #ifdef _OPENMP
    CORES = omp_get_max_threads();
  #endif


    // control parameters
    paraRdr = paraRdr_in;
    OPERATION = paraRdr->getVal("operation");
    MODE = paraRdr->getVal("mode");


    DIMENSION = paraRdr->getVal("dimension");

    if(DIMENSION == 2)
    {
      y_tab_length = 1;
    }
    else if(DIMENSION == 3)
    {
      eta_tab_length = 1;
    }
    else
    {
      printf("EmissionFunctionArray error: need to set dimension = (2,3)\n");
      exit(-1);
    }


    DF_MODE = paraRdr->getVal("df_mode");

    if(DF_MODE == 1)
    {
      df_correction = "Grad 14-moment approximation";
    }
    else if(DF_MODE == 2)
    {
      df_correction = "RTA Chapman-Enskog expansion";
    }
    else if(DF_MODE == 3)
    {
      df_correction = "PTM modified equilibrium distribution";
    }
    else if(DF_MODE == 4)
    {
      df_correction = "PTB modified equilibrium distribution";
    }
    else if(DF_MODE == 5)
    {
      df_correction = "PTM modified anisotropic distribution";
    }
    else
    {
      printf("EmissionFunctionArray error: need to set df_mode = (1,2,3,4,5)\n");
      exit(-1);
    }

    printf("Running particlization with %s...\n\n", df_correction.c_str());


    INCLUDE_BARYON = paraRdr->getVal("include_baryon");
    INCLUDE_BULK_DELTAF = paraRdr->getVal("include_bulk_deltaf");
    INCLUDE_SHEAR_DELTAF = paraRdr->getVal("include_shear_deltaf");
    INCLUDE_BARYONDIFF_DELTAF = paraRdr->getVal("include_baryondiff_deltaf");

    REGULATE_DELTAF = paraRdr->getVal("regulate_deltaf");
    OUTFLOW = paraRdr->getVal("outflow");

    DETA_MIN = paraRdr->getVal("deta_min");
    GROUP_PARTICLES = paraRdr->getVal("group_particles");
    PARTICLE_DIFF_TOLERANCE = paraRdr->getVal("particle_diff_tolerance");

    MASS_PION0 = paraRdr->getVal("mass_pion0");

    LIGHTEST_PARTICLE = paraRdr->getVal("lightest_particle");
    DO_RESONANCE_DECAYS = paraRdr->getVal("do_resonance_decays");

    OVERSAMPLE = paraRdr->getVal("oversample");
    FAST = paraRdr->getVal("fast");
    MIN_NUM_HADRONS = paraRdr->getVal("min_num_hadrons");
    MAX_NUM_SAMPLES = paraRdr->getVal("max_num_samples");
    SAMPLER_SEED = paraRdr->getVal("sampler_seed");

    if(OPERATION == 2)
    {
      printf("Sampler seed set to %ld \n", SAMPLER_SEED);
    }


    // parameters for sampler test
    //::::::::::::::::::::::::::::::::::::::::::::::::::::
    TEST_SAMPLER = paraRdr->getVal("test_sampler");

    PT_MIN = paraRdr->getVal("pT_min");
    PT_MAX = paraRdr->getVal("pT_max");
    PT_BINS = paraRdr->getVal("pT_bins");
    PT_WIDTH = (PT_MAX - PT_MIN) / (double)PT_BINS;

    Y_CUT = paraRdr->getVal("y_cut");
    Y_BINS = paraRdr->getVal("y_bins");
    Y_WIDTH = 2.0 * Y_CUT / (double)Y_BINS;

    PHIP_BINS = paraRdr->getVal("phip_bins");
    PHIP_WIDTH = two_pi / (double)PHIP_BINS;

    ETA_CUT = paraRdr->getVal("eta_cut");
    ETA_BINS = paraRdr->getVal("eta_bins");
    ETA_WIDTH = 2.0 * ETA_CUT / (double)ETA_BINS;

    TAU_MIN = paraRdr->getVal("tau_min");
    TAU_MAX = paraRdr->getVal("tau_max");
    TAU_BINS = paraRdr->getVal("tau_bins");
    TAU_WIDTH = (TAU_MAX - TAU_MIN) / (double)TAU_BINS;

    R_MIN = paraRdr->getVal("r_min");
    R_MAX = paraRdr->getVal("r_max");
    R_BINS = paraRdr->getVal("r_bins");
    R_WIDTH = (R_MAX - R_MIN) / (double)R_BINS;
    //::::::::::::::::::::::::::::::::::::::::::::::::::::

    particles = particles_in;
    Nparticles = Nparticles_in;
    surf_ptr = surf_ptr_in;
    FO_length = FO_length_in;
    df_data = df_data_in;
    number_of_chosen_particles = chosen_particles_in->getNumberOfRows();

    // allocate memory for sampled distributions / spectra (for sampler testing)
    dN_dy_count = (double**)calloc(number_of_chosen_particles, sizeof(double));
    dN_deta_count = (double**)calloc(number_of_chosen_particles, sizeof(double));
    dN_dphipdy_count = (double***)calloc(number_of_chosen_particles, sizeof(double));
    dN_2pipTdpTdy_count = (double***)calloc(number_of_chosen_particles, sizeof(double));

    pT_count = (double**)calloc(number_of_chosen_particles, sizeof(double));
    vn_real_count = (double****)calloc(K_MAX, sizeof(double));
    vn_imag_count = (double****)calloc(K_MAX, sizeof(double));

    dN_taudtaudy_count = (double**)calloc(number_of_chosen_particles, sizeof(double));
    dN_twopirdrdy_count = (double**)calloc(number_of_chosen_particles, sizeof(double));
    dN_dphisdy_count = (double**)calloc(number_of_chosen_particles, sizeof(double));

    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      dN_dy_count[ipart] = (double*)calloc(Y_BINS, sizeof(double));
      dN_deta_count[ipart] = (double*)calloc(ETA_BINS, sizeof(double));
      
      pT_count[ipart] = (double*)calloc(PT_BINS, sizeof(double));

      dN_taudtaudy_count[ipart] = (double*)calloc(TAU_BINS, sizeof(double));
      dN_twopirdrdy_count[ipart] = (double*)calloc(R_BINS, sizeof(double));
      dN_dphisdy_count[ipart] = (double*)calloc(PHIP_BINS, sizeof(double));

      dN_2pipTdpTdy_count[ipart] = (double**)calloc(Y_BINS, sizeof(double));
      dN_dphipdy_count[ipart] = (double**)calloc(Y_BINS, sizeof(double));

      for(int i = 0; i < Y_BINS; i++)
      { 
        dN_2pipTdpTdy_count[ipart][i] = (double*)calloc(PT_BINS, sizeof(double));
        dN_dphipdy_count[ipart][i] = (double*)calloc(PHIP_BINS, sizeof(double));
      }
    }

    for(int k = 0; k < K_MAX; k++)
    {
      vn_real_count[k] = (double***)calloc(number_of_chosen_particles, sizeof(double));
      vn_imag_count[k] = (double***)calloc(number_of_chosen_particles, sizeof(double));

      for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
      {
        vn_real_count[k][ipart] = (double**)calloc(Y_BINS, sizeof(double));
        vn_imag_count[k][ipart] = (double**)calloc(Y_BINS, sizeof(double));

        for(int i = 0; i < Y_BINS; i++)
        {
          vn_real_count[k][ipart][i] = (double*)calloc(PT_BINS, sizeof(double));
          vn_imag_count[k][ipart][i] = (double*)calloc(PT_BINS, sizeof(double));
        }
      }
    }


    chosen_particles_01_table = new int[Nparticles];

    //a class member to hold 3D smooth CF spectra for all chosen particles
    dN_pTdpTdphidy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
    // holds smooth CF spectra of a given parent resonance
    logdN_PTdPTdPhidY = new double [pT_tab_length * phi_tab_length * y_tab_length];

    //zero the array
    for (int iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; iSpectra++)
    {
      dN_pTdpTdphidy[iSpectra] = 0.0;
    }
    for(int iS_parent = 0; iS_parent < pT_tab_length * phi_tab_length * y_tab_length; iS_parent++)
    {
      logdN_PTdPTdPhidY[iS_parent] = 0.0; // is it harmful to have a y_tab_length =/= 1 if DIMENSION = 2 (waste of memory?)
    }

    if (MODE == 5)
    {
      //class member to hold polarization vector of chosen particles
      St = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      Sx = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      Sy = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      Sn = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];
      //holds the normalization of the polarization vector of chosen particles
      Snorm = new double [number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length];

      for (int iSpectra = 0; iSpectra < number_of_chosen_particles * pT_tab_length * phi_tab_length * y_tab_length; iSpectra++)
      {
        St[iSpectra] = 0.0;
        Sx[iSpectra] = 0.0;
        Sy[iSpectra] = 0.0;
        Sn[iSpectra] = 0.0;
        Snorm[iSpectra] = 0.0;
      }
    } // if (MODE == 5)



    // how much of this do we still need?


    for (int n = 0; n < Nparticles; n++) chosen_particles_01_table[n] = 0;

    //only grab chosen particles from the table
    for (int m = 0; m < number_of_chosen_particles; m++)
    { //loop over all chosen particles
      int mc_id = chosen_particles_in->get(1, m + 1);

      for (int n = 0; n < Nparticles; n++)
      {
        if (particles[n].mc_id == mc_id)
        {
          chosen_particles_01_table[n] = 1;
          break;
        }
      }
    } // for (int m = 0; m < number_of_chosen_particles; m++)

    // next, for sampling processes
    chosen_particles_sampling_table = new int[number_of_chosen_particles];
    // first copy the chosen_particles table, but now using indices instead of mc_id
    int current_idx = 0;
    for (int m = 0; m < number_of_chosen_particles; m++)
    {
      int mc_id = chosen_particles_in->get(1, m + 1);
      for (int n = 0; n < Nparticles; n++)
      {
        if (particles[n].mc_id == mc_id)
        {
          chosen_particles_sampling_table[current_idx] = n;
          current_idx ++;
          break;
        }
      }
    } //for (int m = 0; m < number_of_chosen_particles; m++)

    // next re-order them so that particles with similar mass are adjacent
    if (GROUP_PARTICLES == 1) // sort particles according to their mass; bubble-sorting
    {
      for (int m = 0; m < number_of_chosen_particles; m++)
      {
        for (int n = 0; n < number_of_chosen_particles - m - 1; n++)
        {
          if (particles[chosen_particles_sampling_table[n]].mass > particles[chosen_particles_sampling_table[n + 1]].mass)
          {
            // swap them
            int particle_idx = chosen_particles_sampling_table[n + 1];
            chosen_particles_sampling_table[n + 1] = chosen_particles_sampling_table[n];
            chosen_particles_sampling_table[n] = particle_idx;
          }
        } // for (int n = 0; n < number_of_chosen_particles - m - 1; n++)
      } // for (int m = 0; m < number_of_chosen_particles; m++)
    } // if (GROUP_PARTICLES == 1)
  } // EmissionFunctionArray::EmissionFunctionArray

  EmissionFunctionArray::~EmissionFunctionArray()
  {
    delete[] chosen_particles_01_table;
    delete[] chosen_particles_sampling_table;
    delete[] dN_pTdpTdphidy; //for holding 3d spectra of all chosen particles
    delete[] logdN_PTdPTdPhidY;
  }



  //*********************************************************************************************
  void EmissionFunctionArray::calculate_spectra(std::vector<std::vector<Sampled_Particle>> &particle_event_list_in)
  {

#ifdef OPENMP
    double t1 = omp_get_wtime();
#else
    // Stopwatch sw;
    // sw.tic();
    clock_t start = clock();

#endif

    printf("Allocating memory for individual arrays to hold particle and freezeout surface info\n");


    // particle info of chosen particles
    double *Mass = (double*)calloc(number_of_chosen_particles, sizeof(double));
    double *Sign = (double*)calloc(number_of_chosen_particles, sizeof(double));
    double *Degeneracy = (double*)calloc(number_of_chosen_particles, sizeof(double));
    double *Baryon = (double*)calloc(number_of_chosen_particles, sizeof(double));
    int *MCID = (int*)calloc(number_of_chosen_particles, sizeof(int));

    double *Equilibrium_Density = (double*)calloc(number_of_chosen_particles, sizeof(double));
    double *Bulk_Density = (double*)calloc(number_of_chosen_particles, sizeof(double));
    double *Diffusion_Density = (double*)calloc(number_of_chosen_particles, sizeof(double));

    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      int chosen_index = chosen_particles_sampling_table[ipart];  // chosen particle's PDG index

      Mass[ipart] = particles[chosen_index].mass;                 // mass of chosen particles
      Sign[ipart] = particles[chosen_index].sign;                 // quantum statistics sign
      Degeneracy[ipart] = particles[chosen_index].gspin;          // spin degeneracy factor
      Baryon[ipart] = particles[chosen_index].baryon;             // baryon number
      MCID[ipart] = particles[chosen_index].mc_id;                // Monte-Carlo ID

      Equilibrium_Density[ipart] = particles[chosen_index].equilibrium_density; // neq
      Bulk_Density[ipart] = particles[chosen_index].bulk_density;               // dn_bulk (omitted Pi * u.d\sigma)
      Diffusion_Density[ipart] = particles[chosen_index].diff_density;          // dn_diff (omitted V.d\sigma)
    }


    // particle info of entire PDG table (remember to skip photons in calculation)
    double *Mass_PDG = (double*)calloc(Nparticles, sizeof(double));
    double *Sign_PDG = (double*)calloc(Nparticles, sizeof(double));
    double *Degeneracy_PDG = (double*)calloc(Nparticles, sizeof(double));
    double *Baryon_PDG = (double*)calloc(Nparticles, sizeof(double));

    for(int ipart = 0; ipart < Nparticles; ipart++)
    {
      Mass_PDG[ipart] = particles[ipart].mass;
      Sign_PDG[ipart] = particles[ipart].sign;
      Degeneracy_PDG[ipart] = particles[ipart].gspin;
      Baryon_PDG[ipart] = particles[ipart].baryon;
    }


    Gauss_Laguerre * gla = new Gauss_Laguerre;  // load gauss laguerre/legendre roots and weights
    Gauss_Legendre * legendre = new Gauss_Legendre;
    gla->load_roots_and_weights("tables/gauss/gla_roots_weights.txt");
    legendre->load_roots_and_weights("tables/gauss/gauss_legendre.dat");


    Plasma * QGP = new Plasma;
    QGP->load_thermodynamic_averages();         // load averaged thermodynamic variables


    // freezeout surface info
    double *tau = (double*)calloc(FO_length, sizeof(double));
    double *x = (double*)calloc(FO_length, sizeof(double));
    double *y = (double*)calloc(FO_length, sizeof(double));
    double *eta = (double*)calloc(FO_length, sizeof(double));

    double *dat = (double*)calloc(FO_length, sizeof(double));
    double *dax = (double*)calloc(FO_length, sizeof(double));
    double *day = (double*)calloc(FO_length, sizeof(double));
    double *dan = (double*)calloc(FO_length, sizeof(double));

    double *ux = (double*)calloc(FO_length, sizeof(double));
    double *uy = (double*)calloc(FO_length, sizeof(double));
    double *un = (double*)calloc(FO_length, sizeof(double));

    double *E = (double*)calloc(FO_length, sizeof(double));
    double *T = (double*)calloc(FO_length, sizeof(double));
    double *P = (double*)calloc(FO_length, sizeof(double));

    double *pixx = (double*)calloc(FO_length, sizeof(double));
    double *pixy = (double*)calloc(FO_length, sizeof(double));
    double *pixn = (double*)calloc(FO_length, sizeof(double));
    double *piyy = (double*)calloc(FO_length, sizeof(double));
    double *piyn = (double*)calloc(FO_length, sizeof(double));

    double *bulkPi = (double*)calloc(FO_length, sizeof(double));


    // baryon chemical potential effects
    double *muB;                      // muB
    double *nB;                       // nB
    double *Vx;                       // V^x
    double *Vy;                       // V^y
    double *Vn;                       // V^\eta

    if(INCLUDE_BARYON)
    {
      muB = (double*)calloc(FO_length, sizeof(double));
      nB = (double*)calloc(FO_length, sizeof(double));
      if(INCLUDE_BARYONDIFF_DELTAF){
        Vx = (double*)calloc(FO_length, sizeof(double));
        Vy = (double*)calloc(FO_length, sizeof(double));
        Vn = (double*)calloc(FO_length, sizeof(double));
      }
    }


    // thermal vorticity tensor for polarization studies
    double *wtx;
    double *wty;
    double *wtn;
    double *wxy;
    double *wxn;
    double *wyn;

    if(MODE == 5)
    {
      wtx = (double*)calloc(FO_length, sizeof(double));
      wty = (double*)calloc(FO_length, sizeof(double));
      wtn = (double*)calloc(FO_length, sizeof(double));
      wxy = (double*)calloc(FO_length, sizeof(double));
      wxn = (double*)calloc(FO_length, sizeof(double));
      wyn = (double*)calloc(FO_length, sizeof(double));
    }


    for(long icell = 0; icell < FO_length; icell++)
    {
      FO_surf *surf = &surf_ptr[icell];     // get local freezeout surface info

      tau[icell] = surf->tau;
      x[icell] = surf->x;
      y[icell] = surf->y;
      eta[icell] = surf->eta;

      ux[icell] = surf->ux;
      uy[icell] = surf->uy;
      un[icell] = surf->un;

      dat[icell] = surf->dat;
      dax[icell] = surf->dax;
      day[icell] = surf->day;
      dan[icell] = surf->dan;

      E[icell] = surf->E;
      T[icell] = surf->T;
      P[icell] = surf->P;

      pixx[icell] = surf->pixx;
      pixy[icell] = surf->pixy;
      pixn[icell] = surf->pixn;
      piyy[icell] = surf->piyy;
      piyn[icell] = surf->piyn;

      bulkPi[icell] = surf->bulkPi;

      if(INCLUDE_BARYON)
      {
        muB[icell] = surf->muB;
        nB[icell] = surf->nB;
        if(INCLUDE_BARYONDIFF_DELTAF){
          Vx[icell] = surf->Vx;
          Vy[icell] = surf->Vy;
          Vn[icell] = surf->Vn;
        }
      }

      if(MODE == 5)
      {
        wtx[icell] = surf->wtx;
        wty[icell] = surf->wty;
        wtn[icell] = surf->wtn;
        wxy[icell] = surf->wxy;
        wxn[icell] = surf->wxn;
        wyn[icell] = surf->wyn;
      }
    }


    switch(OPERATION)
    {
      case 0:
      {
        printf("Computing particle spacetime distributions...\n");

        switch(DF_MODE)
        {
          case 1:
          case 2:
          {
            calculate_dN_dX(MCID, Mass, Sign, Degeneracy, Baryon, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data);
            break;
          }
          case 3:
          case 4:
          {
            calculate_dN_dX_feqmod(MCID, Mass, Sign, Degeneracy, Baryon, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, gla, df_data);
            break;
          }
          case 5:
          {
            printf("calculate_spectra error: no spacetime distribution routine for famod yet\n");
            exit(-1);
            break;
          }
          default:
          {
            printf("calculate_spectra error: need to set df_mode = (1, 2, 3, 4, 5)\n");
            exit(-1);
          }
        }
        break;
      }
      case 1:
      {
        printf("Computing continuous momentum spectra...\n");

        switch(DF_MODE)
        {
          case 1:
          case 2:
          {
            calculate_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, T, P, E, tau, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data);
            break;
          }
          case 3:
          case 4:
          {
            calculate_dN_pTdpTdphidy_feqmod(Mass, Sign, Degeneracy, Baryon, T, P, E, tau, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, gla, df_data);
            break;
          }
          case 5:
          {
            calculate_dN_pTdpTdphidy_famod(Mass, Sign, Degeneracy, Baryon, T, P, E, tau, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, Nparticles, Mass_PDG, Sign_PDG, Degeneracy_PDG, Baryon_PDG);
            break;
          }
          default:
          {
            printf("calculate_spectra error: need to set df_mode = (1, 2, 3, 4, 5)\n");
            exit(-1);
          }
        }
        printf("===================================================\n");
        printf("Writing results to files...\n\n");
        write_dN_pTdpTdphidy_toFile(MCID);   // write continuous particle momentum spectra to file
        write_continuous_vn_toFile(MCID);
        write_dN_twopipTdpTdy_toFile(MCID);
        write_dN_dphidy_toFile(MCID);
        write_dN_dy_toFile(MCID);

        break;
      }
      case 2:
      {
        if(OVERSAMPLE)
        {
          // estimate average particle yield
          double Ntotal = calculate_total_yield(Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla);

          Nevents = (long)min(ceil(MIN_NUM_HADRONS / Ntotal), MAX_NUM_SAMPLES);   // number of events to sample

          printf("Sampling %ld particlization events...\n", Nevents);
        }
        else
        {
          printf("Sampling 1 particlization event...\n");
        }



        particle_event_list.resize(Nevents);


        switch(DF_MODE)
        {
          case 1:
          case 2:
          case 3:
          case 4:
          {
            sample_dN_pTdpTdphidy(Mass, Sign, Degeneracy, Baryon, MCID, Equilibrium_Density, Bulk_Density, Diffusion_Density, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, df_data, gla, legendre);
            break;
          }
          case 5:
          {
            sample_dN_pTdpTdphidy_famod(Mass, Sign, Degeneracy, Baryon, MCID, T, P, E, tau, x, y, eta, ux, uy, un, dat, dax, day, dan, pixx, pixy, pixn, piyy, piyn, bulkPi, muB, nB, Vx, Vy, Vn, Nparticles, Mass_PDG, Sign_PDG, Degeneracy_PDG, Baryon_PDG);

            break;
          }
          default:
          {
            printf("calculate_spectra error: need to set df_mode = (1, 2, 3, 4, 5)\n");
            exit(-1);
          }
        }

        printf("===================================================\n");
        printf("Writing results to files...\n\n");

        write_particle_list_OSC();                      // write OSCAR particle list to file (if not using JETSCAPE)
        
        if(TEST_SAMPLER)
        {
          write_sampled_dN_dy_to_file_test(MCID);         // write particle distributions to file
          write_sampled_dN_deta_to_file_test(MCID);       // only for testing the particle sampler
          write_sampled_dN_2pipTdpTdy_to_file_test(MCID);
          write_sampled_dN_dphipdy_to_file_test(MCID);
          write_sampled_vn_to_file_test(MCID);
          write_sampled_dN_dX_to_file_test(MCID);
        }

        particle_event_list_in = particle_event_list;     // store particlization events

        break;
      }
      default:
      {
        printf("calculate_spectra error: need to set operation = (0, 1, 2)\n");
        exit(-1);
      }
    }


    if(MODE == 5)
    {
      printf("Computing spin polarization...\n");
      calculate_spin_polzn(Mass, Sign, Degeneracy, tau, eta, ux, uy, un, dat, dax, day, dan, wtx, wty, wtn, wxy, wxn, wyn, QGP);
      write_polzn_vector_toFile();
    }

    printf("===================================================\n");
    printf("Freeing memory...\n");

    free(Mass);
    free(Sign);
    free(Degeneracy);
    free(Baryon);
    free(MCID);

    free(Equilibrium_Density);
    free(Bulk_Density);
    free(Diffusion_Density);

    free(Mass_PDG);
    free(Sign_PDG);
    free(Degeneracy_PDG);
    free(Baryon_PDG);

    free(tau);
    free(x);
    free(y);
    free(eta);

    free(dat);
    free(dax);
    free(day);
    free(dan);

    free(ux);
    free(uy);
    free(un);

    free(E);
    free(T);
    free(P);

    free(pixx);
    free(pixy);
    free(pixn);
    free(piyy);
    free(piyn);

    free(bulkPi);

    if(INCLUDE_BARYON)
    {
      free(muB);
      free(nB);
      if(INCLUDE_BARYONDIFF_DELTAF){
        free(Vx);
        free(Vy);
        free(Vn);
      }
    }

    if(MODE == 5)
    {
      free(wtx);
      free(wty);
      free(wtn);
      free(wxy);
      free(wxn);
      free(wyn);
    }

  #ifdef OPENMP
    double t2 = omp_get_wtime();
    cout << "Used OpenMP. Spectra calculation took " << (t2 - t1) << " seconds, (" << (t2 - t1)/60. << " minutes)" << endl;
  #else
    // sw.toc();
    // cout << "\nSpectra calculation took " << sw.takeTime() << " seconds\n" << endl;
    double duration = (clock() - start) / (double)CLOCKS_PER_SEC;

    cout << "Spectra calculation took " << duration << " seconds, (" << duration/60. << " minutes)" << endl;

  #endif
  }







