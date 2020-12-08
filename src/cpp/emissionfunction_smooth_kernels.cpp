#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <array>
//#ifdef _OMP
//#include <omp.h>
//#endif
#include "iS3D.h"
#include "readindata.h"
#include "emissionfunction.h"
#include "Stopwatch.h"
#include "arsenal.h"
#include "Macros.h"
#include "ParameterReader.h"
#include "deltafReader.h"
#include <gsl/gsl_sf_bessel.h> //for modified bessel functions
#include <gsl/gsl_linalg.h>
#include "gaussThermal.h"
#include "particle.h"

using namespace std;

void EmissionFunctionArray::calculate_dN_pTdpTdphidy(double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data)
  {
    double prefactor = pow(2.0 * M_PI * hbarC, -3);   // prefactor of CFF

    long FO_chunk = FO_length / CORES;
    long remainder = FO_length  -  CORES * FO_chunk;

    cout << "Number of cores : " << CORES << endl;
    cout << "Chunk size = " << FO_chunk << endl;
    cout << "Remainder cells = " << remainder << endl;

    if(remainder != 0) FO_chunk++;

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];

    for(long iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab -> get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);
    }

    // pT array
    double pTValues[pT_tab_length];

    for(long ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab -> get(1, ipT + 1);
    }

    // y and eta arrays
    double yValues[y_tab_length];
    double etaValues[eta_tab_length];
    double etaWeights[eta_tab_length];

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(long ieta = 0; ieta < eta_tab_length; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;
      etaWeights[0] = 1.0;
      for(long iy = 0; iy < y_tab_length; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    //declare a huge array of size CORES * npart * pT_tab_length * phi_tab_length * y_tab_length to hold spectra for each chunk
    long npart = (long)number_of_chosen_particles;
    double *dN_pTdpTdphidy_all = (double*)calloc(CORES * npart * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

    // subdivide bite size chunks of freezeout surface across cores
    #pragma omp parallel for
    for(long n = 0; n < CORES; n++)
    {
      long endFO = FO_chunk;

      for(long icell = 0; icell < endFO; icell++)  // cell index inside each chunk
      {
      	if((icell == endFO - 1) && (remainder != 0) && (n > remainder - 1)) continue;

        long icell_glb = n  +  icell * CORES;

        double tau = tau_fo[icell_glb];         // longitudinal proper time
        double tau2 = tau * tau;
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
        }

        double dat = dat_fo[icell_glb];         // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];         // dan should be 0 for 2+1d

        double ux = ux_fo[icell_glb];           // contravariant fluid velocity
        double uy = uy_fo[icell_glb];           // enforce normalization
        double un = un_fo[icell_glb];
        double ux2 = ux * ux;                   // useful expressions
        double uy2 = uy * uy;
        double utperp = sqrt(1.0 + ux2 + uy2);
        double tau2_un = tau2 * un;
        double ut = sqrt(utperp * utperp  +  tau2_un * un);
        double ut2 = ut * ut;

        // skip cells with u.dsigma < 0
        if(ut * dat  +  ux * dax  +  uy * day  +  un * dan <= 0.0) continue;

        double T = T_fo[icell_glb];             // temperature (GeV)
        double P = P_fo[icell_glb];             // equilibrium pressure (GeV/fm^3)
        double E = E_fo[icell_glb];             // energy density (GeV/fm^3)

        double pitt = 0.0;                      // contravariant shear stress tensor pi^munu (GeV/fm^3)
        double pitx = 0.0;                      // enforce orthogonality pi.u = 0
        double pity = 0.0;                      // and tracelessness Tr(pi) = 0
        double pitn = 0.0;
        double pixx = 0.0;
        double pixy = 0.0;
        double pixn = 0.0;
        double piyy = 0.0;
        double piyn = 0.0;
        double pinn = 0.0;

        if(INCLUDE_SHEAR_DELTAF)
        {
          pixx = pixx_fo[icell_glb];
          pixy = pixy_fo[icell_glb];
          pixn = pixn_fo[icell_glb];
          piyy = piyy_fo[icell_glb];
          piyn = piyn_fo[icell_glb];
          pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2_un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
          pitn = (pixn * ux  +  piyn * uy  +  tau2_un * pinn) / ut;
          pity = (pixy * ux  +  piyy * uy  +  tau2_un * piyn) / ut;
          pitx = (pixx * ux  +  pixy * uy  +  tau2_un * pixn) / ut;
          pitt = (pitx * ux  +  pity * uy  +  tau2_un * pitn) / ut;
        }

        double bulkPi = 0.0;                    // bulk pressure (GeV/fm^3)

        if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell_glb];

        double muB = 0.0;                       // baryon chemical potential (GeV)
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density (fm^-3)
        double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
        double Vx = 0.0;                        // enforce orthogonality V.u = 0
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          muB = muB_fo[icell_glb];
          nB = nB_fo[icell_glb];
          Vx = Vx_fo[icell_glb];
          Vy = Vy_fo[icell_glb];
          Vn = Vn_fo[icell_glb];
          Vt = (Vx * ux  +  Vy * uy  +  Vn * tau2_un) / ut;

          alphaB = muB / T;
          baryon_enthalpy_ratio = nB / (E + P);
        }

        double tau2_pitn = tau2 * pitn;   // useful expressions
        double tau2_pixn = tau2 * pixn;
        double tau2_piyn = tau2 * piyn;
        double tau4_pinn = tau2 * tau2 * pinn;
        double tau2_Vn = tau2 * Vn;


        // set df coefficients
        deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

        double c0 = df.c0;             // 14 moment coefficients
        double c1 = df.c1;
        double c2 = df.c2;
        double c3 = df.c3;
        double c4 = df.c4;
        double shear14_coeff = df.shear14_coeff;

        double F = df.F;               // Chapman Enskog
        double G = df.G;
        double betabulk = df.betabulk;
        double betaV = df.betaV;
        double betapi = df.betapi;

        // shear and bulk coefficients
        double shear_coeff = 0.0;
        double bulk0_coeff = 0.0;
        double bulk1_coeff = 0.0;
        double bulk2_coeff = 0.0;
        double diff0_coeff = 0.0;
        double diff1_coeff = 0.0;

        switch(DF_MODE)
        {
          case 1: // 14 moment
          {
            shear_coeff = 1.0 / shear14_coeff;
            bulk0_coeff = (c0 - c2) * bulkPi;
            bulk1_coeff = c1 * bulkPi;
            bulk2_coeff = (4.*c2 - c0) * bulkPi;
            diff0_coeff = c3;
            diff1_coeff = c4;
            break;
          }
          case 2: // Chapman enskog
          {
            shear_coeff = 0.5 / (betapi * T);
            bulk0_coeff = F / (T * T * betabulk) * bulkPi;
            bulk1_coeff = G / betabulk * bulkPi;
            bulk2_coeff = bulkPi / (3.0 * T * betabulk);
            diff0_coeff = baryon_enthalpy_ratio / betaV;
            diff1_coeff = 1.0 / betaV;
            break;
          }
          default:
          {
            printf("Error: set df_mode = (1,2) in parameters.dat\n");
          }
        }


        // now loop over all particle species and momenta
        for(long ipart = 0; ipart < npart; ipart++)
        {
          long iS0D = pT_tab_length * ipart;

          double mass = Mass[ipart];              // mass (GeV)
          double mass_squared = mass * mass;
          double sign = Sign[ipart];              // quantum statistics sign
          double degeneracy = Degeneracy[ipart];  // spin degeneracy
          double baryon = Baryon[ipart];          // baryon number
          double chem = baryon * alphaB;          // chemical potential term in feq

          for(long ipT = 0; ipT < pT_tab_length; ipT++)
          {
            long iS1D =  phi_tab_length * (ipT + iS0D);

            double pT = pTValues[ipT];              // p_T (GeV)
            double mT = sqrt(mass_squared  +  pT * pT);    // m_T (GeV)
            double mT_over_tau = mT / tau;

            for(long iphip = 0; iphip < phi_tab_length; iphip++)
            {
              long iS2D = y_tab_length * (iphip + iS1D);

              double px = pT * cosphiValues[iphip]; // p^x
              double py = pT * sinphiValues[iphip]; // p^y

              double px_dax = px * dax;   // useful expressions
              double py_day = py * day;

              double px_ux = px * ux;
              double py_uy = py * uy;

              double pixx_px_px = pixx * px * px;
              double piyy_py_py = piyy * py * py;
              double pitx_px = pitx * px;
              double pity_py = pity * py;
              double pixy_px_py = pixy * px * py;
              double tau2_pixn_px = tau2_pixn * px;
              double tau2_piyn_py = tau2_piyn * py;

              double Vx_px = Vx * px;
              double Vy_py = Vy * py;

              for(long iy = 0; iy < y_tab_length; iy++)
              {
                long iS3D = iy + iS2D;

                double y = yValues[iy];

                double eta_integral = 0.0;

                // sum over eta
                for(long ieta = 0; ieta < eta_tab_length; ieta++)
                {
                  double eta = etaValues[ieta];
                  double eta_weight = etaWeights[ieta];

                  double sinhyeta = sinh(y - eta);
                  double coshyeta = sqrt(1.0  +  sinhyeta * sinhyeta);

                  double pt = mT * coshyeta;           // p^tau
                  double pn = mT_over_tau * sinhyeta;  // p^eta

                  double pdotdsigma = pt * dat  +  px_dax  +  py_day  +  pn * dan;

                  if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                  double E = pt * ut  -  px_ux  -  py_uy  -  pn * tau2_un;  // u.p
                  double feq = 1.0 / (exp(E/T  -  chem) + sign);

                  double feqbar = 1.0  -  sign * feq;

                  // pi^munu.p_mu.p_nu
                  double pimunu_pmu_pnu = pitt * pt * pt  +  pixx_px_px  +  piyy_py_py  +  tau4_pinn * pn * pn
                      + 2.0 * (-(pitx_px + pity_py) * pt  +  pixy_px_py  +  pn * (tau2_pixn_px  +  tau2_piyn_py  -  tau2_pitn * pt));

                  // V^mu.p_mu
                  double Vmu_pmu = Vt * pt  -  Vx_px  -  Vy_py  -  tau2_Vn * pn;

                  double df;

                  switch(DF_MODE)
                  {
                    case 1: // 14 moment
                    {
                      double df_shear = shear_coeff * pimunu_pmu_pnu;
                      double df_bulk = bulk0_coeff * mass_squared  +  (bulk1_coeff * baryon  +  bulk2_coeff * E) * E;
                      double df_diff = (diff0_coeff * baryon +  diff1_coeff * E) * Vmu_pmu;

                      df = feqbar * (df_shear + df_bulk + df_diff);
                      break;
                    }
                    case 2: // Chapman enskog
                    {
                      double df_shear = shear_coeff * pimunu_pmu_pnu / E;
                      double df_bulk = bulk0_coeff * E  +  bulk1_coeff * baryon  +  bulk2_coeff * (E  -  mass_squared / E);
                      double df_diff = (diff0_coeff  -  diff1_coeff * baryon / E) * Vmu_pmu;

                      df = feqbar * (df_shear + df_bulk + df_diff);
                      break;
                    }
                    default:
                    {
                      printf("Error: set df_mode = (1,2) in parameters.dat\n");
                    }
                  } // DF_MODE

                  if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0));

                  double f = feq * (1.0 + df);

                  eta_integral += eta_weight * pdotdsigma * f;

                } // ieta

                dN_pTdpTdphidy_all[n  +  CORES * iS3D] += (prefactor * degeneracy * eta_integral);

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

        } // particle species (ipart)

      } // freezeout cells in the chunk (icell)

    } // number of chunks / cores (n)




    // now perform the reduction over cores
    #pragma omp parallel for collapse(4)
    for(long ipart = 0; ipart < npart; ipart++)
    {
      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          for(long iy = 0; iy < y_tab_length; iy++)
          {
            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            double dN_pTdpTdphidy_tmp = 0.0; // reduction variable

            #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
            for(long n = 0; n < CORES; n++)
            {
              dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[n  +  CORES * iS3D];

            } // sum over the cores

            dN_pTdpTdphidy[iS3D] = dN_pTdpTdphidy_tmp;

          } // rapidity points (iy)

        } // azimuthal angle points (iphip)

      } // transverse momentum points (ipT)

    } // particle species (ipart)

    // free memory
    free(dN_pTdpTdphidy_all);
  }



  void EmissionFunctionArray::calculate_dN_pTdpTdphidy_feqmod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, Deltaf_Data * df_data)
  {
    double prefactor = pow(2.0 * M_PI * hbarC, -3);

    long FO_chunk = FO_length / CORES;
    long remainder = FO_length  -  CORES * FO_chunk;

    cout << "Number of cores = " << CORES << endl;
    cout << "Chunk size = " << FO_chunk << endl;
    cout << "Remainder cells = " << remainder << endl;

    if(remainder != 0) FO_chunk++;

    double detA_min = DETA_MIN;   // default value for minimum detA
    long breakdown = 0;           // number of times feqmod breaks down
    double tau_breakdown = 0;     // tau until feqmod stops breaking down

    long pl_negative = 0;         // number of times pl < 0
    double tau_pl = 0;            // tau until pl > 0

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];
    double phiWeights[phi_tab_length];

    for(long iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab->get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);
      phiWeights[iphip] = phi_tab->get(2, iphip + 1);
    }

    // pT array
    double pTValues[pT_tab_length];

    for(long ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab->get(1, ipT + 1);
    }

    double yValues[y_tab_length];
    double etaValues[eta_tab_length];
    double etaWeights[eta_tab_length];

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(long ieta = 0; ieta < eta_tab_length; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;
      etaWeights[0] = 1.0;
      for(long iy = 0; iy < y_tab_length; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    /// gauss laguerre roots
    const int pbar_pts = laguerre->points;

    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];

    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    //declare a huge array of size CORES * npart * pT_tab_length * phi_tab_length * y_tab_length to hold spectra for each chunk
    long npart = (long)number_of_chosen_particles;
    double *dN_pTdpTdphidy_all = (double*)calloc(CORES * npart * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));


    // subdivide bite size chunks of freezeout surface across cores
    #pragma omp parallel for
    for(long n = 0; n < CORES; n++)
    {
      double ** A_copy = (double**)calloc(3, sizeof(double*));
      for(int i = 0; i < 3; i++) A_copy[i] = (double*)calloc(3, sizeof(double));

      double ** A_inv = (double**)calloc(3, sizeof(double*));
      for(int i = 0; i < 3; i++) A_inv[i] = (double*)calloc(3, sizeof(double));

      long endFO = FO_chunk;

      for(long icell = 0; icell < endFO; icell++)  // cell index inside each chunk
      {
        if((icell == endFO - 1) && (remainder != 0) && (n > remainder - 1)) continue;

        long icell_glb = n  +  icell * CORES;

        double tau = tau_fo[icell_glb];     // longitudinal proper time
        double tau2 = tau * tau;
        if(DIMENSION == 3)
        {
          etaValues[0] = eta_fo[icell_glb]; // spacetime rapidity from freezeout cell
        }

        double dat = dat_fo[icell_glb];     // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];

        double ux = ux_fo[icell_glb];       // contravariant fluid velocity
        double uy = uy_fo[icell_glb];       // enforce normalization u.u = 1
        double un = un_fo[icell_glb];
        double ut = sqrt(1.0 +  ux * ux  +  uy * uy  +  tau2 * un * un);

        // skip cells with u.dsigma < 0
        if(ut * dat  +  ux * dax  +  uy * day  +  un * dan <= 0.0) continue;

        double ut2 = ut * ut;               // useful expressions
        double ux2 = ux * ux;
        double uy2 = uy * uy;
        double uperp = sqrt(ux * ux  +  uy * uy);
        double utperp = sqrt(1.0  +   ux * ux  +  uy * uy);

        double T = T_fo[icell_glb];             // temperature (GeV)
        double P = P_fo[icell_glb];             // equilibrium pressure (GeV/fm^3)
        double E = E_fo[icell_glb];             // energy density (GeV/fm^3)

        double pitt = 0.0;                  // contravariant shear stress tensor pi^munu
        double pitx = 0.0;                  // enforce orthogonality pi.u = 0
        double pity = 0.0;                  // and tracelessness Tr(pi) = 0
        double pitn = 0.0;
        double pixx = 0.0;
        double pixy = 0.0;
        double pixn = 0.0;
        double piyy = 0.0;
        double piyn = 0.0;
        double pinn = 0.0;

        if(INCLUDE_SHEAR_DELTAF)
        {
          pixx = pixx_fo[icell_glb];
          pixy = pixy_fo[icell_glb];
          pixn = pixn_fo[icell_glb];
          piyy = piyy_fo[icell_glb];
          piyn = piyn_fo[icell_glb];
          pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
          pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
          pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
          pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
          pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
        }

        double bulkPi = 0.0;                // bulk pressure (GeV/fm^3)

        if(INCLUDE_BULK_DELTAF)
        {
          bulkPi = bulkPi_fo[icell_glb];
        }


        double muB = 0.0;                       // baryon chemical potential (GeV)
        double alphaB = 0.0;                    // muB / T
        double nB = 0.0;                        // net baryon density (fm^-3)
        double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
        double Vx = 0.0;                        // enforce orthogonality V.u = 0
        double Vy = 0.0;
        double Vn = 0.0;
        double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

        if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
        {
          muB = muB_fo[icell_glb];
          nB = nB_fo[icell_glb];
          Vx = Vx_fo[icell_glb];
          Vy = Vy_fo[icell_glb];
          Vn = Vn_fo[icell_glb];
          Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

          alphaB = muB / T;
          baryon_enthalpy_ratio = nB / (E + P);
        }

        // regulate bulk pressure if goes out of bounds given
        // by Jonah's feqmod to avoid gsl interpolation errors
        if(DF_MODE == 4)
        {
          double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;

          if(bulkPi < - P)
          {
            bulkPi = - (1.0 - 1.e-5) * P;
          }
          else if(bulkPi / P > bulkPi_over_Peq_max)
          {
            bulkPi = P * (bulkPi_over_Peq_max - 1.e-5);
          }
        }


        // check if pl went negative
        double zt = tau * un / utperp;
        double zn = ut / (tau * utperp);
        double pl = P  +  bulkPi  +  zt * zt * pitt  +  tau2 * tau2 * zn * zn * pinn  +  2. * tau2 * zt * zn * pitn;

        #pragma omp critical
        if(pl < 0)
        {
          pl_negative++;
          tau_pl = tau;
        }



        // set df coefficients
        deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

        // modified coefficients (Mike / Jonah)
        double F = df.F;
        double G = df.G;
        double betabulk = df.betabulk;
        double betaV = df.betaV;
        double betapi = df.betapi;
        double lambda = df.lambda;
        double z = df.z;
        double delta_lambda = df.delta_lambda;
        double delta_z = df.delta_z;

        // milne basis class
        Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
        basis_vectors.test_orthonormality(tau2);

        double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
        double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
        double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
        double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

        // shear stress class
        Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
        pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
        pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

        // baryon diffusion class
        Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
        Vmu.test_Vmu_orthogonality(ut, ux, uy, un, tau2);
        Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);


        // modified temperature / chemical potential
        double T_mod = T;
        double alphaB_mod = alphaB;

        if(DF_MODE == 3)
        {
          T_mod = T  +  bulkPi * F / betabulk;
          alphaB_mod = alphaB  +  bulkPi * G / betabulk;
        }

        // linearized Chapman Enskog df coefficients (for Mike only)
        double shear_coeff = 0.5 / (betapi * T);      // Jonah linear df also shares shear coeff
        double bulk0_coeff = F / (T * T * betabulk);
        double bulk1_coeff = G / betabulk;
        double bulk2_coeff = 1.0 / (3.0 * T * betabulk);

        // pimunu and Vmu LRF components
        double pixx_LRF = pimunu.pixx_LRF;
        double pixy_LRF = pimunu.pixy_LRF;
        double pixz_LRF = pimunu.pixz_LRF;
        double piyy_LRF = pimunu.piyy_LRF;
        double piyz_LRF = pimunu.piyz_LRF;
        double pizz_LRF = pimunu.pizz_LRF;

        double Vx_LRF = Vmu.Vx_LRF;
        double Vy_LRF = Vmu.Vy_LRF;
        double Vz_LRF = Vmu.Vz_LRF;

        // local momentum transformation matrix Mij = Aij
        // Aij = ideal + shear + bulk is symmetric
        // Mij is not symmetric if include baryon diffusion (leave for future work)

        // coefficients in Aij
        double shear_mod = 0.5 / betapi;
        double bulk_mod = bulkPi / (3.0 * betabulk);

        if(DF_MODE == 4)
        {
          bulk_mod = lambda;
        }

        double Axx = 1.0  +  pixx_LRF * shear_mod  +  bulk_mod;
        double Axy = pixy_LRF * shear_mod;
        double Axz = pixz_LRF * shear_mod;
        double Ayx = Axy;
        double Ayy = 1.0  +  piyy_LRF * shear_mod  +  bulk_mod;
        double Ayz = piyz_LRF * shear_mod;
        double Azx = Axz;
        double Azy = Ayz;
        double Azz = 1.0  +  pizz_LRF * shear_mod  +  bulk_mod;

        double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);
        double detA_bulk_two_thirds = pow(1.0 + bulk_mod, 2);

        // set Mij matrix
        double A[] = {Axx, Axy, Axz,
                      Ayx, Ayy, Ayz,
                      Azx, Azy, Azz};           // gsl matrix format

        A_copy[0][0] = Axx;  A_copy[0][1] = Axy;  A_copy[0][2] = Axz;
        A_copy[1][0] = Ayx;  A_copy[1][1] = Ayy;  A_copy[1][2] = Ayz;
        A_copy[2][0] = Azx;  A_copy[2][1] = Azy;  A_copy[2][2] = Azz;

        int s;
        gsl_matrix_view M = gsl_matrix_view_array(A, 3, 3);
        gsl_matrix_view LU = gsl_matrix_view_array(A, 3, 3);

        gsl_permutation * p = gsl_permutation_calloc(3);

        gsl_linalg_LU_decomp(&LU.matrix, p, &s);

        gsl_matrix * A_inverse = gsl_matrix_alloc(3,3);

        gsl_linalg_LU_invert(&LU.matrix, p, A_inverse);

        for(int i = 0; i < 3; i++)
        {
          for(int j = 0; j < 3; j++)
          {
            A_inv[i][j] = gsl_matrix_get(A_inverse, i, j);
          }
        }

         // prefactors for equilibrium, linear bulk correction and modified densities (Mike's feqmod)
        double neq_fact = T * T * T / two_pi2_hbarC3;
        double dn_fact = bulkPi / betabulk;
        double J20_fact = T * neq_fact;
        double N10_fact = neq_fact;
        double nmod_fact = T_mod * T_mod * T_mod / two_pi2_hbarC3;

        // determine if feqmod breaks down
        bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, detA_min, z, laguerre, DF_MODE, 0, T, F, betabulk);

        #pragma omp critical
        if(feqmod_breaks_down)
        {
          breakdown++;
          tau_breakdown = tau;
        }

        // uniformly rescale eta space by detA if modified momentum space elements are shrunk
        // this rescales the dsigma components orthogonal to the eta direction (only works for 2+1d, y = 0)
        // for integrating modified distribution with narrow (y-eta) distributions
        double eta_scale = 1.0;
        if(detA > detA_min && DIMENSION == 2)
        {
          eta_scale = detA / detA_bulk_two_thirds;
        }

        // loop over hadrons
        for(long ipart = 0; ipart < npart; ipart++)
        {
          long iS0D = pT_tab_length * ipart;

          // set particle properties
          double mass = Mass[ipart];              // mass (GeV)
          double mass2 = mass * mass;
          double sign = Sign[ipart];              // quantum statistics sign
          double degeneracy = Degeneracy[ipart];  // spin degeneracy
          double baryon = Baryon[ipart];          // baryon number

          double chem = baryon * alphaB;          // chemical potential term in feq
          double chem_mod = baryon * alphaB_mod;  // chemical potential term in feqmod

          // modified renormalization factor
          double renorm = 1.0;

          if(INCLUDE_BULK_DELTAF)
          {
            if(DF_MODE == 3)
            {
              double mbar = mass / T;
              double mbar_mod = mass / T_mod;

              double neq = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

              double N10 = baryon * N10_fact * degeneracy * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

              double J20 = J20_fact * degeneracy * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);

              double n_linear = neq  +  dn_fact * (neq  +  N10 * G  +  J20 * F / T / T);

              double n_mod = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);

              renorm = n_linear / n_mod;
            }
            else if(DF_MODE == 4)
            {
              renorm = z;
            }

          }

          if(DIMENSION == 2)
          {
            renorm /= detA_bulk_two_thirds;
          }
          else if(DIMENSION == 3)
          {
            renorm /= detA;
          }

          if((std::isnan(renorm) || std::isinf(renorm)))
          {
            cout << "Error: renormalization factor is " << renorm << endl;
            continue;
          }

          for(long ipT = 0; ipT < pT_tab_length; ipT++)
          {
            long iS1D =  phi_tab_length * (ipT + iS0D);

            double pT = pTValues[ipT];              // p_T (GeV)
            double mT = sqrt(mass2  +  pT * pT);    // m_T (GeV)
            double mT_over_tau = mT / tau;

            for(long iphip = 0; iphip < phi_tab_length; iphip++)
            {
              long iS2D = y_tab_length * (iphip + iS1D);

              double px = pT * cosphiValues[iphip]; // p^x
              double py = pT * sinphiValues[iphip]; // p^y

              for(long iy = 0; iy < y_tab_length; iy++)
              {
                long iS3D = iy + iS2D;

                double y = yValues[iy];

                double eta_integral = 0.0;  // Cooper Frye integral over eta

                // integrate over eta
                for(long ieta = 0; ieta < eta_tab_length; ieta++)
                {
                  double eta = etaValues[ieta];
                  double eta_weight = etaWeights[ieta];

                  bool feqmod_breaks_down_narrow = false;

                  if(DIMENSION == 3 && !feqmod_breaks_down)
                  {
                    if(detA < 0.01 && fabs(y - eta) < detA)
                    {
                      feqmod_breaks_down_narrow = true;
                    }
                  }

                  double pdotdsigma;
                  double f;           // feqmod (if breakdown do feq(1+df))

                  // calculate feqmod
                  if(feqmod_breaks_down || feqmod_breaks_down_narrow)
                  {
                    double pt = mT * cosh(y - eta);          // p^\tau (GeV)
                    double pn = mT_over_tau * sinh(y - eta); // p^\eta (GeV^2)
                    double tau2_pn = tau2 * pn;

                    pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day)  +  pn * dan;

                    if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                    if(DF_MODE == 3)
                    {
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                      double feqbar = 1.0  -  sign * feq;

                       // pi^munu.p_mu.p_nu
                      double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                       + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                      // V^mu.p_mu
                      double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;

                      double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu  -  mass2 / pdotu)) * bulkPi;
                      double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;

                      double df = feqbar * (df_shear + df_bulk + df_diff);

                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                      f = feq * (1.0 + df);
                    }
                    else if(DF_MODE == 4)
                    {
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      double feq = 1.0 / (exp(pdotu / T) + sign);
                      double feqbar = 1.0  -  sign * feq;

                       // pi^munu.p_mu.p_nu
                      double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                       + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                      double df_shear = feqbar * shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = delta_z  -  3.0 * delta_lambda  +  feqbar * delta_lambda * (pdotu  -  mass2 / pdotu) / T;

                      double df = df_shear + df_bulk;

                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                      f = feq * (1.0 + df);
                    }
                  } // feqmod breaks down
                  else
                  {
                    double pt = mT * cosh(y - eta_scale * eta);          // p^\tau (GeV)
                    double pn = mT_over_tau * sinh(y - eta_scale * eta); // p^\eta (GeV^2)
                    double tau2_pn = tau2 * pn;

                    pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day)  +  pn * dan;

                    if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                    // LRF momentum components pi_LRF = - Xi.p
                    double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;
                    double py_LRF = Yx * px  +  Yy * py;
                    double pz_LRF = -Zt * pt  +  Zn * tau2_pn;

                    double pLRF[3] = {px_LRF, py_LRF, pz_LRF};
                    double pLRF_prev[3];

                    double pLRF_mod_prev[3];
                    double pLRF_mod[3];

                    double dpLRF[3];
                    double dpLRF_mod[3];

                    matrix_multiplication(A_inv, pLRF, pLRF_mod, 3, 3);   // evaluate p_mod = A^-1.p at least once

                    double dp;
                    double eps = 1.e-16;

                    for(int i = 0; i < 5; i++)
                    {
                      vector_copy(pLRF_mod, pLRF_mod_prev, 3);                        // copy result for iteration
                      matrix_multiplication(A_copy, pLRF_mod_prev, pLRF_prev, 3, 3);  // compute pLRF error
                      vector_subtraction(pLRF, pLRF_prev, dpLRF, 3);

                      dp = sqrt(dpLRF[0] * dpLRF[0]  +  dpLRF[1] * dpLRF[1]  +  dpLRF[2] * dpLRF[2]);

                      if(dp <= eps) break;

                      matrix_multiplication(A_inv, dpLRF, dpLRF_mod, 3, 3);           // compute correction to pLRF_mod
                      vector_addition(pLRF_mod_prev, dpLRF_mod, pLRF_mod, 3);         // add correction to pLRF_mod
                    }

                    double px_LRF_mod = pLRF_mod[0];
                    double py_LRF_mod = pLRF_mod[1];
                    double pz_LRF_mod = pLRF_mod[2];

                    double E_mod = sqrt(mass2  +  px_LRF_mod * px_LRF_mod  +  py_LRF_mod * py_LRF_mod  +  pz_LRF_mod * pz_LRF_mod);

                    f = fabs(renorm) / (exp(E_mod / T_mod  -  chem_mod) + sign); // feqmod
                  }

                  eta_integral += (pdotdsigma * f); // add contribution to integral

                } // eta points (ieta)

                dN_pTdpTdphidy_all[n  +  CORES * iS3D] += (prefactor * degeneracy * eta_integral);

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

        } // particle species (ipart)

        gsl_matrix_free(A_inverse);
        gsl_permutation_free(p);

      } // freezeout cells in the chunk (icell)

      free_2D(A_copy, 3);
      free_2D(A_inv, 3);

    } // number of chunks / cores (n)


    // now perform the reduction over cores
    #pragma omp parallel for collapse(4)
    for(long ipart = 0; ipart < npart; ipart++)
    {
      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          for(long iy = 0; iy < y_tab_length; iy++)
          {
            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            double dN_pTdpTdphidy_tmp = 0.0; // reduction variable

            #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
            for(long n = 0; n < CORES; n++)
            {
              dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[n  +  CORES * iS3D];

            } // sum over the cores

            dN_pTdpTdphidy[iS3D] = dN_pTdpTdphidy_tmp;

          } // rapidity points (iy)

        } // azimuthal angle points (iphip)

      } // transverse momentum points (ipT)

    } // particle species (ipart)


    printf("\nfeqmod breaks down for %ld / %ld cells until t = %.3f fm/c\n", breakdown, FO_length, tau_breakdown);
    printf("pl went negative for %ld / %ld cells until t = %.3f fm/c\n\n", pl_negative, FO_length, tau_pl);

    //free memory
    free(dN_pTdpTdphidy_all);
  }




  void EmissionFunctionArray::calculate_dN_pTdpTdphidy_famod(double *Mass, double *Sign, double *Degeneracy, double *Baryon, double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo, double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo, double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, double *Mass_PDG, double *Sign_PDG, double *Degeneracy_PDG, double *Baryon_PDG)
  {
    double prefactor = pow(2.0 * M_PI * hbarC, -3);

    long FO_chunk = FO_length / CORES;
    long remainder = FO_length  -  CORES * FO_chunk;

    printf("Number of cores = %ld\n", CORES);
    printf("Chunk size = %ld\n", FO_chunk);
    printf("Remainder cells = %ld\n", remainder);

    if(remainder != 0)
    {
      FO_chunk++;                           // round up by one
    }

    double detB_min = DETA_MIN;             // default value for minimum detB = detC . detA
    long breakdown = 0;                     // number of times fa or famod breaks down
    double tau_breakdown = 0;               // tau until feqmod stops breaking down

    long pl_negative = 0;                   // number of times pl < 0 (use continue or set fa = feq?)
    double tau_pl = 0;                      // tau until pl > 0

    double cosphi_values[phi_tab_length];   // phi arrays
    double sinphi_values[phi_tab_length];
    double phi_weights[phi_tab_length];

    double pT_values[pT_tab_length];        // pT array

    double y_values[y_tab_length];          // rapidity and spacetime rapidity arrays
    double eta_values[eta_tab_length];
    double eta_weights[eta_tab_length];

    for(long iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab->get(1, iphip + 1);
      cosphi_values[iphip] = cos(phi);
      sinphi_values[iphip] = sin(phi);
      phi_weights[iphip] = phi_tab->get(2, iphip + 1);
    }

    for(long ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pT_values[ipT] = pT_tab->get(1, ipT + 1);
    }

    if(DIMENSION == 2)
    {
      y_values[0] = 0;

      for(long ieta = 0; ieta < eta_tab_length; ieta++)
      {
        eta_values[ieta] = eta_tab->get(1, ieta + 1);
        eta_weights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      eta_values[0] = 0;
      eta_weights[0] = 1;

      for(long iy = 0; iy < y_tab_length; iy++)
      {
        y_values[iy] = y_tab->get(1, iy + 1);
      }
    }


    // ********

    // gauss laguerre roots
    const int pbar_pts = laguerre->points;

    double * pbar_root1 = laguerre->root[1];          // a = 1, 2  gla needed
    double * pbar_root2 = laguerre->root[2];

    double * pbar_weight1 = laguerre->weight[1];      // I might need more than this...
    double * pbar_weight2 = laguerre->weight[2];




    // allocate a huge array to hold momentum spectra for each chunk
    long npart = (long)number_of_chosen_particles;
    double *dN_pTdpTdphidy_all = (double*)calloc(CORES * npart * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));


    // subdivide bite size chunks of freezeout surface across cores
    #pragma omp parallel for
    for(long n = 0; n < CORES; n++)
    {
      double **B_copy = (double**)calloc(3, sizeof(double*));   // momentum transformation matrix
      double **B_inv  = (double**)calloc(3, sizeof(double*));

      for(int i = 0; i < 3; i++)
      {
        B_copy[i] = (double*)calloc(3, sizeof(double));
        B_inv[i]  = (double*)calloc(3, sizeof(double));
      }

      long endFO = FO_chunk;

      for(long icell = 0; icell < endFO; icell++)  // cell index inside each chunk
      {
        if((icell == endFO - 1) && (remainder != 0) && (n > remainder - 1))
        {
          continue;
        }

        long icell_glb = n  +  icell * CORES;   // global freezeout cell index


        // freezeout cell info
        double tau = tau_fo[icell_glb];         // longitudinal proper time
        double tau2 = tau * tau;

        if(DIMENSION == 3)
        {
          eta_values[0] = eta_fo[icell_glb];     // spacetime rapidity of freezeout cell
        }

        double dat = dat_fo[icell_glb];         // covariant normal surface vector
        double dax = dax_fo[icell_glb];
        double day = day_fo[icell_glb];
        double dan = dan_fo[icell_glb];

        double ux = ux_fo[icell_glb];           // contravariant fluid velocity
        double uy = uy_fo[icell_glb];           // enforce normalization u.u = 1
        double un = un_fo[icell_glb];
        double ut = sqrt(1.  +  ux * ux  +  uy * uy  +  tau2 * un * un);

        if(ut * dat  +  ux * dax  +  uy * day  +  un * dan <= 0)
        {
          continue;                             // skip cells with u.dsigma < 0
        }

        double ut2 = ut * ut;                   // useful expressions
        double ux2 = ux * ux;
        double uy2 = uy * uy;
        double uperp = sqrt(ux * ux  +  uy * uy);
        double utperp = sqrt(1.  +   ux * ux  +  uy * uy);

        double T = T_fo[icell_glb];             // temperature [GeV]
        double P = P_fo[icell_glb];             // equilibrium pressure [GeV/fm^3]
        double E = E_fo[icell_glb];             // energy density (GeV/fm^3)

        double pixx = pixx_fo[icell_glb];       // contravariant standard shear stress tensor pi^\munu
        double pixy = pixy_fo[icell_glb];       // reconstruct remaining components to enforce
        double pixn = pixn_fo[icell_glb];       // orthogonality and tracelessness pi.u = Tr(pi) = 0
        double piyy = piyy_fo[icell_glb];
        double piyn = piyn_fo[icell_glb];

        double pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2. * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
        double pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
        double pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
        double pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
        double pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;

        double bulkPi = bulkPi_fo[icell_glb];   // bulk pressure (GeV/fm^3)

        double muB = 0;                         // baryon chemical potential (GeV)
        double Vt = 0;                          // contravariant net baryon diffusion V^mu
        double Vx = 0;                          // enforce orthogonality V.u = 0
        double Vy = 0;
        double Vn = 0;

        if(INCLUDE_BARYON)
        {
          muB = muB_fo[icell_glb];

          if(INCLUDE_BARYONDIFF_DELTAF)
          {
            Vx = Vx_fo[icell_glb];              // baryon diffusion not included in famod yet
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
            Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;
          }
        }

        double alphaB = muB / T;                // muB / T


        // milne basis vector components
        Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
        basis_vectors.test_orthonormality(tau2);

        double Xt = basis_vectors.Xt;
        double Xx = basis_vectors.Xx;
        double Xy = basis_vectors.Xy;
        double Xn = basis_vectors.Xn;

        double Yx = basis_vectors.Yx;
        double Yy = basis_vectors.Yy;

        double Zt = basis_vectors.Zt;
        double Zn = basis_vectors.Zn;


        // compute shear stress in LRF
        Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
        pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
        pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

        double pixx_LRF = pimunu.pixx_LRF;      // standard pi^munu LRF components
        double pixy_LRF = pimunu.pixy_LRF;
        double pixz_LRF = pimunu.pixz_LRF;
        double piyy_LRF = pimunu.piyy_LRF;
        double piyz_LRF = pimunu.piyz_LRF;
        double pizz_LRF = pimunu.pizz_LRF;


        Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);   // compute baryon diffusion in LRF
        Vmu.test_Vmu_orthogonality(ut, ux, uy, un, tau2);
        Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);

        double Vx_LRF = Vmu.Vx_LRF;             // standard V^\mu LRF components
        double Vy_LRF = Vmu.Vy_LRF;
        double Vz_LRF = Vmu.Vz_LRF;


        // anisotropic hydrodynamic variables
        double pl = P + bulkPi + pizz_LRF;      // longitudinal pressure
        double pt = P + bulkPi - pizz_LRF/2.;   // transverse pressure

        double piTxx_LRF = 0;                   // piperp^\munu LRF components
        double piTxy_LRF = 0;
        double piTyy_LRF = 0;
        double WTzx_LRF = 0;                    // Wperpz^\mu LRF components
        double WTzy_LRF = 0;

        if(INCLUDE_SHEAR_DELTAF)                // include residual shear corrections
        {
          piTxx_LRF = (pixx_LRF - piyy_LRF) / 2.;
          piTxy_LRF = pixy_LRF;
          piTyy_LRF = -(piTxx_LRF);

          WTzx_LRF = pixz_LRF;
          WTzy_LRF = piyz_LRF;
        }


        #pragma omp critical
        if(pl < 0)
        {
          pl_negative++;          // isn't there a better pragma to use?
          tau_pl = tau;
        }


        double Lambda = T;                      // effective temperature
        double aT = 1;                          // transverse momentum scale
        double aL = 1;                          // longitudinal momentum scale
        double upsilonB = alphaB;               // effective chemical potential


        // **** reconstruct anisotropic variables ***

        // do this once I've got the setup right


        // compute aniso CE coefficients (after reconstruct aniso variables)

        double shear_coeff = 0;    // 1 / (2.betapi_perp)
        double diff_coeff = 0;     // 1 / betaW_perp

        // perhaps I can use ~ 1/ 2.betapi as a substitute?


        // leading order deformation matrix Aij (diagonal)
        double Axx = aT;
        double Ayy = aT;
        double Azz = aL;

        double detA = Axx * Ayy * Azz;


        // residual shear deformation matrix Cij (asymmetric)
        double Cxx = 1  +  shear_coeff * piTxx_LRF;
        double Cxy = shear_coeff * piTxy_LRF;
        double Cxz = diff_coeff * WTzx_LRF * aT / (aT + aL);

        double Cyx = Cxy;
        double Cyy = 1  +  shear_coeff * piTyy_LRF;
        double Cyz = diff_coeff * WTzy_LRF * aT / (aT + aL);

        double Czx = diff_coeff * WTzx_LRF * aL / (aT + aL);
        double Czy = diff_coeff * WTzy_LRF * aL / (aT + aL);
        double Czz = 1;

        double detC = Cxx * (Cyy * Czz  -  Cyz * Czy)  -  Cxy * (Cyx * Czz  -  Cyz * Czx)  +  Cxz * (Cyx * Czy  -  Cyy * Czx);


        // total momentum transformation matrix Bij (symmetric)
        double Bxx = Axx  +  aT * shear_coeff * piTxx_LRF;
        double Bxy = aT * shear_coeff * piTxy_LRF;
        double Bxz = diff_coeff * WTzx_LRF * aT * aL / (aT + aL);

        double Byx = Bxy;
        double Byy = Ayy  +  aT * shear_coeff * piTyy_LRF;
        double Byz = diff_coeff * WTzy_LRF * aT * aL / (aT + aL);

        double Bzx = Bxz;
        double Bzy = Byz;
        double Bzz = Azz;

        double detB = detC * detA;


        // set momentum transformation matrix
        double B[] = {Bxx, Bxy, Bxz,
                      Byx, Byy, Byz,
                      Bzx, Bzy, Bzz};           // gsl matrix format

        B_copy[0][0] = Bxx;  B_copy[0][1] = Bxy;  B_copy[0][2] = Bxz;
        B_copy[1][0] = Byx;  B_copy[1][1] = Byy;  B_copy[1][2] = Byz;
        B_copy[2][0] = Bzx;  B_copy[2][1] = Bzy;  B_copy[2][2] = Bzz;

        int s;
        gsl_matrix_view M = gsl_matrix_view_array(B, 3, 3);
        gsl_matrix_view LU = gsl_matrix_view_array(B, 3, 3);

        gsl_permutation *p = gsl_permutation_calloc(3);

        gsl_linalg_LU_decomp(&LU.matrix, p, &s);

        gsl_matrix *B_inverse = gsl_matrix_alloc(3,3);

        gsl_linalg_LU_invert(&LU.matrix, p, B_inverse);

        for(int i = 0; i < 3; i++)
        {
          for(int j = 0; j < 3; j++)
          {
            B_inv[i][j] = gsl_matrix_get(B_inverse, i, j);
          }
        }



        // *** need to review this (check feqmod breakdown function)

        bool fa_famod_breaks_down = false;      // replace with a function (or if/else statement)




        #pragma omp critical
        if(fa_famod_breaks_down)
        {
          breakdown++;
          tau_breakdown = tau;
        }

        // uniformly rescale eta space by detA if modified momentum space elements are shrunk
        // this rescales the dsigma components orthogonal to the eta direction (only works for 2+1d, y = 0)
        // for integrating modified distribution with narrow (y-eta) distributions
        double eta_scale = 1;


        // I should rescale by the total deformation detB


        // *** need to review this

        if(detB > detB_min && DIMENSION == 2)
        {
          eta_scale = detB;     // how does this enter?   rescale by detB (rather than detC)
        }



        // *** need to review this (Zn = 1/detC)

        double renorm = detB / detC;                // renormalization factor (for dimension = 2)

        if(DIMENSION == 3)
        {
          renorm = 1. / detC;
        }

        if((std::isnan(renorm) || std::isinf(renorm)))
        {
        #ifdef FLAGS
          printf("calculate_dN_pTdpTdphidy_famod flag: renormalization factor = %lf is negative\n", renorm);
        #endif
          continue;
        }




        // loop over hadrons
        for(long ipart = 0; ipart < npart; ipart++)   // loop over chosen particles
        {
          long iS0D = pT_tab_length * ipart;

          double mass = Mass[ipart];                  // mass [GeV]
          double mass2 = mass * mass;
          double sign = Sign[ipart];                  // quantum statistics sign
          double degeneracy = Degeneracy[ipart];      // spin degeneracy
          double baryon = Baryon[ipart];              // baryon number
          double chem = baryon * alphaB;              // chemical potential term in feq
          double chem_effect = baryon * upsilonB;     // effective chemical potential term in fa, famod

          for(long ipT = 0; ipT < pT_tab_length; ipT++)   // loop over transverse momentum
          {
            long iS1D =  phi_tab_length * (ipT + iS0D);

            double pT = pT_values[ipT];                   // pT [GeV]
            double mT = sqrt(mass2  +  pT * pT);          // mT [GeV]
            double mT_over_tau = mT / tau;

            for(long iphip = 0; iphip < phi_tab_length; iphip++)  // loop over azimuthal angle
            {
              long iS2D = y_tab_length * (iphip + iS1D);

              double px = pT * cosphi_values[iphip];      // p^x
              double py = pT * sinphi_values[iphip];      // p^y

              for(long iy = 0; iy < y_tab_length; iy++)   // loop over rapidity points
              {
                long iS3D = iy + iS2D;

                double y = y_values[iy];                  // rapidity

                double eta_integral = 0;                  // Cooper-Frye eta_s integral

                for(long ieta = 0; ieta < eta_tab_length; ieta++) // loop over spacetime rapidity
                {
                  double eta = eta_values[ieta];          // spacetime rapidity
                  double eta_weight = eta_weights[ieta];  // integration weight

                  bool fa_famod_breaks_down_narrow = false;



                  // *** review this

                  if(DIMENSION == 3 && !fa_famod_breaks_down)
                  {
                    if(detB < 0.01 && fabs(y - eta) < detB)
                    {
                      fa_famod_breaks_down_narrow = true;
                    }
                  }



                  double p_dsigma;  // p.d\sigma
                  double f;         // fa, famod (or feq if fa, famod breaks down)

                  // calculate distribution
                  if(fa_famod_breaks_down || fa_famod_breaks_down_narrow) // set f = feq
                  {
                    double pt = mT * cosh(y - eta);          // p^\tau [GeV]
                    double pn = mT_over_tau * sinh(y - eta); // p^\eta [GeV/fm]
                    double tau2_pn = tau2 * pn;

                    p_dsigma = pt * dat  +  px * dax  +  py * day  +  pn * dan;

                    if(OUTFLOW && p_dsigma <= 0)
                    {
                      continue;     // enforce outflow Theta(p.d\sigma)
                    }

                    double u_p = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;    // u.p = LRF energy

                    f = 1. / (exp(u_p / T  -  chem)  +  sign);
                  }
                  else
                  {
                    double pt = mT * cosh(y - eta_scale * eta);          // p^\tau [GeV]
                    double pn = mT_over_tau * sinh(y - eta_scale * eta); // p^\eta [GeV/fm]
                    double tau2_pn = tau2 * pn;

                    p_dsigma = pt * dat  +  px * dax  +  py * day  +  pn * dan;

                    if(OUTFLOW && p_dsigma <= 0.0)
                    {
                      continue;
                    }

                    // LRF momentum components
                    double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;    // -X.p
                    double py_LRF = Yx * px  +  Yy * py;                                  // -Y.p
                    double pz_LRF = -Zt * pt  +  Zn * tau2_pn;                            // -Z.p

                    double pLRF[3] = {px_LRF, py_LRF, pz_LRF};          // pLRF components
                    double pLRF_prev[3];                                // track pLRF reproduction

                    double pLRF_mod_prev[3];                            // pLRF_mod (previous iteration)
                    double pLRF_mod[3];                                 // pLRF_mod (current)

                    double dpLRF[3];                                    // pLRF reproduction error
                    double dpLRF_mod[3];                                // pLRF_mod correction


                    // solve matrix equation: B.pLRF_mod = p_LRF for pLRF_mod

                    matrix_multiplication(B_inv, pLRF, pLRF_mod, 3, 3);               // invert p_mod = A^-1.p at least once

                    double dp;                                                        // norm of p_LRF reproduction error
                    double eps = 1.e-16;                                              // error tolerance

                    for(int i = 0; i < 5; i++)                                        // iterate solution until converged
                    {
                      vector_copy(pLRF_mod, pLRF_mod_prev, 3);                        // copy result for iteration

                      matrix_multiplication(B_copy, pLRF_mod_prev, pLRF_prev, 3, 3);  // compute pLRF error
                      vector_subtraction(pLRF, pLRF_prev, dpLRF, 3);

                      dp = sqrt(dpLRF[0] * dpLRF[0]  +  dpLRF[1] * dpLRF[1]  +  dpLRF[2] * dpLRF[2]);

                      if(dp <= eps)
                      {
                        break;        // found solution
                      }

                      matrix_multiplication(B_inv, dpLRF, dpLRF_mod, 3, 3);           // compute correction to pLRF_mod
                      vector_addition(pLRF_mod_prev, dpLRF_mod, pLRF_mod, 3);         // add correction to pLRF_mod
                    }

                    double px_LRF_mod = pLRF_mod[0];                                  // set pLRF_mod components
                    double py_LRF_mod = pLRF_mod[1];
                    double pz_LRF_mod = pLRF_mod[2];

                    double E_mod = sqrt(mass2  +  px_LRF_mod * px_LRF_mod  +  py_LRF_mod * py_LRF_mod  +  pz_LRF_mod * pz_LRF_mod);


                    // *** review this

                    f = fabs(renorm) / (exp(E_mod / Lambda  -  chem_effect)  +  sign); // compute famod


                  }

                  eta_integral += (eta_weight * p_dsigma * f);      // add contribution to integral

                } // eta points (ieta)

                dN_pTdpTdphidy_all[n  +  CORES * iS3D] += (prefactor * degeneracy * eta_integral);

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

        } // particle species (ipart)

        gsl_matrix_free(B_inverse);
        gsl_permutation_free(p);

      } // freezeout cells in the chunk (icell)

      free_2D(B_copy, 3);
      free_2D(B_inv, 3);

    } // number of chunks / cores (n)


    // now perform the reduction over cores
    #pragma omp parallel for collapse(4)
    for(long ipart = 0; ipart < npart; ipart++)
    {
      for(long ipT = 0; ipT < pT_tab_length; ipT++)
      {
        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          for(long iy = 0; iy < y_tab_length; iy++)
          {
            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            double dN_pTdpTdphidy_tmp = 0.0; // reduction variable

            #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
            for(long n = 0; n < CORES; n++)
            {
              dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[n  +  CORES * iS3D];

            } // sum over the cores

            dN_pTdpTdphidy[iS3D] = dN_pTdpTdphidy_tmp;

          } // rapidity points (iy)

        } // azimuthal angle points (iphip)

      } // transverse momentum points (ipT)

    } // particle species (ipart)


    printf("\nfeqmod breaks down for %ld / %ld cells until t = %.3f fm/c\n", breakdown, FO_length, tau_breakdown);
    printf("pl went negative for %ld / %ld cells until t = %.3f fm/c\n\n", pl_negative, FO_length, tau_pl);

    //free memory
    free(dN_pTdpTdphidy_all);
  }




  void EmissionFunctionArray::calculate_dN_dX(int *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Deltaf_Data *df_data)
  {
    printf("computing thermal spacetime distribution from vhydro with df...\n\n");

    // dX = tau.dtau.deta, 2.pi.r.dr.deta or 2.pi.tau.r.dtau.dr.deta
    // only have boost invariance in mind right now so
    // deta = dy and only need to integrate over (pT,phi)

    double prefactor = pow(2.0 * M_PI * hbarC, -3);   // prefactor of CFF

    long FO_chunk = FO_length / CORES;
    long remainder = FO_length  -  CORES * FO_chunk;

    //cout << "Max number of threads = " << omp_get_max_threads() << endl;
    cout << "Number of threads : " << CORES << endl;
    cout << "Chunk size = " << FO_chunk << endl;
    cout << "Remainder cells = " << remainder << endl;

    if(remainder != 0) FO_chunk++;

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];
    double phiWeights[phi_tab_length];

    for(long iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab -> get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);

      phiWeights[iphip] = phi_tab -> get(2, iphip + 1);
    }

    // pT array
    double pTValues[pT_tab_length];
    double pTWeights[pT_tab_length];

    for(long ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab -> get(1, ipT + 1);
      pTWeights[ipT] = pT_tab -> get(2, ipT + 1);
    }

    // y and eta arrays
    double yValues[y_tab_length];
    double etaValues[eta_tab_length];
    double etaWeights[eta_tab_length];

    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(long ieta = 0; ieta < eta_tab_length; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
      for(long iy = 0; iy < y_tab_length; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    // (tau,r,phi) grid (bin midpoints)
    const long taubins = TAU_BINS;
    const long rbins = R_BINS;
    const long phibins = PHIP_BINS;

    double tau_midpoint[taubins];
    double r_midpoint[rbins];
    double phi_midpoint[phibins];
    for(long itau = 0; itau < taubins; itau++)  tau_midpoint[itau] = TAU_MIN + TAU_WIDTH * ((double)itau + 0.5);
    for(long ir = 0; ir < rbins; ir++) r_midpoint[ir] = R_MIN + R_WIDTH * ((double)ir + 0.5);
    for(long iphi = 0; iphi < phibins; iphi++) phi_midpoint[iphi] = PHIP_WIDTH * ((double)iphi + 0.5);


    const int npart = number_of_chosen_particles;

    double dN_taudtaudy[taubins];
    double dN_twopirdrdy[rbins];
    double dN_dphidy[phibins];

    double *dN_taudtaudy_all =  (double*)calloc(CORES * taubins, sizeof(double));
    double *dN_twopirdrdy_all = (double*)calloc(CORES * rbins, sizeof(double));
    double *dN_dphidy_all =  (double*)calloc(CORES * phibins, sizeof(double));
    //double * dN_dy = (double*)calloc(npart, sizeof(double));
    //double ** dN_dydeta = (double**)calloc(npart, sizeof(double*));
    //for(int i = 0; i < npart; i++) dN_dydeta[i] = (double*)calloc(eta_pts, sizeof(double));

    // calculate the spacetime distributions for each particle species
    for(int ipart = 0; ipart < npart; ipart++)
    {
      printf("Starting spacetime distribution %d\n", MCID[ipart]);

      char file_time[255] = "";
      char file_radial[255] = "";
      char file_azimuthal[255] = "";

      sprintf(file_time, "results/continuous/dN_taudtaudy_%d.dat", MCID[ipart]);
      sprintf(file_radial, "results/continuous/dN_2pirdrdy_%d.dat", MCID[ipart]);
      sprintf(file_azimuthal, "results/continuous/dN_dphidy_%d.dat", MCID[ipart]);

      ofstream time_distribution(file_time, ios_base::app);
      ofstream radial_distribution(file_radial, ios_base::app);
      ofstream azimuthal_distribution(file_azimuthal, ios_base::app);

      // rapidity distribution
      //char file_rapidity[255] = "";
      //sprintf(file_rapidity, "results/continuous/dN_dydeta_%d_%dpt.dat", MCID[ipart], eta_tab_length);
      //ofstream rapidity_distribution(file_rapidity, ios_base::out);

      // set particle properties
      double mass = Mass[ipart];              // mass (GeV)
      double mass2 = mass * mass;
      double sign = Sign[ipart];              // quantum statistics sign
      double degeneracy = Degeneracy[ipart];  // spin degeneracy
      double baryon = Baryon[ipart];          // baryon number

      // reset spacetime distributions to zero
      for(long itau = 0; itau < taubins; itau++) dN_taudtaudy[itau] = 0.0;
      for(long ir = 0; ir < rbins; ir++) dN_twopirdrdy[ir] = 0.0;
      for(long iphi = 0; iphi < phibins; iphi++) dN_dphidy[iphi] = 0.0;


      // reset spacetime distributions of chunks to zero
      memset(dN_taudtaudy_all, 0.0, CORES * taubins);
      memset(dN_twopirdrdy_all, 0.0, CORES * rbins);
      memset(dN_dphidy_all, 0.0, CORES * phibins);

      #pragma omp parallel for
      for(long n = 0; n < CORES; n++)
      {
        long endFO = FO_chunk;

        for(long icell = 0; icell < endFO; icell++)  // cell index inside each chunk
        {
		      if((icell == endFO - 1) && (remainder != 0) && (n > remainder - 1)) continue;

		      long icell_glb = n  +  icell * CORES;

          double tau = tau_fo[icell_glb];         // longitudinal proper time
          double x_pos = x_fo[icell_glb];         // x position
          double y_pos = y_fo[icell_glb];         // y position

          double tau2 = tau * tau;
          if(DIMENSION == 3)
          {
            etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
          }

          double dat = dat_fo[icell_glb];         // covariant normal surface vector
          double dax = dax_fo[icell_glb];
          double day = day_fo[icell_glb];
          double dan = dan_fo[icell_glb];         // dan should be 0 for 2+1d

          double ux = ux_fo[icell_glb];           // contravariant fluid velocity
          double uy = uy_fo[icell_glb];           // enforce normalization
          double un = un_fo[icell_glb];
          double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un);

           // skip cells with u.dsigma < 0
          if(ut * dat  +  ux * dax  +  uy * day  +  un * dan <= 0.0) continue;

          double ux2 = ux * ux;                   // useful expressions
          double uy2 = uy * uy;
          double ut2 = ut * ut;
          double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

          double T = T_fo[icell_glb];             // temperature (GeV)
          double P = P_fo[icell_glb];             // equilibrium pressure (GeV/fm^3)
          double E = E_fo[icell_glb];             // energy density (GeV/fm^3)

          double pitt = 0.0;                      // contravariant shear stress tensor pi^munu (GeV/fm^3)
          double pitx = 0.0;                      // enforce orthogonality pi.u = 0
          double pity = 0.0;                      // and tracelessness Tr(pi) = 0
          double pitn = 0.0;
          double pixx = 0.0;
          double pixy = 0.0;
          double pixn = 0.0;
          double piyy = 0.0;
          double piyn = 0.0;
          double pinn = 0.0;

          if(INCLUDE_SHEAR_DELTAF)
          {
            pixx = pixx_fo[icell_glb];
            pixy = pixy_fo[icell_glb];
            pixn = pixn_fo[icell_glb];
            piyy = piyy_fo[icell_glb];
            piyn = piyn_fo[icell_glb];
            pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
            pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
            pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
            pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
            pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
          }

          double bulkPi = 0.0;                    // bulk pressure (GeV/fm^3)

          if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell_glb];

          double muB = 0.0;                       // baryon chemical potential (GeV)
          double alphaB = 0.0;                    // muB / T
          double nB = 0.0;                        // net baryon density (fm^-3)
          double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
          double Vx = 0.0;                        // enforce orthogonality V.u = 0
          double Vy = 0.0;
          double Vn = 0.0;
          double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

          if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
          {
            muB = muB_fo[icell_glb];
            nB = nB_fo[icell_glb];
            Vx = Vx_fo[icell_glb];
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
            Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

            alphaB = muB / T;
            baryon_enthalpy_ratio = nB / (E + P);
          }

          double chem = baryon * alphaB;          // chemical potential term in feq

          // set df coefficients
          deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

          double c0 = df.c0;             // 14 moment coefficients
          double c1 = df.c1;
          double c2 = df.c2;
          double c3 = df.c3;
          double c4 = df.c4;

          double F = df.F;               // Chapman Enskog
          double G = df.G;
          double betabulk = df.betabulk;
          double betaV = df.betaV;
          double betapi = df.betapi;

          // evaluate shear and bulk coefficients
          double shear_coeff = 0.0;
          double bulk0_coeff = 0.0;
          double bulk1_coeff = 0.0;
          double bulk2_coeff = 0.0;

          switch(DF_MODE)
          {
            case 1: // 14 moment
            {
              shear_coeff = 0.5 / (T * T * (E + P));
              bulk0_coeff = c0 - c2;
              bulk1_coeff = c1;
              bulk2_coeff = 4.0 * c2  -  c0;
              break;
            }
            case 2: // Chapman enskog
            {
              shear_coeff = 0.5 / (betapi * T);
              bulk0_coeff = F / (T * T * betabulk);
              bulk1_coeff = G / betabulk;
              bulk2_coeff = 1.0 / (3.0 * T * betabulk);
              break;
            }
            default:
            {
              printf("Error: set df_mode = (1,2) in parameters.dat\n"); exit(-1);
            }
          }


          // compute the dNdy of each freezeout cell (integrating over pT,phi)
          double dN_dy_cell = 0.0;

          for(long ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pTValues[ipT];              // p_T (GeV)
            double mT = sqrt(mass2  +  pT * pT);    // m_T (GeV)
            double mT_over_tau = mT / tau;

            double pT_weight = pTWeights[ipT];

            for(long iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double px = pT * cosphiValues[iphip]; // p^x
              double py = pT * sinphiValues[iphip]; // p^y

              double phi_weight = phiWeights[iphip];

              for(long iy = 0; iy < y_tab_length; iy++)
              {
                double y = yValues[iy];

                double eta_integral = 0.0;

                // sum over eta
                for(long ieta = 0; ieta < eta_tab_length; ieta++)
                {
                  double eta = etaValues[ieta];
                  double eta_weight = etaWeights[ieta];

                  double pt = mT * cosh(y - eta);           // p^\tau (GeV)
                  double pn = mT_over_tau * sinh(y - eta);  // p^\eta (GeV^2)
                  double tau2_pn = tau2 * pn;

                  double pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan); // p.dsigma

                  if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                  double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;  // u.p

                  double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                  double feqbar = 1.0  -  sign * feq;

                  // pi^munu.p_mu.p_nu
                  double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                  + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                  // V^mu.p_mu
                  double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;

                  double df;  // delta f correction

                  switch(DF_MODE)
                  {
                    case 1: // 14 moment
                    {
                      double df_shear = shear_coeff * pimunu_pmu_pnu;
                      double df_bulk = (bulk0_coeff * mass2  +  (bulk1_coeff * baryon  +  bulk2_coeff * pdotu) * pdotu) * bulkPi;
                      double df_diff = (c3 * baryon  +  c4 * pdotu) * Vmu_pmu;

                      df = feqbar * (df_shear + df_bulk + df_diff);

                      break;
                    }
                    case 2: // Chapman enskog
                    {
                      double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu - mass2 / pdotu)) * bulkPi;
                      double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;

                      df = feqbar * (df_shear + df_bulk + df_diff);
                      break;
                    }
                    default:
                    {
                      printf("Error: set df_mode = (1,2) in parameters.dat\n"); exit(-1);
                    }
                  } // DF_MODE

                  if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                  double f = feq * (1.0 + df);  // distribution function with linear df correction

                  eta_integral += (pdotdsigma * f);

                  //dN_dydeta[ipart][ieta] += (pT_weight * phi_weight * prefactor * degeneracy * pdotdsigma * f / eta_weight);


                } // eta points (ieta)

                dN_dy_cell += (pT_weight * phi_weight * prefactor * degeneracy * eta_integral);

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

          //dN_dy[ipart] += dN_dy_cell;


          // now determine which spacetime bin the freezeout cell lies
          double r = sqrt(x_pos * x_pos  +  y_pos * y_pos);
          double phi = atan2(y_pos, x_pos);
          if(phi < 0.0) phi += two_pi;

          // bin index
          long itau = (int)floor((tau - TAU_MIN) / TAU_WIDTH);
          long ir = (int)floor((r - R_MIN) / R_WIDTH);
          long iphi = (int)floor(phi / PHIP_WIDTH);

          // add dNdy to the corresponding bin(s)
          if(itau >= 0 && itau < taubins)
          {
            long itau_all = itau  +  n * taubins;
            dN_taudtaudy_all[itau_all] += dN_dy_cell;
          }

          if(ir >= 0 && ir < rbins)
          {
            long ir_all = ir  +  n * rbins;
            dN_twopirdrdy_all[ir_all] += dN_dy_cell;
          }

          if(iphi >= 0 && iphi < phibins)
          {
            long iphi_all = iphi  +  n * phibins;
            dN_dphidy_all[iphi_all] += dN_dy_cell;
          }


        } // freezeout cells (icell)

      } // chunks or cores (n)


      // write spacetime distributions to file (normalize by the binwidths)
      for(long ir = 0; ir < rbins; ir++)
      {
        // sum over the cores
        for(long n = 0; n < CORES; n++)
        {
          long ir_all = ir  +  n * rbins;
          dN_twopirdrdy[ir] += dN_twopirdrdy_all[ir_all];
        }

        double r_mid = r_midpoint[ir];

        radial_distribution << setprecision(6) << scientific << r_mid << "\t" << dN_twopirdrdy[ir] / (two_pi * r_mid * R_WIDTH)  << "\n";
      }

      for(long itau = 0; itau < taubins; itau++)
      {
        // sum over the cores
        for(long n = 0; n < CORES; n++)
        {
          long itau_all = itau  +  n * taubins;
          dN_taudtaudy[itau] += dN_taudtaudy_all[itau_all];

        } // sum over the cores

        double tau_mid = tau_midpoint[itau];

        time_distribution << setprecision(6) << scientific << tau_mid << "\t" << dN_taudtaudy[itau] / (tau_mid * TAU_WIDTH) << "\n";
      }

      for(long iphi = 0; iphi < phibins; iphi++)
      {
        // sum over the cores
        for(long n = 0; n < CORES; n++)
        {
          long iphi_all = iphi  +  n * phibins;
          dN_dphidy[iphi] += dN_dphidy_all[iphi_all];
        }

        double phi_mid = phi_midpoint[iphi];

        azimuthal_distribution << setprecision(6) << scientific << phi_mid << "\t" << dN_dphidy[iphi] / PHIP_WIDTH << "\n";
      }

      // close files
      time_distribution.close();
      radial_distribution.close();
      azimuthal_distribution.close();

      // write rapidity distribution to file
      // for(int ieta = 0; ieta < eta_pts; ieta++)
      // {
      //   rapidity_distribution << setprecision(6) << scientific << etaValues[ieta] << "\t" << dN_dydeta[ipart][ieta] << "\n";
      // }
      // rapidity_distribution.close();

    } // hadron species (ipart)

    // for(int i = 0; i < npart; i++)
    // {
    //   printf("dN_dy = %lf\n", dN_dy[i]);
    // }

    //free_2D(dN_dydeta, npart);
    //free(dN_dy);
    free(dN_taudtaudy_all);
    free(dN_twopirdrdy_all);
    free(dN_dphidy_all);

  }


  void EmissionFunctionArray::calculate_dN_dX_feqmod(int *MCID, double *Mass, double *Sign, double *Degeneracy, double *Baryon,
  double *T_fo, double *P_fo, double *E_fo, double *tau_fo, double *x_fo, double *y_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
  double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo,
  double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *bulkPi_fo,
  double *muB_fo, double *nB_fo, double *Vx_fo, double *Vy_fo, double *Vn_fo, Gauss_Laguerre * laguerre, Deltaf_Data *df_data)
  {
    printf("computing thermal spacetime distribution from vhydro with feqmod...\n\n");

    // dX = tau.dtau.deta, 2.pi.r.dr.deta or 2.pi.tau.r.dtau.dr.deta
    // only have boost invariance in mind right now so
    // deta = dy and only need to integrate over (pT,phi)

    // you compute the spatial distributions by setting up a spacetime
    // grid and binning the freezeout cell's mean particle number

    double prefactor = pow(2.0 * M_PI * hbarC, -3);   // prefactor of CFF
    long FO_chunk = FO_length / CORES;
    long remainder = FO_length  -  CORES * FO_chunk;

    cout << "Number of threads : " << CORES << endl;
    cout << "Chunk size = " << FO_chunk << endl;
    cout << "Remainder cells = " << remainder << endl;

    if(remainder != 0) FO_chunk++;

    double detA_min = DETA_MIN;   // default value for minimum detA
    int breakdown = 0;            // # times feqmod breaks down

    // phi arrays
    double cosphiValues[phi_tab_length];
    double sinphiValues[phi_tab_length];
    double phiWeights[phi_tab_length];

    for(int iphip = 0; iphip < phi_tab_length; iphip++)
    {
      double phi = phi_tab -> get(1, iphip + 1);
      cosphiValues[iphip] = cos(phi);
      sinphiValues[iphip] = sin(phi);

      phiWeights[iphip] = phi_tab -> get(2, iphip + 1);
    }

    // pT array
    double pTValues[pT_tab_length];
    double pTWeights[pT_tab_length];

    for(int ipT = 0; ipT < pT_tab_length; ipT++)
    {
      pTValues[ipT] = pT_tab -> get(1, ipT + 1);
      pTWeights[ipT] = pT_tab -> get(2, ipT + 1);
    }

    // y and eta arrays
    double yValues[y_tab_length];
    double etaValues[eta_tab_length];
    double etaWeights[eta_tab_length];


    if(DIMENSION == 2)
    {
      yValues[0] = 0.0;

      for(int ieta = 0; ieta < eta_tab_length; ieta++)
      {
        etaValues[ieta] = eta_tab->get(1, ieta + 1);
        etaWeights[ieta] = eta_tab->get(2, ieta + 1);
      }
    }
    else if(DIMENSION == 3)
    {
      etaValues[0] = 0.0;       // below, will load eta_fo
      etaWeights[0] = 1.0; // 1.0 for 3+1d
      for(int iy = 0; iy < y_tab_length; iy++)
      {
        yValues[iy] = y_tab->get(1, iy + 1);
      }
    }

    // (tau,r,phi) grid (bin midpoints)
    const long taubins = TAU_BINS;
    const long rbins = R_BINS;
    const long phibins = PHIP_BINS;

    double tau_midpoint[taubins];
    double r_midpoint[rbins];
    double phi_midpoint[phibins];
    for(long itau = 0; itau < taubins; itau++)  tau_midpoint[itau] = TAU_MIN + TAU_WIDTH * ((double)itau + 0.5);
    for(long ir = 0; ir < rbins; ir++) r_midpoint[ir] = R_MIN + R_WIDTH * ((double)ir + 0.5);
    for(long iphi = 0; iphi < phibins; iphi++) phi_midpoint[iphi] = PHIP_WIDTH * ((double)iphi + 0.5);

    const int npart = number_of_chosen_particles;

    double dN_taudtaudy[taubins];
    double dN_twopirdrdy[rbins];
    double dN_dphidy[phibins];

    double *dN_taudtaudy_all =  (double*)calloc(CORES * taubins, sizeof(double));
    double *dN_twopirdrdy_all = (double*)calloc(CORES * rbins, sizeof(double));
    double *dN_dphidy_all =  (double*)calloc(CORES * phibins, sizeof(double));


    /// gauss laguerre roots
    const int pbar_pts = laguerre->points;

    double * pbar_root1 = laguerre->root[1];
    double * pbar_root2 = laguerre->root[2];

    double * pbar_weight1 = laguerre->weight[1];
    double * pbar_weight2 = laguerre->weight[2];

    // double * dN_dy = (double*)calloc(npart, sizeof(double));
    // double ** dN_dydeta = (double**)calloc(npart, sizeof(double*));
    // for(int i = 0; i < npart; i++) dN_dydeta[i] = (double*)calloc(eta_tab_length, sizeof(double));

    // calculate the spacetime distributions for each particle species
    for(int ipart = 0; ipart < npart; ipart++)
    {
      // make the files
       printf("Starting spacetime distribution %d\n", MCID[ipart]);

      char file_time[255] = "";
      char file_radial[255] = "";
      char file_azimuthal[255] = "";

      sprintf(file_time, "results/continuous/dN_taudtaudy_%d.dat", MCID[ipart]);
      sprintf(file_radial, "results/continuous/dN_2pirdrdy_%d.dat", MCID[ipart]);
      sprintf(file_azimuthal, "results/continuous/dN_dphidy_%d.dat", MCID[ipart]);

      ofstream time_distribution(file_time, ios_base::app);
      ofstream radial_distribution(file_radial, ios_base::app);
      ofstream azimuthal_distribution(file_azimuthal, ios_base::app);

      // rapidity distribution
      //char file_rapidity[255] = "";
      //sprintf(file_rapidity, "results/continuous/dN_dydeta_%d_%dpt.dat", MCID[ipart], eta_tab_length);
      //ofstream rapidity_distribution(file_rapidity, ios_base::out);

      // set particle properties
      double mass = Mass[ipart];              // mass (GeV)
      double mass2 = mass * mass;
      double sign = Sign[ipart];              // quantum statistics sign
      double degeneracy = Degeneracy[ipart];  // spin degeneracy
      double baryon = Baryon[ipart];          // baryon number

      // reset spacetime distributions to zero
      for(long itau = 0; itau < taubins; itau++) dN_taudtaudy[itau] = 0.0;
      for(long ir = 0; ir < rbins; ir++) dN_twopirdrdy[ir] = 0.0;
      for(long iphi = 0; iphi < phibins; iphi++) dN_dphidy[iphi] = 0.0;


      // reset spacetime distributions of chunks to zero
      memset(dN_taudtaudy_all, 0.0, CORES * taubins);
      memset(dN_twopirdrdy_all, 0.0, CORES * rbins);
      memset(dN_dphidy_all, 0.0, CORES * phibins);


      #pragma omp parallel for
      for(long n = 0; n < CORES; n++)
      {
        double ** A_copy = (double**)calloc(3, sizeof(double*));
        for(int i = 0; i < 3; i++) A_copy[i] = (double*)calloc(3, sizeof(double));

        double ** A_inv = (double**)calloc(3, sizeof(double*));
        for(int i = 0; i < 3; i++) A_inv[i] = (double*)calloc(3, sizeof(double));

        long endFO = FO_chunk;

        for(long icell = 0; icell < endFO; icell++)
        {
          if((icell == endFO - 1) && (remainder != 0) && (n > remainder - 1)) continue;

          long icell_glb = n  +  icell * CORES;   // global cell index

          double tau = tau_fo[icell_glb];         // longitudinal proper time
          double x_pos = x_fo[icell_glb];         // x position
          double y_pos = y_fo[icell_glb];         // y position

          double tau2 = tau * tau;
          if(DIMENSION == 3)
          {
            etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
          }

          double dat = dat_fo[icell_glb];         // covariant normal surface vector
          double dax = dax_fo[icell_glb];
          double day = day_fo[icell_glb];
          double dan = dan_fo[icell_glb];         // dan should be 0 for 2+1d

          double ux = ux_fo[icell_glb];           // contravariant fluid velocity
          double uy = uy_fo[icell_glb];           // enforce normalization
          double un = un_fo[icell_glb];
          double ut = sqrt(1.0  +  ux * ux  +  uy * uy  +  tau2 * un * un);

          double udsigma = ut * dat  +  ux * dax  +  uy * day  +  un * dan;  // u.dsigma / eta_weight

          if(udsigma <= 0.0) continue;            // skip cells with u.dsigma < 0

          double ux2 = ux * ux;                   // useful expressions
          double uy2 = uy * uy;
          double ut2 = ut * ut;
          double uperp = sqrt(ux * ux  +  uy * uy);
          double utperp = sqrt(1.0  +  ux * ux  +  uy * uy);

          double T = T_fo[icell_glb];             // temperature (GeV)
          double P = P_fo[icell_glb];             // equilibrium pressure (GeV/fm^3)
          double E = E_fo[icell_glb];             // energy density (GeV/fm^3)

          double pitt = 0.0;                      // contravariant shear stress tensor pi^munu (GeV/fm^3)
          double pitx = 0.0;                      // enforce orthogonality pi.u = 0
          double pity = 0.0;                      // and tracelessness Tr(pi) = 0
          double pitn = 0.0;
          double pixx = 0.0;
          double pixy = 0.0;
          double pixn = 0.0;
          double piyy = 0.0;
          double piyn = 0.0;
          double pinn = 0.0;

          if(INCLUDE_SHEAR_DELTAF)
          {
            pixx = pixx_fo[icell_glb];
            pixy = pixy_fo[icell_glb];
            pixn = pixn_fo[icell_glb];
            piyy = piyy_fo[icell_glb];
            piyn = piyn_fo[icell_glb];
            pinn = (pixx * (ux2 - ut2)  +  piyy * (uy2 - ut2)  +  2.0 * (pixy * ux * uy  +  tau2 * un * (pixn * ux  +  piyn * uy))) / (tau2 * utperp * utperp);
            pitn = (pixn * ux  +  piyn * uy  +  tau2 * pinn * un) / ut;
            pity = (pixy * ux  +  piyy * uy  +  tau2 * piyn * un) / ut;
            pitx = (pixx * ux  +  pixy * uy  +  tau2 * pixn * un) / ut;
            pitt = (pitx * ux  +  pity * uy  +  tau2 * pitn * un) / ut;
          }

          double bulkPi = 0.0;                    // bulk pressure (GeV/fm^3)

          if(INCLUDE_BULK_DELTAF) bulkPi = bulkPi_fo[icell_glb];

          double muB = 0.0;                       // baryon chemical potential (GeV)
          double alphaB = 0.0;                    // muB / T
          double nB = 0.0;                        // net baryon density (fm^-3)
          double Vt = 0.0;                        // contravariant net baryon diffusion V^mu (fm^-3)
          double Vx = 0.0;                        // enforce orthogonality V.u = 0
          double Vy = 0.0;
          double Vn = 0.0;
          double baryon_enthalpy_ratio = 0.0;     // nB / (E + P)

          if(INCLUDE_BARYON && INCLUDE_BARYONDIFF_DELTAF)
          {
            muB = muB_fo[icell_glb];
            nB = nB_fo[icell_glb];
            Vx = Vx_fo[icell_glb];
            Vy = Vy_fo[icell_glb];
            Vn = Vn_fo[icell_glb];
            Vt = (Vx * ux  +  Vy * uy  +  tau2 * Vn * un) / ut;

            alphaB = muB / T;
            baryon_enthalpy_ratio = nB / (E + P);
          }

          // regulate bulk pressure if goes out of bounds given
          // by Jonah's feqmod to avoid gsl interpolation errors
          if(DF_MODE == 4)
          {
            double bulkPi_over_Peq_max = df_data->bulkPi_over_Peq_max;

            if(bulkPi <= - P) bulkPi = - (1.0 - 1.e-5) * P;
            else if(bulkPi / P >= bulkPi_over_Peq_max) bulkPi = P * (bulkPi_over_Peq_max - 1.e-5);
          }

          // set df coefficients
          deltaf_coefficients df = df_data->evaluate_df_coefficients(T, muB, E, P, bulkPi);

          // modified coefficients (Mike / Jonah)
          double F = df.F;
          double G = df.G;
          double betabulk = df.betabulk;
          double betaV = df.betaV;
          double betapi = df.betapi;
          double lambda = df.lambda;
          double z = df.z;
          double delta_lambda = df.delta_lambda;
          double delta_z = df.delta_z;

          // milne basis class
          Milne_Basis basis_vectors(ut, ux, uy, un, uperp, utperp, tau);
          basis_vectors.test_orthonormality(tau2);

          double Xt = basis_vectors.Xt;   double Yx = basis_vectors.Yx;
          double Xx = basis_vectors.Xx;   double Yy = basis_vectors.Yy;
          double Xy = basis_vectors.Xy;   double Zt = basis_vectors.Zt;
          double Xn = basis_vectors.Xn;   double Zn = basis_vectors.Zn;

          // shear stress class
          Shear_Stress pimunu(pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn);
          pimunu.test_pimunu_orthogonality_and_tracelessness(ut, ux, uy, un, tau2);
          pimunu.boost_pimunu_to_lrf(basis_vectors, tau2);

          // baryon diffusion class
          Baryon_Diffusion Vmu(Vt, Vx, Vy, Vn);
          Vmu.test_Vmu_orthogonality(ut, ux, uy, un, tau2);
          Vmu.boost_Vmu_to_lrf(basis_vectors, tau2);


          // modified temperature / chemical potential
          double T_mod = T;
          double alphaB_mod = alphaB;

          if(DF_MODE == 3)
          {
            T_mod = T  +  bulkPi * F / betabulk;
            alphaB_mod = alphaB  +  bulkPi * G / betabulk;
          }

          double chem = baryon * alphaB;          // chemical potential term in feq
          double chem_mod = baryon * alphaB_mod;  // chemical potential term in feqmod

          // linearized Chapman Enskog df coefficients (for Mike only)
          double shear_coeff = 0.5 / (betapi * T);      // Jonah linear df also shares shear coeff
          double bulk0_coeff = F / (T * T * betabulk);
          double bulk1_coeff = G / betabulk;
          double bulk2_coeff = 1.0 / (3.0 * T * betabulk);

          // pimunu and Vmu LRF components
          double pixx_LRF = pimunu.pixx_LRF;
          double pixy_LRF = pimunu.pixy_LRF;
          double pixz_LRF = pimunu.pixz_LRF;
          double piyy_LRF = pimunu.piyy_LRF;
          double piyz_LRF = pimunu.piyz_LRF;
          double pizz_LRF = pimunu.pizz_LRF;

          double Vx_LRF = Vmu.Vx_LRF;
          double Vy_LRF = Vmu.Vy_LRF;
          double Vz_LRF = Vmu.Vz_LRF;

          // local momentum transformation matrix Mij = Aij
          // Aij = ideal + shear + bulk is symmetric
          // Mij is not symmetric if include baryon diffusion (leave for future work)

          // modified coefficients in Aij
          double shear_mod = 0.5 / betapi;
          double bulk_mod = bulkPi / (3.0 * betabulk);

          if(DF_MODE == 4) bulk_mod = lambda;

          // Aij elements
          double Axx = 1.0  +  pixx_LRF * shear_mod  +  bulk_mod;
          double Axy = pixy_LRF * shear_mod;
          double Axz = pixz_LRF * shear_mod;
          double Ayy = 1.0  +  piyy_LRF * shear_mod  +  bulk_mod;
          double Ayz = piyz_LRF * shear_mod;
          double Azz = 1.0  +  pizz_LRF * shear_mod  +  bulk_mod;

          double detA = Axx * (Ayy * Azz  -  Ayz * Ayz)  -  Axy * (Axy * Azz  -  Ayz * Axz)  +  Axz * (Axy * Ayz  -  Ayy * Axz);
          double detA_bulk_two_thirds = pow(1.0 + bulk_mod, 2);

          // determine if feqmod breaks down
          bool feqmod_breaks_down = does_feqmod_breakdown(MASS_PION0, T, F, bulkPi, betabulk, detA, detA_min, z, laguerre, DF_MODE, 0, T, F, betabulk);

          // set Aij matrix
          double A[] = {Axx, Axy, Axz,
                        Axy, Ayy, Ayz,
                        Axz, Ayz, Azz};           // gsl matrix format

          A_copy[0][0] = Axx;  A_copy[0][1] = Axy;  A_copy[0][2] = Axz;
          A_copy[1][0] = Axy;  A_copy[1][1] = Ayy;  A_copy[1][2] = Ayz;
          A_copy[2][0] = Axz;  A_copy[2][1] = Ayz;  A_copy[2][2] = Azz;

          // compute Aij^-1 using LUP decomposition
          int s;
          gsl_matrix_view LU = gsl_matrix_view_array(A, 3, 3);
          gsl_permutation * p = gsl_permutation_calloc(3);
          gsl_linalg_LU_decomp(&LU.matrix, p, &s);

          gsl_matrix * A_inverse = gsl_matrix_alloc(3, 3);
          gsl_linalg_LU_invert(&LU.matrix, p, A_inverse);

          for(int i = 0; i < 3; i++)
          {
            for(int j = 0; j < 3; j++)
            {
              A_inv[i][j] = gsl_matrix_get(A_inverse, i, j);
            }
          }


          // prefactors for equilibrium, linear bulk correction and modified densities (Mike's feqmod)
          double neq_fact = T * T * T / two_pi2_hbarC3;
          double dn_fact = bulkPi / betabulk;
          double J20_fact = T * neq_fact;
          double N10_fact = neq_fact;
          double nmod_fact = T_mod * T_mod * T_mod / two_pi2_hbarC3;

          // if(feqmod_breaks_down)
          // {
          //   #pragma omp critical
          //   {
          //     breakdown++;
          //     cout << setw(5) << setprecision(4) << "feqmod breaks down for " << breakdown << " / " << FO_length << " cells  (cell = " << icell << ", tau = " << tau << " fm/c, " << ", detA = " << detA << ", detA_min = " << detA_min << ")" << endl;
          //   }
          // }

          // rescale eta by detA if modified momentum space elements are shrunk
          // for integrating modified distribution with narrow (y-eta) distributions
          // note: this only works for boost invariant surfaces, where dsigma is always orthogonal to eta direction (dan = 0)
          double eta_scale = 1.0;
          if(detA > detA_min && DIMENSION == 2)
          {
            eta_scale = detA / detA_bulk_two_thirds;
          }

          // compute the modified renormalization factor
          double renorm = 1.0;

          if(INCLUDE_BULK_DELTAF)
          {
            if(DF_MODE == 3)
            {
              double mbar = mass / T;
              double mbar_mod = mass / T_mod;

              double neq = neq_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

              double N10 = baryon * N10_fact * degeneracy * GaussThermal(J10_int, pbar_root1, pbar_weight1, pbar_pts, mbar, alphaB, baryon, sign);

              double J20 = J20_fact * degeneracy * GaussThermal(J20_int, pbar_root2, pbar_weight2, pbar_pts, mbar, alphaB, baryon, sign);

              double n_linear = neq  +  dn_fact * (neq  +  N10 * G  +  J20 * F / T / T);

              double n_mod = nmod_fact * degeneracy * GaussThermal(neq_int, pbar_root1, pbar_weight1, pbar_pts, mbar_mod, alphaB_mod, baryon, sign);

              renorm = n_linear / n_mod;
            }
            else if(DF_MODE == 4)
            {
              renorm = z;
            }
          }

          if(DIMENSION == 2)
          {
            renorm /= detA_bulk_two_thirds;
          }
          else if(DIMENSION == 3)
          {
            renorm /= detA;
          }

          if((std::isnan(renorm) || std::isinf(renorm)))
          {
            cout << "Error: renormalization factor is " << renorm << endl;
            continue;
          }

          // compute the dNdy of each freezeout cell (integrating over pT,phi,y)
          double dN_dy_cell = 0.0;

          for(int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pTValues[ipT];              // p_T (GeV)
            double mT = sqrt(mass2  +  pT * pT);    // m_T (GeV)
            double mT_over_tau = mT / tau;

            double pT_weight = pTWeights[ipT];

            for(int iphip = 0; iphip < phi_tab_length; iphip++)
            {
              double px = pT * cosphiValues[iphip]; // p^x
              double py = pT * sinphiValues[iphip]; // p^y

              double phi_weight = phiWeights[iphip];

              for(int iy = 0; iy < y_tab_length; iy++)
              {
                double y = yValues[iy];

                double eta_integral = 0.0;

                // sum over eta
                for(int ieta = 0; ieta < eta_tab_length; ieta++)
                {
                  double eta = etaValues[ieta];
                  double eta_weight = etaWeights[ieta];

                  bool feqmod_breaks_down_narrow = false;

                  if(DIMENSION == 3 && !feqmod_breaks_down)
                  {
                    if(detA < 0.01 && fabs(y - eta) < detA)
                    {
                      feqmod_breaks_down_narrow = true;
                    }
                  }

                  double f;              // feqmod (if breakdown do feq(1+df))
                  double pdotdsigma;

                  // calculate feqmod
                  if(feqmod_breaks_down || feqmod_breaks_down_narrow)
                  {
                    double pt = mT * cosh(y - eta);           // p^\tau (GeV)
                    double pn = mT_over_tau * sinh(y - eta);  // p^\eta (GeV^2)
                    double tau2_pn = tau2 * pn;

                    pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan); // p.dsigma

                    if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow

                    if(DF_MODE == 3)
                    {
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      double feq = 1.0 / (exp(pdotu / T  -  chem) + sign);
                      double feqbar = 1.0  -  sign * feq;

                       // pi^munu.p_mu.p_nu
                      double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                       + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                      // V^mu.p_mu
                      double Vmu_pmu = Vt * pt  -  Vx * px  -  Vy * py  -  Vn * tau2_pn;

                      double df_shear = shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = (bulk0_coeff * pdotu  +  bulk1_coeff * baryon +  bulk2_coeff * (pdotu  -  mass2 / pdotu)) * bulkPi;
                      double df_diff = (baryon_enthalpy_ratio  -  baryon / pdotu) * Vmu_pmu / betaV;

                      double df = feqbar * (df_shear + df_bulk + df_diff);

                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                      f = feq * (1.0 + df);
                    }
                    else if(DF_MODE == 4)
                    {
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;
                      double feq = 1.0 / (exp(pdotu / T) + sign);
                      double feqbar = 1.0  -  sign * feq;

                       // pi^munu.p_mu.p_nu
                      double pimunu_pmu_pnu = pitt * pt * pt  +  pixx * px * px  +  piyy * py * py  +  pinn * tau2_pn * tau2_pn
                       + 2.0 * (-(pitx * px  +  pity * py) * pt  +  pixy * px * py  +  tau2_pn * (pixn * px  +  piyn * py  -  pitn * pt));

                      double df_shear = feqbar * shear_coeff * pimunu_pmu_pnu / pdotu;
                      double df_bulk = delta_z  -  3.0 * delta_lambda  +  feqbar * delta_lambda * (pdotu  -  mass2 / pdotu) / T;

                      double df = df_shear + df_bulk;

                      if(REGULATE_DELTAF) df = max(-1.0, min(df, 1.0)); // regulate df

                      f = feq * (1.0 + df);
                    }
                  } // feqmod breaks down
                  else
                  {
                    double pt = mT * cosh(y - eta_scale * eta);           // p^\tau (GeV)
                    double pn = mT_over_tau * sinh(y - eta_scale * eta);  // p^\eta (GeV^2)
                    double tau2_pn = tau2 * pn;

                    pdotdsigma = eta_weight * (pt * dat  +  px * dax  +  py * day  +  pn * dan); // p.dsigma

                    if(OUTFLOW && pdotdsigma <= 0.0) continue;  // enforce outflow
                    // LRF momentum components pi_LRF = - Xi.p
                    double px_LRF = -Xt * pt  +  Xx * px  +  Xy * py  +  Xn * tau2_pn;
                    double py_LRF = Yx * px  +  Yy * py;
                    double pz_LRF = -Zt * pt  +  Zn * tau2_pn;

                    double pLRF[3] = {px_LRF, py_LRF, pz_LRF};
                    double pLRF_mod[3];

                    // LUP method with iterations

                    double pLRF_prev[3];
                    double pLRF_mod_prev[3];

                    double dpLRF[3];
                    double dpLRF_mod[3];

                    matrix_multiplication(A_inv, pLRF, pLRF_mod, 3, 3);   // evaluate pLRF_mod = A^-1.pLRF at least once

                    double dp;                                            // |pLRF| error (i.e. dp = pLRF - A.pLRF_mod)
                    double epsilon = 1.e-16;

                    // iterate the solution p_LRF_mod (sometimes it gets no improvement)
                    for(int i = 0; i < 5; i++)
                    {
                      vector_copy(pLRF_mod, pLRF_mod_prev, 3);                        // copy solution for iteration
                      matrix_multiplication(A_copy, pLRF_mod_prev, pLRF_prev, 3, 3);  // check |pLRF| error
                      vector_subtraction(pLRF, pLRF_prev, dpLRF, 3);

                      dp = sqrt(dpLRF[0] * dpLRF[0]  +  dpLRF[1] * dpLRF[1]  +  dpLRF[2] * dpLRF[2]);

                      if(dp <= epsilon) break;

                      matrix_multiplication(A_inv, dpLRF, dpLRF_mod, 3, 3);           // compute correction to pLRF_mod
                      vector_addition(pLRF_mod_prev, dpLRF_mod, pLRF_mod, 3);         // add correction to pLRF_mod
                    }

                    double E_mod = sqrt(mass2  +  pLRF_mod[0] * pLRF_mod[0]  +  pLRF_mod[1] * pLRF_mod[1]  +  pLRF_mod[2] * pLRF_mod[2]);

                    f = fabs(renorm) / (exp(E_mod / T_mod  -  chem_mod) + sign); // feqmod
                  }

                  eta_integral += (pdotdsigma * f);

                  //dN_dydeta[ipart][ieta] += (pT_weight * phi_weight * prefactor * degeneracy * pdotdsigma * f / eta_weight);

                } // eta points (ieta)

                dN_dy_cell += (pT_weight * phi_weight * prefactor * degeneracy * eta_integral);

              } // rapidity points (iy)

            } // azimuthal angle points (iphip)

          } // transverse momentum points (ipT)

          //dN_dy[ipart] += dN_dy_cell;

          gsl_matrix_free(A_inverse);
          gsl_permutation_free(p);


          // now determine which spacetime bin the freezeout cell lies
          double r = sqrt(x_pos * x_pos  +  y_pos * y_pos);
          double phi = atan2(y_pos, x_pos);
          if(phi < 0.0) phi += two_pi;

          // bin index
          long itau = (int)floor((tau - TAU_MIN) / TAU_WIDTH);
          long ir = (int)floor((r - R_MIN) / R_WIDTH);
          long iphi = (int)floor(phi / PHIP_WIDTH);

          // add dNdy to the corresponding bin(s)
          if(itau >= 0 && itau < taubins)
          {
            long itau_all = itau  +  n * taubins;
            dN_taudtaudy_all[itau_all] += dN_dy_cell;
          }

          if(ir >= 0 && ir < rbins)
          {
            long ir_all = ir  +  n * rbins;
            dN_twopirdrdy_all[ir_all] += dN_dy_cell;
          }

          if(iphi >= 0 && iphi < phibins)
          {
            long iphi_all = iphi  +  n * phibins;
            dN_dphidy_all[iphi_all] += dN_dy_cell;
          }


        } // freezeout cells (icell)


        free_2D(A_copy, 3);
        free_2D(A_inv, 3);

      } // chunks or cores (n)


      // write spacetime distributions to file (normalize by the binwidths)
      for(long ir = 0; ir < rbins; ir++)
      {
        for(long n = 0; n < CORES; n++)
        {
          long ir_all = ir  +  n * rbins;
          dN_twopirdrdy[ir] += dN_twopirdrdy_all[ir_all];
        } // sum over the cores

        double r_mid = r_midpoint[ir];

        radial_distribution << setprecision(6) << scientific << r_mid << "\t" << dN_twopirdrdy[ir] / (two_pi * r_mid * R_WIDTH)  << "\n";
      }

      for(long itau = 0; itau < taubins; itau++)
      {
        for(long n = 0; n < CORES; n++)
        {
          long itau_all = itau  +  n * taubins;
          dN_taudtaudy[itau] += dN_taudtaudy_all[itau_all];

        } // sum over the cores

        double tau_mid = tau_midpoint[itau];

        time_distribution << setprecision(6) << scientific << tau_mid << "\t" << dN_taudtaudy[itau] / (tau_mid * TAU_WIDTH) << "\n";
      }

      for(long iphi = 0; iphi < phibins; iphi++)
      {
        for(long n = 0; n < CORES; n++)
        {
          long iphi_all = iphi  +  n * phibins;
          dN_dphidy[iphi] += dN_dphidy_all[iphi_all];
        } // sum over the cores

        double phi_mid = phi_midpoint[iphi];

        azimuthal_distribution << setprecision(6) << scientific << phi_mid << "\t" << dN_dphidy[iphi] / PHIP_WIDTH << "\n";
      }

      // close files
      time_distribution.close();
      radial_distribution.close();
      azimuthal_distribution.close();


      // write rapidity distribution to file
      // for(int ieta = 0; ieta < eta_tab_length; ieta++)
      // {
      //   rapidity_distribution << setprecision(6) << scientific << etaValues[ieta] << "\t" << dN_dydeta[ipart][ieta] << "\n";
      // }
      // rapidity_distribution.close();

    } // hadron species (ipart)


    // free memory
    free(dN_taudtaudy_all);
    free(dN_twopirdrdy_all);
    free(dN_dphidy_all);

    // for(int i = 0; i < npart; i++)
    // {
    //   printf("dN_dy = %lf\n", dN_dy[i]);
    // }
    // free_2D(dN_dydeta, npart);
    // free(dN_dy);
  }




    void EmissionFunctionArray::calculate_dN_pTdpTdphidy_VAH_PL(double *Mass, double *Sign, double *Degeneracy,
      double *tau_fo, double *eta_fo, double *ux_fo, double *uy_fo, double *un_fo,
      double *dat_fo, double *dax_fo, double *day_fo, double *dan_fo, double *T_fo,
      double *pitt_fo, double *pitx_fo, double *pity_fo, double *pitn_fo, double *pixx_fo, double *pixy_fo, double *pixn_fo, double *piyy_fo, double *piyn_fo, double *pinn_fo,
      double *bulkPi_fo, double *Wx_fo, double *Wy_fo, double *Lambda_fo, double *aL_fo, double *c0_fo, double *c1_fo, double *c2_fo, double *c3_fo, double *c4_fo)
      {
        double prefactor = 1.0 / (8.0 * (M_PI * M_PI * M_PI)) / hbarC / hbarC / hbarC;
        int FO_chunk = 10000;

        double trig_phi_table[phi_tab_length][2]; // 2: 0,1-> cos,sin
        for (int j = 0; j < phi_tab_length; j++)
        {
          double phi = phi_tab->get(1,j+1);
          trig_phi_table[j][0] = cos(phi);
          trig_phi_table[j][1] = sin(phi);
        }

        //fill arrays with values points in pT and y
        double pTValues[pT_tab_length];
        for (int ipT = 0; ipT < pT_tab_length; ipT++) pTValues[ipT] = pT_tab->get(1, ipT + 1);

        // fill arrays with values points in y and eta
        int y_pts = y_tab_length;    // defaults pts for 3+1d
        int eta_pts = 1;

        if(DIMENSION == 2)
        {
          y_pts = 1;
          eta_pts = eta_tab_length;
        }

        double yValues[y_pts];
        double etaValues[eta_pts];
        double etaWeights[eta_pts];

        double delta_eta = (eta_tab->get(1,2)) - (eta_tab->get(1,1));  // assume uniform grid

        if(DIMENSION == 2)
        {
          yValues[0] = 0.0;
          for(int ieta = 0; ieta < eta_pts; ieta++)
          {
            etaValues[ieta] = eta_tab->get(1, ieta + 1);
            etaWeights[ieta] = (eta_tab->get(2, ieta + 1)) * delta_eta;
            //cout << etaValues[ieta] << "\t" << etaWeights[ieta] << endl;
          }
        }
        else if(DIMENSION == 3)
        {
          etaValues[0] = 0.0;       // below, will load eta_fo
          etaWeights[0] = 1.0; // 1.0 for 3+1d
          for (int iy = 0; iy < y_pts; iy++) yValues[iy] = y_tab->get(1, iy + 1);
        }

        //declare a huge array of size npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length
        //to hold the spectra for each surface cell in a chunk, for all particle species
        int npart = number_of_chosen_particles;
        double *dN_pTdpTdphidy_all;
        dN_pTdpTdphidy_all = (double*)calloc(npart * FO_chunk * pT_tab_length * phi_tab_length * y_tab_length, sizeof(double));

        //loop over bite size chunks of FO surface
        for (int n = 0; n < (FO_length / FO_chunk) + 1; n++)
        {
          printf("Progress : Finished chunk %d of %ld \n", n, FO_length / FO_chunk);
          int endFO = FO_chunk;
          if (n == (FO_length / FO_chunk)) endFO = FO_length - (n * FO_chunk); // don't go out of array bounds
          #pragma omp parallel for
          for (int icell = 0; icell < endFO; icell++) // cell index inside each chunk
          {
            int icell_glb = n * FO_chunk + icell; // global FO cell index

            // set freezeout info to local varibles
            double tau = tau_fo[icell_glb];         // longitudinal proper time
            double tau2 = tau * tau;                // useful expression
            if(DIMENSION == 3)
            {
              etaValues[0] = eta_fo[icell_glb];     // spacetime rapidity from surface file
            }
            double dat = dat_fo[icell_glb];         // covariant normal surface vector
            double dax = dax_fo[icell_glb];
            double day = day_fo[icell_glb];
            double dan = dan_fo[icell_glb];

            double T = T_fo[icell_glb];

            double ux = ux_fo[icell_glb];           // contravariant fluid velocity
            double uy = uy_fo[icell_glb];           // enforce normalization: u.u = 1
            double un = un_fo[icell_glb];
            double ut = sqrt(1.0 + ux*ux + uy*uy + tau2*un*un);

            double u0 = sqrt(1.0 + ux*ux + uy*uy);
            double zt = tau * un / u0;
            double zn = ut / (u0 * tau);            // z basis vector

            double pitt = pitt_fo[icell_glb];       // piperp^munu
            double pitx = pitx_fo[icell_glb];
            double pity = pity_fo[icell_glb];
            double pitn = pitn_fo[icell_glb];
            double pixx = pixx_fo[icell_glb];
            double pixy = pixy_fo[icell_glb];
            double pixn = pixn_fo[icell_glb];
            double piyy = piyy_fo[icell_glb];
            double piyn = piyn_fo[icell_glb];
            double pinn = pinn_fo[icell_glb];

            double bulkPi = bulkPi_fo[icell_glb];   // residual bulk pressure

            double Wx = Wx_fo[icell_glb];           // W^mu
            double Wy = Wy_fo[icell_glb];           // reinforce orthogonality
            double Wt = (ux * Wx  +  uy * Wy) * ut / (u0 * u0);
            double Wn = Wt * un / ut;

            double Lambda = Lambda_fo[icell_glb];   // anisotropic variables
            double aL = aL_fo[icell_glb];

            double c0 = c0_fo[icell_glb];           // 14-moment coefficients
            double c1 = c1_fo[icell_glb];
            double c2 = c2_fo[icell_glb];
            double c3 = c3_fo[icell_glb];
            double c4 = c4_fo[icell_glb];


            //now loop over all particle species and momenta
            for (int ipart = 0; ipart < number_of_chosen_particles; ipart++)
            {
              // set particle properties
              double mass = Mass[ipart];    // (GeV)
              double mass2 = mass * mass;
              double sign = Sign[ipart];
              double degeneracy = Degeneracy[ipart];

              for (int ipT = 0; ipT < pT_tab_length; ipT++)
              {
                // set transverse radial momentum and transverse mass (GeV)
                double pT = pTValues[ipT];
                double mT = sqrt(mass2 + pT * pT);
                // useful expression
                double mT_over_tau = mT / tau;

                for (int iphip = 0; iphip < phi_tab_length; iphip++)
                {
                  double px = pT * trig_phi_table[iphip][0]; //contravariant
                  double py = pT * trig_phi_table[iphip][1]; //contravariant

                  for (int iy = 0; iy < y_pts; iy++)
                  {
                    // all vector components are CONTRAVARIANT EXCEPT the surface normal vector dat, dax, day, dan, which are COVARIANT
                    double y = yValues[iy];

                    double pdotdsigma_f_eta_sum = 0.0;

                    // sum over eta
                    for (int ieta = 0; ieta < eta_pts; ieta++)
                    {
                      double eta = etaValues[ieta];
                      double eta_weight = etaWeights[ieta];

                      double pt = mT * cosh(y - eta);
                      double pn = mT_over_tau * sinh(y - eta);

                      // useful expression
                      double tau2_pn = tau2 * pn;

                      // momentum vector is contravariant, surface normal vector is COVARIANT
                      double pdotdsigma = pt * dat  +  px * dax  +  py * day  +  pn * dan;

                      //thermal equilibrium distributions - for viscous hydro
                      double pdotu = pt * ut  -  px * ux  -  py * uy  -  tau2_pn * un;    // u.p = LRF energy
                      double pdotz = pt * zt  -  tau2_pn * zn;                            // z.p = - LRF longitudinal momentum

                      double xiL = 1.0 / (aL * aL)  -  1.0;

                      #ifdef FORCE_F0
                      Lambda = T;
                      xiL = 0.0;
                      #endif

                      double Ea = sqrt(pdotu * pdotu  +  xiL * pdotz * pdotz);
                      double fa = 1.0 / ( exp(Ea / Lambda) + sign);

                      // residual viscous corrections
                      double fabar = 1.0 - sign*fa;

                      // residual shear correction:
                      double df_shear = 0.0;

                      if (INCLUDE_SHEAR_DELTAF)
                      {
                        // - (-z.p)p_{mu}.W^mu
                        double Wmu_pmu_pz = pdotz * (Wt * pt  -  Wx * px  -  Wy * py  -  Wn * tau2_pn);

                        // p_{mu.p_nu}.piperp^munu
                        double pimunu_pmu_pnu = pitt * pt * pt + pixx * px * px + piyy * py * py + pinn * tau2_pn * tau2_pn
                        + 2.0 * (-(pitx * px + pity * py) * pt + pixy * px * py + tau2_pn * (pixn * px + piyn * py - pitn * pt));

                        df_shear = c3 * Wmu_pmu_pz  +  c4 * pimunu_pmu_pnu;  // df / (feq*feqbar)
                      }

                      // residual bulk correction:
                      double df_bulk = 0.0;
                      if (INCLUDE_BULK_DELTAF) df_bulk = (c0 * mass2  +  c1 * pdotz * pdotz  +  c2 * pdotu * pdotu) * bulkPi;

                      double df = df_shear + df_bulk;

                      if (REGULATE_DELTAF)
                      {
                        double reg_df = max( -1.0, min( fabar * df, 1.0 ) );
                        pdotdsigma_f_eta_sum += (eta_weight * pdotdsigma * fa * (1.0 + reg_df));
                      }
                      else pdotdsigma_f_eta_sum += (eta_weight * pdotdsigma * fa * (1.0 + fabar * df));

                    } // ieta

                    long long int iSpectra = icell + endFO * (ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy)));

                    dN_pTdpTdphidy_all[iSpectra] = (prefactor * degeneracy * pdotdsigma_f_eta_sum);
                  } //iy
                } //iphip
              } //ipT
            } //ipart
          } //icell
          if(endFO != 0)
          {
            //now perform the reduction over cells
            #pragma omp parallel for collapse(4)
            for (int ipart = 0; ipart < npart; ipart++)
            {
              for (int ipT = 0; ipT < pT_tab_length; ipT++)
              {
                for (int iphip = 0; iphip < phi_tab_length; iphip++)
                {
                  for (int iy = 0; iy < y_pts; iy++)
                  {
                    //long long int is = ipart + (npart * ipT) + (npart * pT_tab_length * iphip) + (npart * pT_tab_length * phi_tab_length * iy);
                    long long int iS3D = ipart + npart * (ipT + pT_tab_length * (iphip + phi_tab_length * iy));
                    double dN_pTdpTdphidy_tmp = 0.0; //reduction variable
                    #pragma omp simd reduction(+:dN_pTdpTdphidy_tmp)
                    for (int icell = 0; icell < endFO; icell++)
                    {
                      //long long int iSpectra = icell + (endFO * ipart) + (endFO * npart * ipT) + (endFO * npart * pT_tab_length * iphip) + (endFO * npart * pT_tab_length * phi_tab_length * iy);
                      long long int iSpectra = icell + endFO * iS3D;
                      dN_pTdpTdphidy_tmp += dN_pTdpTdphidy_all[iSpectra];
                    }//icell
                    dN_pTdpTdphidy[iS3D] += dN_pTdpTdphidy_tmp; //sum over all chunks
                  }//iy
                }//iphip
              }//ipT
            }//ipart species
          } //if (endFO != 0 )
        }//n FO chunk

        //free memory
        free(dN_pTdpTdphidy_all);
      }