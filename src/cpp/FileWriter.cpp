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


  // try combining common spectra file functions to reduce clutter...
  // and also move to a separate source file

  void EmissionFunctionArray::write_dN_pTdpTdphidy_toFile(int *MCID)
  {
    printf("Writing thermal spectra to file...\n");

    char filename[255] = "";

    for(long ipart  = 0; ipart < number_of_chosen_particles; ipart++)
    {
      sprintf(filename, "results/continuous/dN_pTdpTdphidy_%d.dat", MCID[ipart]);
      ofstream spectra(filename, ios_base::out);

      spectra << "y" << "\t" << "phip" << "\t" << "pT" << "\t" << "dN_pTdpTdphidy" << "\n";

      for(long iy = 0; iy < y_tab_length; iy++)
      {
        double y = 0.0;
        if(DIMENSION == 3) y = y_tab->get(1,iy + 1);

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);

          for(long ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            spectra << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << dN_pTdpTdphidy[iS3D] << "\n";
          } //ipT
          spectra << "\n";
        } //iphip
      } //iy
      spectra.close();
    }
  }



  void EmissionFunctionArray::write_dN_dphidy_toFile(int *MCID)
  {
    printf("Writing thermal dN_dphidy to file...\n");
    char filename[255] = "";

    // write a separate file for each species
    for(long ipart  = 0; ipart < number_of_chosen_particles; ipart++)
    {
      sprintf(filename, "results/continuous/dN_dphidy_%d.dat", MCID[ipart]);
      ofstream spectra(filename, ios_base::out);

      for(long iy = 0; iy < y_tab_length; iy++)
      {
        double y = 0.0;
        if(DIMENSION == 3) y = y_tab->get(1,iy + 1);

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          double dN_dphidy = 0.0;

          for(int ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT_weight = pT_tab->get(2, ipT + 1);

            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            dN_dphidy += pT_weight * dN_pTdpTdphidy[iS3D];
          }
          spectra << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << dN_dphidy << "\n";
        }
        if(iy < y_tab_length - 1) spectra << "\n";
      }
      spectra.close();
    }
  }


  void EmissionFunctionArray::write_dN_twopipTdpTdy_toFile(int *MCID)
  {
    printf("Writing thermal dN_twopipTdpTdy to file...\n");

    char filename[255] = "";

    for(long ipart  = 0; ipart < number_of_chosen_particles; ipart++)
    {
      sprintf(filename, "results/continuous/dN_2pipTdpTdy_%d.dat", MCID[ipart]);
      ofstream spectra(filename, ios_base::out);

      for(long iy = 0; iy < y_tab_length; iy++)
      {
        double y = 0.0;

        if(DIMENSION == 3) y = y_tab->get(1, iy + 1);

        for(long ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pT_tab->get(1, ipT + 1);

          double dN_twopipTdpTdy = 0.0;

          for(long iphip = 0; iphip < phi_tab_length; iphip++)
          {
            double phip_weight = phi_tab->get(2, iphip + 1);

            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            dN_twopipTdpTdy += phip_weight * dN_pTdpTdphidy[iS3D] / two_pi;
          }

          spectra << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT << "\t" << dN_twopipTdpTdy << "\n";
        }

        if(iy < y_tab_length - 1) spectra << "\n";
      }

      spectra.close();
    }
  }


  void EmissionFunctionArray::write_dN_dy_toFile(int *MCID)
  {
    printf("Writing thermal dN_dy to file...\n");
    char filename[255] = "";

    //write a separate file for each species
    for(long ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      sprintf(filename, "results/continuous/dN_dy_%d.dat", MCID[ipart]);
      ofstream spectra(filename, ios_base::out);

      for(long iy = 0; iy < y_tab_length; iy++)
      {
        double y = 0.0;
        if(DIMENSION == 3) y = y_tab->get(1, iy + 1);

        double dN_dy = 0.0;

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip_weight = phi_tab->get(2, iphip + 1);

          for(long ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT_weight = pT_tab->get(2, ipT + 1);
            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            dN_dy += phip_weight * pT_weight * dN_pTdpTdphidy[iS3D];
          }
        }
        spectra << setw(5) << setprecision(8) << y << "\t" << dN_dy << endl;
      }
      spectra.close();
    }
  }



  void EmissionFunctionArray::write_continuous_vn_toFile(int *MCID)
  {
    printf("Writing continuous vn(pT,y) to file (for testing vn's)...\n");
    char filename[255] = "";

    const complex<double> I(0.0,1.0);   // imaginary i

    const int k_max = 7;                // v_n = {v_1, ..., v_7}

    // write a separate file for each species
    for(long ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      sprintf(filename, "results/continuous/vn_%d.dat", MCID[ipart]);
      ofstream vn_File(filename, ios_base::out);

      for(long iy = 0; iy < y_tab_length; iy++)
      {
        double y = 0.0;
        if(DIMENSION == 3) y = y_tab->get(1, iy + 1);

        for(long ipT = 0; ipT < pT_tab_length; ipT++)
        {
          double pT = pT_tab->get(1, ipT + 1);

          double Vn_real_numerator[k_max];
          double Vn_imag_numerator[k_max];

          for(int k = 0; k < k_max; k++)
          {
            Vn_real_numerator[k] = 0.0;
            Vn_imag_numerator[k] = 0.0;
          }

          double vn_denominator = 0.0;

          for(long iphip = 0; iphip < phi_tab_length; iphip++)
          {
            double phip = phi_tab->get(1, iphip + 1);
            double phip_weight = phi_tab->get(2, iphip + 1);

            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));

            for(int k = 0; k < k_max; k++)
            {
              Vn_real_numerator[k] += cos(((double)k + 1.0) * phip) * phip_weight * dN_pTdpTdphidy[iS3D];
              Vn_imag_numerator[k] += sin(((double)k + 1.0) * phip) * phip_weight * dN_pTdpTdphidy[iS3D];
            }
            vn_denominator += phip_weight * dN_pTdpTdphidy[iS3D];

          } //iphip

          vn_File << scientific <<  setw(5) << setprecision(8) << y << "\t" << pT;

          for(int k = 0; k < k_max; k++)
          {
            double vn = abs(Vn_real_numerator[k]  +  I * Vn_imag_numerator[k]) / vn_denominator;

            if(vn_denominator < 1.e-15) vn = 0.0;

            vn_File << "\t" << vn;
          }

          vn_File << "\n";

        } //ipT

        vn_File << "\n";

      } //iy

      vn_File.close();

    }

  }


  void EmissionFunctionArray::write_polzn_vector_toFile()
  {
    printf("Writing polarization vector to file...\n");

    char filename_t[255] = "";
    char filename_x[255] = "";
    char filename_y[255] = "";
    char filename_n[255] = "";
    sprintf(filename_t, "results/St.dat");
    sprintf(filename_x, "results/Sx.dat");
    sprintf(filename_y, "results/Sy.dat");
    sprintf(filename_n, "results/Sn.dat");
    ofstream StFile(filename_t, ios_base::out);
    ofstream SxFile(filename_x, ios_base::out);
    ofstream SyFile(filename_y, ios_base::out);
    ofstream SnFile(filename_n, ios_base::out);

    for(long ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      for(long iy = 0; iy < y_tab_length; iy++)
      {
        double y = 0.0;
        if(DIMENSION == 3) y = y_tab->get(1,iy + 1);

        for(long iphip = 0; iphip < phi_tab_length; iphip++)
        {
          double phip = phi_tab->get(1,iphip + 1);
          for(long ipT = 0; ipT < pT_tab_length; ipT++)
          {
            double pT = pT_tab->get(1,ipT + 1);
            long iS3D = iy  +  y_tab_length * (iphip  +  phi_tab_length * (ipT  +  pT_tab_length * ipart));
            StFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (St[iS3D] / Snorm[iS3D]) << "\n";
            SxFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (Sx[iS3D] / Snorm[iS3D]) << "\n";
            SyFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (Sy[iS3D] / Snorm[iS3D]) << "\n";
            SnFile << scientific <<  setw(5) << setprecision(8) << y << "\t" << phip << "\t" << pT << "\t" << (Sn[iS3D] / Snorm[iS3D]) << "\n";

          } //ipT
          StFile << "\n";
          SxFile << "\n";
          SyFile << "\n";
          SnFile << "\n";
        } //iphip
      } //iy
    }//ipart
    StFile.close();
    SxFile.close();
    SyFile.close();
    SnFile.close();
  }


  void EmissionFunctionArray::write_particle_list_toFile()
  {
    printf("Writing sampled particles list to file...\n");

    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      char filename[255] = "";
      sprintf(filename, "results/particle_list_%d.dat", ievent + 1);

      //ofstream spectraFile(filename, ios_base::app);
      ofstream spectraFile(filename, ios_base::out);

      int num_particles = particle_event_list[ievent].size();

      //write the header
      spectraFile << "mcid" << "," << "tau" << "," << "x" << "," << "y" << "," << "eta" << "," << "E" << "," << "px" << "," << "py" << "," << "pz" << "\n";
      for (int ipart = 0; ipart < num_particles; ipart++)
      {
        int mcid = particle_event_list[ievent][ipart].mcID;
        double tau = particle_event_list[ievent][ipart].tau;
        double x = particle_event_list[ievent][ipart].x;
        double y = particle_event_list[ievent][ipart].y;
        double eta = particle_event_list[ievent][ipart].eta;
        double E = particle_event_list[ievent][ipart].E;
        double px = particle_event_list[ievent][ipart].px;
        double py = particle_event_list[ievent][ipart].py;
        double pz = particle_event_list[ievent][ipart].pz;
        spectraFile << scientific <<  setw(5) << setprecision(8) << mcid << "," << tau << "," << x << "," << y << "," << eta << "," << E << "," << px << "," << py << "," << pz << "\n";
      }//ipart
      spectraFile.close();
    } // ievent
  }


  //write particle list in oscar format for UrQMD/SMASH afterburner
  void EmissionFunctionArray::write_particle_list_OSC()
  {
    printf("Writing sampled particles list to OSCAR File...\n");

    char filename[255] = "";
    sprintf(filename, "results/particle_list_osc.dat");
    ofstream spectraFile(filename, ios_base::out);

    char line_buffer[500];

    // OSCAR header file;
    std::string OSCAR_header_filename;
    OSCAR_header_filename = "tables/OSCAR_header.txt";
    ifstream oscar_header(OSCAR_header_filename.c_str());
    if (!oscar_header.is_open()) {
      cout << "Error: OSCAR header file " << OSCAR_header_filename << " not found." << endl;
    } else {
      while (!oscar_header.eof()) {
        oscar_header.getline(line_buffer, 500);
        spectraFile << line_buffer << endl;
      }
      oscar_header.close();
    }
    
    for(int ievent = 0; ievent < Nevents; ievent++)
    {
      
      int num_particles = particle_event_list[ievent].size();

      //note only write events to file with at least one particle, else urqmd-afterburner will crash
      if (num_particles > 0){
        //write the header
        //spectraFile << "#" << " " << num_particles << "\n";
        //spectraFile << "n pid px py pz E m x y z t" << "\n";
        spectraFile << setw(10) << ievent << "  " 
                      << setw(10) << num_particles << "  " 
                      << setw(8) << 0.0 << "  " << setw(8) << 0.0 << endl;
        for (int ipart = 0; ipart < num_particles; ipart++)
        {
          int mcid = particle_event_list[ievent][ipart].mcID;
          double x = particle_event_list[ievent][ipart].x;
          double y = particle_event_list[ievent][ipart].y;
          double t = particle_event_list[ievent][ipart].t;
          double z = particle_event_list[ievent][ipart].z;

          double m  = particle_event_list[ievent][ipart].mass;
          double E  = particle_event_list[ievent][ipart].E;
          double px = particle_event_list[ievent][ipart].px;
          double py = particle_event_list[ievent][ipart].py;
          double pz = particle_event_list[ievent][ipart].pz;
          //spectraFile << ipart << " " << mcid << " " << scientific <<  setw(5) << setprecision(16) << px << " " << py << " " << pz << " " 
          //            << E << " " << m << " " << x << " " << y << " " << z << " " << t << "\n";
          spectraFile << setw(10) << ipart + 1 << "  " << setw(10) << mcid << "  ";
          sprintf(line_buffer, "%24.16e  %24.16e  %24.16e  %24.16e  %24.16e  %24.16e  %24.16e  %24.16e  %24.16e", px, py, pz, E, m, x, y, z, t);
          spectraFile << line_buffer << endl;
        }//ipart
      }
    } // ievent

    spectraFile.close();
  }


  // can I combine the sampled functions?

  void EmissionFunctionArray::write_sampled_dN_dy_to_file_test(int * MCID)
  {
    printf("Writing event-averaged dN/dy of each species to file...\n");

    // set up the y midpoint-grid (midpoints of each bin)
    double y_mid[Y_BINS];
    for(int iy = 0; iy < Y_BINS; iy++) y_mid[iy] = -Y_CUT + Y_WIDTH * ((double)iy + 0.5);

    // write dN/dy for each species
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char file[255] = "";
      char file2[255] = "";

      sprintf(file, "results/sampled/dN_dy/dN_dy_%d_test.dat", MCID[ipart]);
      sprintf(file2, "results/sampled/dN_dy/dN_dy_%d_average_test.dat", MCID[ipart]);
      ofstream dN_dy(file, ios_base::out);
      ofstream dN_dy_avg(file2, ios_base::out);

      double average = 0.0;

      for(int iy = 0; iy < Y_BINS; iy++)
      {
        average += dN_dy_count[ipart][iy];

        dN_dy << setprecision(6) << y_mid[iy] << "\t" << dN_dy_count[ipart][iy] / (Y_WIDTH * (double)Nevents) << endl;

      } // iy

      dN_dy_avg << setprecision(6) << average / (2.0 * Y_CUT * (double)Nevents) << endl;

      dN_dy.close();
      dN_dy_avg.close();

    } // ipart

    free_2D(dN_dy_count, number_of_chosen_particles);
  }


  void EmissionFunctionArray::write_sampled_dN_deta_to_file_test(int * MCID)
  {
    printf("Writing event-averaged dN/deta of each species to file...\n");

    double eta_mid[ETA_BINS];
    for(int ieta = 0; ieta < ETA_BINS; ieta++) eta_mid[ieta] = -ETA_CUT + ETA_WIDTH * ((double)ieta + 0.5);

    // write dN/deta for each species
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char file[255] = "";
      sprintf(file, "results/sampled/dN_deta/dN_deta_%d_test.dat", MCID[ipart]);
      ofstream dN_deta(file, ios_base::out);

      for(int ieta = 0; ieta < ETA_BINS; ieta++)
      {
        dN_deta << setprecision(6) << eta_mid[ieta] << "\t" << dN_deta_count[ipart][ieta] / (ETA_WIDTH * (double)Nevents) << endl;
      }
      dN_deta.close();

    } // ipart

    free_2D(dN_deta_count, number_of_chosen_particles);
  }


  void EmissionFunctionArray::write_sampled_dN_2pipTdpTdy_to_file_test(int * MCID)
  {
    printf("Writing event-averaged dN/2pipTdpTdy of each species to file...\n");

    double pT_mid[PT_BINS];
    for(int ipT = 0; ipT < PT_BINS; ipT++) pT_mid[ipT] = PT_MIN  +  PT_WIDTH * ((double)ipT + 0.5);

    double y_mid[Y_BINS];
    for(int iy = 0; iy < Y_BINS; iy++) y_mid[iy] = -Y_CUT + Y_WIDTH * ((double)iy + 0.5);

    // write dN/2pipTdpTdy for each species
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char file[255] = "";
      sprintf(file, "results/sampled/dN_2pipTdpTdy/dN_2pipTdpTdy_%d_test.dat", MCID[ipart]);
      ofstream dN_2pipTdpTdy(file, ios_base::out);

      for(int iy = 0; iy < Y_BINS; iy++)
      {
        for(int ipT = 0; ipT < PT_BINS; ipT++)
        {
          dN_2pipTdpTdy << setprecision(6) << scientific << y_mid[iy] << "\t" << setprecision(6) << scientific << pT_mid[ipT] << "\t" 
          << dN_2pipTdpTdy_count[ipart][iy][ipT] / (two_pi * Y_WIDTH * PT_WIDTH * pT_mid[ipT] * (double)Nevents) 
          << "\n";
        }

        dN_2pipTdpTdy << "\n";
      }

      dN_2pipTdpTdy.close();

    } // ipart

    free_3D(dN_2pipTdpTdy_count, number_of_chosen_particles, Y_BINS);
  }


   void EmissionFunctionArray::write_sampled_dN_dphipdy_to_file_test(int * MCID)
  {
    printf("Writing event-averaged dN/dphipdy of each species to file...\n");

    double phip_mid[PHIP_BINS];
    for(int iphip = 0; iphip < PHIP_BINS; iphip++) phip_mid[iphip] = PHIP_WIDTH * ((double)iphip + 0.5);

    double y_mid[Y_BINS];
    for(int iy = 0; iy < Y_BINS; iy++) y_mid[iy] = -Y_CUT + Y_WIDTH * ((double)iy + 0.5);

    // write dN/2pipTdpTdy for each species
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char file[255] = "";
      sprintf(file, "results/sampled/dN_dphipdy/dN_dphipdy_%d_test.dat", MCID[ipart]);
      ofstream dN_dphipdy(file, ios_base::out);

      for(int iy = 0; iy < Y_BINS; iy++)
      {
        for(int iphip = 0; iphip < PHIP_BINS; iphip++)
        {
          dN_dphipdy << setprecision(6) << scientific << y_mid[iy] << "\t"  << setprecision(6) << scientific << phip_mid[iphip] << "\t" 
          << dN_dphipdy_count[ipart][iy][iphip] / (Y_WIDTH * PHIP_WIDTH * (double)Nevents) << "\n"; 
          // note 2*y_cut is the total y range
        }

        dN_dphipdy << "\n";
      }
      
      dN_dphipdy.close();

    } // ipart

    free_3D(dN_dphipdy_count, number_of_chosen_particles, Y_BINS);
  }



  void EmissionFunctionArray::write_sampled_vn_to_file_test(int * MCID)
  {
    printf("Writing event-averaged vn(pT) of each species to file...\n");

    const complex<double> I(0,1.0); // imaginary i

    double pT_mid[PT_BINS];
    for(int ipT = 0; ipT < PT_BINS; ipT++) pT_mid[ipT] = PT_MIN +  PT_WIDTH * ((double)ipT + 0.5);

    double y_mid[Y_BINS];
    for(int iy = 0; iy < Y_BINS; iy++) y_mid[iy] = -Y_CUT + Y_WIDTH * ((double)iy + 0.5);

    // write vn(pT) for each species
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char file[255] = "";
      sprintf(file, "results/sampled/vn/vn_%d_test.dat", MCID[ipart]);
      ofstream vn(file, ios_base::out);

      for(int iy = 0; iy < Y_BINS; iy++)
      {

        for(int ipT = 0; ipT < PT_BINS; ipT++)
        {
          vn << setprecision(6) << scientific << y_mid[iy] << "\t" << setprecision(6) << scientific << pT_mid[ipT] << "\t";

          for(int k = 0; k < K_MAX; k++)
          {
            double vn_abs = abs(vn_real_count[k][ipart][iy][ipT]  +  I * vn_imag_count[k][ipart][iy][ipT]) / pT_count[ipart][ipT];
            if(std::isnan(vn_abs) || std::isinf(vn_abs)) vn_abs = 0.0;

            vn << setprecision(6) << scientific << vn_abs << "\t";
          }

          vn << "\n";

        } // ipT

        vn << "\n";
      }
      
      vn.close();
    } // ipart

    free_2D(pT_count, number_of_chosen_particles);
    free_4D(vn_real_count, K_MAX, number_of_chosen_particles, Y_BINS);
    free_4D(vn_imag_count, K_MAX, number_of_chosen_particles, Y_BINS);
  }


 void EmissionFunctionArray::write_sampled_dN_dX_to_file_test(int * MCID)
  {
    printf("Writing event-averaged boost invariant spacetime distributions dN_dX of each species to file...\n");

    // dX = taudtaudeta or 2pirdrdeta (only have boost invariance in mind so deta = dy)

    double tau_mid[TAU_BINS];
    for(int itau = 0; itau < TAU_BINS; itau++) tau_mid[itau] = TAU_MIN + TAU_WIDTH * ((double)itau + 0.5);

    double r_mid[R_BINS];
    for(int ir = 0; ir < R_BINS; ir++) r_mid[ir] = R_MIN + R_WIDTH * ((double)ir + 0.5);

    double phi_mid[PHIP_BINS];
    for(int iphi = 0; iphi < PHIP_BINS; iphi++) phi_mid[iphi] = PHIP_WIDTH * ((double)iphi + 0.5);

    // now event-average dN_dXdy and normalize to dNdy and write them to file
    for(int ipart = 0; ipart < number_of_chosen_particles; ipart++)
    {
      char file_time[255] = "";
      char file_radial[255] = "";
      char file_azimuthal[255] = "";

      sprintf(file_time, "results/sampled/dN_taudtaudy/dN_taudtaudy_%d_test.dat", MCID[ipart]);
      sprintf(file_radial, "results/sampled/dN_2pirdrdy/dN_2pirdrdy_%d_test.dat", MCID[ipart]);
      sprintf(file_azimuthal, "results/sampled/dN_dphisdy/dN_dphisdy_%d_test.dat", MCID[ipart]);

      ofstream dN_taudtaudy(file_time, ios_base::out);
      ofstream dN_twopirdrdy(file_radial, ios_base::out);
      ofstream dN_dphisdy(file_azimuthal, ios_base::out);

      // normalize spacetime distributions by the binwidth, jacobian factor, events and rapidity cut range
      for(int ir = 0; ir < R_BINS; ir++)
      {
        dN_twopirdrdy << setprecision(6) << scientific << r_mid[ir] << "\t" << dN_twopirdrdy_count[ipart][ir] / (two_pi * r_mid[ir] * R_WIDTH * (double)Nevents * 2.0 * Y_CUT) << "\n";
      }

      for(int itau = 0; itau < TAU_BINS; itau++)
      {
        dN_taudtaudy << setprecision(6) << scientific << tau_mid[itau] << "\t" << dN_taudtaudy_count[ipart][itau] / (tau_mid[itau] * TAU_WIDTH * (double)Nevents * 2.0 * Y_CUT) << "\n";
      }

      for(int iphi = 0; iphi < PHIP_BINS; iphi++)
      {
        dN_dphisdy << setprecision(6) << scientific << phi_mid[iphi] << "\t" << dN_dphisdy_count[ipart][iphi] / (PHIP_WIDTH * (double)Nevents * 2.0 * Y_CUT) << "\n";
      }


      dN_taudtaudy.close();
      dN_twopirdrdy.close();
      dN_dphisdy.close();
    } // ipart

    free_2D(dN_taudtaudy_count, number_of_chosen_particles);
    free_2D(dN_twopirdrdy_count, number_of_chosen_particles);
    free_2D(dN_dphisdy_count, number_of_chosen_particles);
  }

