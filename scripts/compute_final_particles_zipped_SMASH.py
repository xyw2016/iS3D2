import numpy as np
import pandas as pd
import math
import sys

#species dependent info
#pion 211
pi_pT  = []
pi_y = []

#kaon 321
k_pT  = []
k_y = []

#proton 2212
p_pT  = []
p_y = []

#antiproton -2212
pm_pT  = []
pm_y = []


# Path to particle list file
particle_list_file = 'particle_list'

with open(particle_list_file, 'r') as file:
    lines = file.readlines()

    nsamples = 0

    # Loop over lines in the file
    for line in lines:
        line_split = line.strip().split()

        # Check if the line contains the count of particles for a new event
        if len(line_split) == 1:
            # Process the start of a new event
            num_particles = int(line_split[0])
            current_particle = 0
            nsamples  += 1
        else:
            # Extract particle information
            if current_particle < num_particles:

                mcid = int(line_split[0])
                energy = float(line_split[10])
                px = float(line_split[11])
                py = float(line_split[12])
                pz = float(line_split[13])

                pT = math.sqrt(px**2 + py**2)

                if ( mcid == 211 ):
                    pi_pT.append( pT )
                    arg = (energy + pz) / (energy - pz)
                    if (pz == 0):
                        y = 0
                    else:
                        y = 0.5 * math.log( abs(arg) )
                    pi_y.append( y )

                elif ( mcid == 321 ):
                    k_pT.append( pT )
                    arg = (energy + pz) / (energy - pz)
                    if (pz == 0):
                        y = 0
                    else:
                        y = 0.5 * math.log( abs(arg) )
                    k_y.append( y )

                elif ( mcid == 2212 ):
                    p_pT.append( pT )
                    arg = (energy + pz) / (energy - pz)
                    if (pz == 0):
                        y = 0
                    else:
                        y = 0.5 * math.log( abs(arg) )
                    p_y.append( y )

                elif ( mcid == -2212 ):
                    pm_pT.append( pT )
                    arg = (energy + pz) / (energy - pz)
                    if (pz == 0):
                        y = 0
                    else:
                        y = 0.5 * math.log( abs(arg) )
                    pm_y.append( y )

                current_particle += 1


# Define parameters
Ymax = 4.0  # Maximum rapidity value
dy = 0.5    # Bin width for binning the range [0, Ymax]

# Initialize lists to store mean yields for different ranges
mean_yield_pi_rap_list = []
mean_yield_k_rap_list = []
mean_yield_p_rap_list = []
mean_yield_pm_rap_list = []

# Loop over bins with width dy
for ymin in range(0, int(Ymax / dy)):
    ymin = ymin * dy
    ymax = ymin + dy

    print("ymin = {0} and ymax = {1}".format(ymin, ymax))

    # Initialize lists to store pT values within the current range
    pi_pT_rap = []
    k_pT_rap = []
    p_pT_rap = []
    pm_pT_rap = []

    # Filter pT values within the current range [ymin, ymax]
    for i in range(len(pi_y)):
        if ymin <= pi_y[i] <= ymax:
            pi_pT_rap.append(pi_pT[i])

    for i in range(len(k_y)):
        if ymin <= k_y[i] <= ymax:
            k_pT_rap.append(k_pT[i])

    for i in range(len(p_y)):
        if ymin <= p_y[i] <= ymax:
            p_pT_rap.append(p_pT[i])

    for i in range(len(pm_y)):
        if ymin <= pm_y[i] <= ymax:
            pm_pT_rap.append(pm_pT[i])

    # Calculate mean yields for the current range and append to the respective lists
    mean_yield_pi_rap_list.append(len(pi_pT_rap)/dy/nsamples)
    mean_yield_k_rap_list.append(len(k_pT_rap)/dy/nsamples)
    mean_yield_p_rap_list.append(len(p_pT_rap)/dy/nsamples)
    mean_yield_pm_rap_list.append(len(pm_pT_rap)/dy/nsamples)

# Output mean yield lists for different ranges
print("Mean yield for pi for each range:", mean_yield_pi_rap_list)
print("Mean yield for k for each range:", mean_yield_k_rap_list)
print("Mean yield for p for each range:", mean_yield_p_rap_list)
print("Mean yield for pm for each range:", mean_yield_pm_rap_list)


#write to files 
# file = open("mean_vals.dat","w")
# file.write("pi : ( dN/dy, <pT>, sigma_pT ) = ( " + str(mean_yield_pi_midrap/dy/nsamples) + ", " + str(mean_pT_pi_midrap) + ", " + str(var_pT_pi_midrap) + " ) \n" )
# file.write("k  : ( dN/dy, <pT>, sigma_pT ) = ( " + str(mean_yield_k_midrap/dy/nsamples) + ", " +  str(mean_pT_k_midrap) + ", " + str(var_pT_k_midrap)  + " ) \n" )
# file.write("p  : ( dN/dy, <pT>, sigma_pT ) = ( " + str(mean_yield_p_midrap/dy/nsamples) + ", " +  str(mean_pT_p_midrap) + ", " + str(var_pT_p_midrap)  + " ) \n" )
# file.close()


