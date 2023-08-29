#!/bin/bash

### This script computes the heatwave indices under considering etccdi standards ###

path1=/media/kenz/1B8D1A637BBA134B/CHIRTS

#chmod u+w /media/kenz/1B8D1A637BBA134B/CHIRTS/
# chmod u+w /media/kenz/1B8D1A637BBA134B/CHIRTS/Tmean/CHIRTS_Tmean_WA_1983_2016.nc

tmax="${path1}/Tmax/chirts.Tmax.1983-2016.WA.days_p25.nc"
tmin="${path1}/Tmin/chirts.Tmin.1983.2016.WA.days_p25.nc"

tmean="${path1}/Tmean/chirts.Tmean.1983.2016.WA.days_p25.nc"

#rm $tmean
#cdo -O ensmean "$tmax" "$tmin" "$tmean"



output2="${path1}/Tmean/chirts.Tmean90.1983.2016.WA.days_p25.nc"

## computing EHI rollings ###
cdo -O runmean,3 $tmean "${path1}/Tmean/r3.nc"
cdo -O runmean,30 $tmean "${path1}/Tmean/r30.nc"

## Calculate EHIaccl
cdo -O sub "${path1}/Tmean/r3.nc" "${path1}/Tmean/r30.nc" "${path1}/EHIaccl.nc"

# calculate the 10 day rolling percentile ###
 ## cdo runpctl,p,n ifile ofile ###
 ###########################################################

for dt in Tmax Tmin Tmean
do
#	rm ${path1}/${dt}/CHIRTS_${dt}90_WA_1983_2016.nc
    input_file=${path1}/${dt}/chirts.${dt}.1983.2016.WA.days_p25.nc
    output_file=${path1}/${dt}/chirts.${dt}90.1983.2016.WA.days_p25.nc
    
    cdo runpctl,90,10 $input_file $output_file

done

## Calculate EHI_sig
cdo -O sub "${path1}/Tmean/r3.nc" "$output2" "${path1}/EHFsig.nc"

cdo -ifthen -gec,1 "${path1}/EHIaccl.nc" "${path1}/EHIaccl.nc" ofile_ge0.nc

cdo mul ofile_ge0.nc "${path1}/EHFsig.nc" EHF.nc

rm ofile_ge0.nc r3.nc r30.nc "${path1}/EHIaccl.nc" "${path1}/EHFsig.nc" 

# -mul "${path1}/EHIaccl.nc" "${path1}/EHIsig.nc" -f nc "${path1}/EHF.nc"
 #### Call python3 to plot the graphs

 #python3 output_file


exit

